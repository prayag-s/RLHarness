"""Durable job records: what the studio is doing, and what it was doing when it died.

Synthesis and training take minutes and used to exist only as in-process coroutines. That made the
API un-scalable (one worker), unrecoverable (a restart lost the work silently) and un-cancellable.
A job row is claimed, updated as it moves through stages, and finished — so the API can answer
"what happened to my run?" from any process, and a restart converts interrupted work into an
explicit, retryable failure instead of a spinner that never resolves.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl_ide.state.sqlite_utils import bounded_connection

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL = (SUCCEEDED, FAILED, CANCELLED)

# Stages a chat job reports, in order. The UI maps them to human text.
STAGES = ("queued", "intake", "env_synthesis", "verification", "arena", "training", "ready")


class JobCancelled(RuntimeError):
    """Raised inside a running job when the user asked for it to stop at a stage boundary."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """SQLite-backed job queue living beside the conversations it belongs to."""

    def __init__(self, db_path: Path | str = "runs/conversations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> Any:
        """Context manager yielding a connection that is always closed (see ``sqlite_utils``)."""
        return bounded_connection(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    owner_id TEXT NOT NULL DEFAULT 'local',
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT 'queued',
                    detail TEXT,
                    payload TEXT,
                    result TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind, status, created_at);")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_conversation ON jobs(conversation_id, created_at);"
            )
            conn.commit()

    # ------------------------------------------------------------------ writes
    def create(
        self,
        *,
        kind: str,
        conversation_id: str | None = None,
        owner_id: str = "local",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = _now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, conversation_id, owner_id, kind, status, stage, payload,
                                  created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    job_id,
                    conversation_id,
                    owner_id,
                    kind,
                    QUEUED,
                    "queued",
                    json.dumps(payload or {}, default=str),
                    now,
                    now,
                ),
            )
            conn.commit()
        return self.get(job_id)  # type: ignore[return-value]

    def update(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "stage", "detail", "result", "error", "cancel_requested"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        if "result" in updates and not isinstance(updates["result"], str):
            updates["result"] = json.dumps(updates["result"], default=str)
        sets = ", ".join(f"{key} = ?" for key in updates)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE jobs SET {sets}, updated_at = ? WHERE job_id = ?;",
                [*updates.values(), _now(), job_id],
            )
            conn.commit()

    def set_stage(self, job_id: str, stage: str, detail: str | None = None) -> None:
        self.update(job_id, stage=stage, detail=detail, status=RUNNING)

    def finish(
        self,
        job_id: str,
        *,
        status: str = SUCCEEDED,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.update(job_id, status=status, stage=status, result=result, error=error)

    def request_cancel(self, job_id: str) -> bool:
        """Flags a job for cooperative cancellation; returns False when it already finished."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? "
                "WHERE job_id = ? AND status IN (?, ?);",
                (_now(), job_id, QUEUED, RUNNING),
            )
            conn.commit()
            return cur.rowcount > 0

    def claim_next(self, kind: str) -> dict[str, Any] | None:
        """Atomically takes the oldest queued job of ``kind``; safe with several API processes."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT * FROM jobs WHERE kind = ? AND status = ? ORDER BY created_at LIMIT 1;",
                (kind, QUEUED),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            cur = conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, updated_at = ? WHERE job_id = ? AND status = ?;",
                (RUNNING, "starting", _now(), row["job_id"], QUEUED),
            )
            conn.commit()
            if cur.rowcount == 0:  # another process won the race
                return None
        return self.get(row["job_id"])

    # ------------------------------------------------------------------- reads
    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,)).fetchone()
        return self._decode(row) if row else None

    def list_jobs(
        self,
        *,
        conversation_id: str | None = None,
        owner_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if owner_id:
            clauses.append("owner_id = ?")
            params.append(owner_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ?;",
                [*params, int(limit)],
            ).fetchall()
        return [self._decode(row) for row in rows]

    def active_job(self, conversation_id: str, kind: str | None = None) -> dict[str, Any] | None:
        clauses = ["conversation_id = ?", "status IN (?, ?)"]
        params: list[Any] = [conversation_id, QUEUED, RUNNING]
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT 1;",
                params,
            ).fetchone()
        return self._decode(row) if row else None

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT cancel_requested FROM jobs WHERE job_id = ?;", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def reconcile_interrupted(self) -> int:
        """Turns work a dead process left behind into an explicit, retryable failure."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, error = ?, updated_at = ? "
                "WHERE status IN (?, ?);",
                (
                    FAILED,
                    FAILED,
                    "Interrupted: the API restarted while this job was running. "
                    "Nothing was charged for the unfinished step — retry it.",
                    _now(),
                    QUEUED,
                    RUNNING,
                ),
            )
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------ helper
    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        job = dict(row)
        for key in ("payload", "result"):
            if job.get(key):
                try:
                    job[key] = json.loads(job[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        job["cancel_requested"] = bool(job.get("cancel_requested"))
        job["retryable"] = job["status"] == FAILED
        return job


_global_job_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _global_job_store
    if _global_job_store is None:
        _global_job_store = JobStore()
    return _global_job_store
