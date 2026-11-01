"""Durable, sequenced transcript for one conversation — the single place work is narrated.

The studio used to narrate in two panes and three components, with no actor, no turn id and no
sequence number; only user/assistant rows survived a reload, and everything else was relabelled.
This store is the authority for the unified thread: append-only, monotonically sequenced, grouped
into turns, and replayed verbatim on load.

A turn has exactly one owner. ``begin_turn`` is a compare-and-set on the conversation's state, so a
second caller gets ``TurnInProgressError`` instead of two parties writing into the same turn.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from rl_ide.state.sqlite_utils import bounded_connection

ACTORS = ("user", "studio", "trainer", "coach", "gatekeeper", "system")

#: ``running`` is the only state that refuses a new turn.
TURN_STATES = ("idle", "running", "decision_required", "blocked")

#: How long a turn may stay open before a restart is allowed to reclaim it.
STALE_TURN_SECONDS = 6 * 60 * 60


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TurnInProgressError(RuntimeError):
    """Raised when a turn is requested while another party still owns the conversation."""

    def __init__(self, conversation_id: str, active_turn_id: str | None):
        super().__init__(
            f"Conversation '{conversation_id}' is mid-turn (turn {active_turn_id or 'unknown'})."
        )
        self.conversation_id = conversation_id
        self.active_turn_id = active_turn_id or ""


def _normalize(entry: dict[str, Any], fallback_turn_id: str | None, seq: int) -> dict[str, Any]:
    actor = str(entry.get("actor") or "system")
    if actor not in ACTORS:
        actor = "system"
    kind = str(entry.get("kind") or "result")
    status = str(entry.get("status") or "success")
    anchor = entry.get("anchor")
    actions = entry.get("actions")
    data = entry.get("data")
    return {
        "seq": seq,
        "turn_id": str(entry.get("turn_id") or fallback_turn_id or "turn_legacy"),
        "actor": actor,
        "kind": kind,
        "status": status,
        "title": str(entry.get("title") or ""),
        "body": str(entry.get("body") or ""),
        "at": str(entry.get("at") or utc_now_iso()),
        "anchor": json.dumps(anchor) if anchor else None,
        "actions": json.dumps(actions) if actions else None,
        "data": json.dumps(data) if data else None,
    }


def _row_to_entry(row: Any) -> dict[str, Any]:
    d = dict(row)
    d["anchor"] = json.loads(d["anchor"]) if d.get("anchor") else None
    d["actions"] = json.loads(d["actions"]) if d.get("actions") else None
    d["data"] = json.loads(d["data"]) if d.get("data") else None
    return d


class ThreadStore:
    """SQLite-backed transcript and turn ledger, sharing the conversation database file."""

    def __init__(self, db_path: Path | str = "runs/conversations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialises writers in this process; ``BEGIN IMMEDIATE`` serialises across processes.
        self._write_lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with bounded_connection(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_entries (
                    conversation_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    turn_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    at TEXT NOT NULL,
                    anchor TEXT,
                    actions TEXT,
                    data TEXT,
                    PRIMARY KEY (conversation_id, seq)
                );
                """
            )
            columns = [
                row["name"] for row in conn.execute("PRAGMA table_info(thread_entries);").fetchall()
            ]
            if "data" not in columns:
                conn.execute("ALTER TABLE thread_entries ADD COLUMN data TEXT;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_turns (
                    conversation_id TEXT PRIMARY KEY,
                    turn_id TEXT,
                    state TEXT NOT NULL DEFAULT 'idle',
                    actor TEXT,
                    opened_at TEXT,
                    closed_at TEXT,
                    outcome TEXT
                );
                """
            )

    # ------------------------------------------------------------------ writes

    def append(
        self,
        conversation_id: str,
        entries: Iterable[dict[str, Any]],
        turn_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Appends entries, assigning the next sequence numbers in one transaction."""
        pending = list(entries)
        if not pending or not conversation_id:
            return []

        with self._write_lock, bounded_connection(self.db_path) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM thread_entries WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                seq = int(row[0] or 0)
                written: list[dict[str, Any]] = []
                for entry in pending:
                    seq += 1
                    record = _normalize(entry, turn_id, seq)
                    conn.execute(
                        """
                        INSERT INTO thread_entries (
                            conversation_id, seq, turn_id, actor, kind, status, title, body, at,
                            anchor, actions, data
                        ) VALUES (
                            :conversation_id, :seq, :turn_id, :actor, :kind, :status, :title,
                            :body, :at, :anchor, :actions, :data
                        )
                        """,
                        {**record, "conversation_id": conversation_id},
                    )
                    written.append(record)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return written

    def begin_turn(
        self,
        conversation_id: str,
        actor: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Claims the conversation for ``actor``; raises if another turn is still running."""
        turn_id = turn_id or f"turn_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        with self._write_lock, bounded_connection(self.db_path) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO thread_turns (conversation_id, state) VALUES (?, 'idle')
                    ON CONFLICT(conversation_id) DO NOTHING
                    """,
                    (conversation_id,),
                )
                cur = conn.execute(
                    """
                    UPDATE thread_turns
                       SET state = 'running', turn_id = ?, actor = ?, opened_at = ?,
                           closed_at = NULL, outcome = NULL
                     WHERE conversation_id = ? AND state <> 'running'
                    """,
                    (turn_id, actor, now, conversation_id),
                )
                if cur.rowcount == 0:
                    row = conn.execute(
                        "SELECT turn_id FROM thread_turns WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                    active = row["turn_id"] if row else None
                    conn.execute("ROLLBACK")
                    raise TurnInProgressError(conversation_id, active)
                conn.execute("COMMIT")
            except TurnInProgressError:
                raise
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self.state(conversation_id)

    def close_turn(
        self,
        conversation_id: str,
        turn_id: str | None = None,
        state: str = "idle",
        outcome: str | None = None,
    ) -> dict[str, Any]:
        """Hands the conversation back. Only a running turn can be closed."""
        if state not in TURN_STATES:
            raise ValueError(f"Unknown turn state: {state}")
        now = utc_now_iso()
        with self._write_lock, bounded_connection(self.db_path) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                if turn_id:
                    conn.execute(
                        """
                        UPDATE thread_turns
                           SET state = ?, outcome = ?, closed_at = ?
                         WHERE conversation_id = ? AND turn_id = ? AND state = 'running'
                        """,
                        (state, outcome, now, conversation_id, turn_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE thread_turns
                           SET state = ?, outcome = ?, closed_at = ?
                         WHERE conversation_id = ? AND state = 'running'
                        """,
                        (state, outcome, now, conversation_id),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self.state(conversation_id)

    def handover_turn(self, conversation_id: str, turn_id: str, actor: str) -> dict[str, Any]:
        """Moves ownership of an open turn, e.g. studio -> trainer when a run is queued."""
        now = utc_now_iso()
        with self._write_lock, bounded_connection(self.db_path) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    UPDATE thread_turns
                       SET actor = ?, opened_at = COALESCE(opened_at, ?)
                     WHERE conversation_id = ? AND turn_id = ? AND state = 'running'
                    """,
                    (actor, now, conversation_id, turn_id),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self.state(conversation_id)

    def reconcile_interrupted(self, max_age_seconds: int = STALE_TURN_SECONDS) -> int:
        """Reclaims turns a previous process left running, so a restart never locks the studio."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        now = utc_now_iso()
        with self._write_lock, bounded_connection(self.db_path) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    """
                    UPDATE thread_turns
                       SET state = 'idle', outcome = 'interrupted', closed_at = ?
                     WHERE state = 'running' AND COALESCE(opened_at, '') < ?
                    """,
                    (now, cutoff),
                )
                conn.execute("COMMIT")
                return int(cur.rowcount or 0)
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def delete(self, conversation_id: str) -> None:
        """Removes the thread with the conversation it belongs to."""
        with self._write_lock, bounded_connection(self.db_path) as conn:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM thread_entries WHERE conversation_id = ?", (conversation_id,)
                )
                conn.execute("DELETE FROM thread_turns WHERE conversation_id = ?", (conversation_id,))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------- reads

    def state(self, conversation_id: str) -> dict[str, Any]:
        with bounded_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM thread_turns WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
        if not row:
            return {
                "conversation_id": conversation_id,
                "turn_id": None,
                "state": "idle",
                "actor": None,
                "opened_at": None,
                "closed_at": None,
                "outcome": None,
            }
        return dict(row)

    def entries(
        self,
        conversation_id: str,
        since_seq: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Returns entries in ascending sequence order, optionally only what came after a cursor."""
        with bounded_connection(self.db_path) as conn:
            sql = (
                "SELECT * FROM thread_entries WHERE conversation_id = ? AND seq > ? "
                "ORDER BY seq ASC"
            )
            params: list[Any] = [conversation_id, int(since_seq or 0)]
            if limit:
                sql += " LIMIT ?"
                params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_entry(r) for r in rows]

    def tail(self, conversation_id: str, limit: int = 500) -> list[dict[str, Any]]:
        """Returns the most recent entries, still in ascending order — what a reload needs."""
        with bounded_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM thread_entries
                     WHERE conversation_id = ?
                     ORDER BY seq DESC
                     LIMIT ?
                ) ORDER BY seq ASC
                """,
                (conversation_id, int(limit)),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def count(self, conversation_id: str) -> int:
        with bounded_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM thread_entries WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row[0] or 0) if row else 0

    # ------------------------------------------------------------- legacy data

    def migrate_legacy(
        self, conversation_id: str, chat_history: list[dict[str, Any]] | None
    ) -> int:
        """Seeds the thread from a pre-thread conversation, once, without rewording anything.

        Older rows carry only ``role``/``content``/``timestamp`` (plus ``questions``/``answers``).
        They map to the closest honest actor instead of being lumped into a single label.
        """
        if not chat_history or self.count(conversation_id):
            return 0

        mapped: list[dict[str, Any]] = []
        for msg in chat_history:
            role = str(msg.get("role") or "")
            timestamp = msg.get("timestamp") or utc_now_iso()
            if role == "user":
                mapped.append(
                    {
                        "actor": "user",
                        "kind": "instruction",
                        "status": "success",
                        "title": "You",
                        "body": str(msg.get("content") or ""),
                        "at": timestamp,
                    }
                )
            elif role == "user_answers":
                answers = msg.get("answers") or {}
                body = "\n".join(f"{k}: {v}" for k, v in answers.items())
                mapped.append(
                    {
                        "actor": "user",
                        "kind": "answer",
                        "status": "success",
                        "title": "You answered",
                        "body": body,
                        "at": timestamp,
                    }
                )
            elif role == "assistant":
                questions = msg.get("questions") or []
                mapped.append(
                    {
                        "actor": "studio",
                        "kind": "question" if questions else "answer",
                        "status": "success",
                        "title": "Studio asked" if questions else "Studio",
                        "body": str(msg.get("content") or ""),
                        "at": timestamp,
                    }
                )
        if not mapped:
            return 0
        return len(self.append(conversation_id, mapped, turn_id="turn_legacy"))


_global_thread_store: ThreadStore | None = None


def get_thread_store() -> ThreadStore:
    global _global_thread_store
    if _global_thread_store is None:
        _global_thread_store = ThreadStore()
    return _global_thread_store
