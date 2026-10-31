"""Durable coach findings, judge cache and per-conversation LLM budget.

Findings outlive the process that produced them: a warning the user acknowledged must not
reappear on every render, and a blocker must still block after a restart. The same database file
as the conversation and job stores is used so retention and backups already cover it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rl_ide.coach.models import GuidanceItem
from rl_ide.state.sqlite_utils import bounded_connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CoachStore:
    def __init__(self, db_path: Path | str = "runs/conversations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return bounded_connection(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coach_findings (
                    conversation_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    why TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, fingerprint)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_coach_stage "
                "ON coach_findings(conversation_id, stage, status);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coach_judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coach_budget (
                    conversation_id TEXT PRIMARY KEY,
                    llm_calls INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                """
            )

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    def record(
        self,
        conversation_id: str,
        items: Iterable[GuidanceItem],
        *,
        stages: Iterable[str] | None = None,
    ) -> dict[str, str]:
        """Upserts this evaluation's findings and resolves the ones that no longer apply.

        Only the stages being evaluated are resolved: evaluating results must not close an
        objective warning the user has not revisited.
        """
        incoming = {item.fingerprint(): item for item in items}
        now = _now()
        scope = list(stages) if stages is not None else sorted({i.stage.value for i in items})

        with self._conn() as conn:
            for fingerprint, item in incoming.items():
                conn.execute(
                    """
                    INSERT INTO coach_findings (
                        conversation_id, fingerprint, rule_id, stage, severity, title, why,
                        evidence, actions, source, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
                    ON CONFLICT(conversation_id, fingerprint) DO UPDATE SET
                        severity = excluded.severity,
                        title = excluded.title,
                        why = excluded.why,
                        evidence = excluded.evidence,
                        actions = excluded.actions,
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (
                        conversation_id,
                        fingerprint,
                        item.rule_id,
                        item.stage.value,
                        item.severity.value,
                        item.title,
                        item.why,
                        json.dumps(item.evidence, sort_keys=True, default=str),
                        json.dumps(
                            [a.model_dump(mode="json") for a in item.actions],
                            sort_keys=True,
                        ),
                        item.source,
                        now,
                        now,
                    ),
                )

            if scope:
                placeholders = ",".join("?" for _ in scope)
                rows = conn.execute(
                    f"SELECT fingerprint FROM coach_findings "
                    f"WHERE conversation_id = ? AND stage IN ({placeholders}) "
                    f"AND status IN ('open', 'acknowledged')",
                    (conversation_id, *scope),
                ).fetchall()
                stale = [
                    row["fingerprint"]
                    for row in rows
                    if row["fingerprint"] not in incoming
                ]
                for fingerprint in stale:
                    conn.execute(
                        "UPDATE coach_findings SET status = 'resolved', updated_at = ? "
                        "WHERE conversation_id = ? AND fingerprint = ?",
                        (now, conversation_id, fingerprint),
                    )

            statuses = {
                row["fingerprint"]: row["status"]
                for row in conn.execute(
                    "SELECT fingerprint, status FROM coach_findings WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchall()
            }
        return {fp: statuses.get(fp, "open") for fp in incoming}

    def items(
        self,
        conversation_id: str,
        *,
        stage: str | None = None,
        include_resolved: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM coach_findings WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if stage:
            query += " AND stage = ?"
            params.append(stage)
        if not include_resolved:
            query += " AND status != 'resolved'"
        query += " ORDER BY CASE severity WHEN 'blocker' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, created_at"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def acknowledge(
        self,
        conversation_id: str,
        rule_id: str,
        *,
        fingerprint: str | None = None,
        note: str = "",
    ) -> list[dict[str, Any]]:
        now = _now()
        query = (
            "UPDATE coach_findings SET status = 'acknowledged', note = ?, updated_at = ? "
            "WHERE conversation_id = ? AND rule_id = ? AND status = 'open'"
        )
        params: list[Any] = [note, now, conversation_id, rule_id]
        if fingerprint:
            query += " AND fingerprint = ?"
            params.append(fingerprint)
        with self._conn() as conn:
            conn.execute(query, params)
            rows = conn.execute(
                "SELECT * FROM coach_findings WHERE conversation_id = ? AND rule_id = ? "
                "AND status = 'acknowledged'",
                (conversation_id, rule_id),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def counts(self, conversation_id: str) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT severity, status, COUNT(*) AS n FROM coach_findings "
                "WHERE conversation_id = ? AND status != 'resolved' GROUP BY severity, status",
                (conversation_id,),
            ).fetchall()
        counts = {"blocker": 0, "warning": 0, "tip": 0, "open": 0, "acknowledged": 0}
        for row in rows:
            counts[row["severity"]] = counts.get(row["severity"], 0) + int(row["n"])
            counts[row["status"]] = counts.get(row["status"], 0) + int(row["n"])
        return counts

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "conversation_id": row["conversation_id"],
            "fingerprint": row["fingerprint"],
            "rule_id": row["rule_id"],
            "stage": row["stage"],
            "severity": row["severity"],
            "title": row["title"],
            "why": row["why"],
            "evidence": json.loads(row["evidence"] or "{}"),
            "actions": json.loads(row["actions"] or "[]"),
            "source": row["source"],
            "status": row["status"],
            "note": row["note"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # Judge cache and budget
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(stage: str, facts: dict[str, Any], prompt: str | None = None) -> str:
        payload = json.dumps(
            {"stage": stage, "facts": facts, "prompt": prompt or ""},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cache_get(self, key: str) -> list[dict[str, Any]] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM coach_judge_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError:
            return None

    def cache_put(self, key: str, items: list[GuidanceItem]) -> None:
        payload = json.dumps([i.model_dump(mode="json") for i in items])
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO coach_judge_cache (cache_key, payload, created_at) "
                "VALUES (?, ?, ?)",
                (key, payload, _now()),
            )

    def try_reserve_llm_call(self, conversation_id: str, limit: int) -> bool:
        if limit <= 0:
            return False
        with self._conn() as conn:
            row = conn.execute(
                "SELECT llm_calls FROM coach_budget WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            current = int(row["llm_calls"]) if row else 0
            if current >= limit:
                return False
            conn.execute(
                "INSERT INTO coach_budget (conversation_id, llm_calls, updated_at) "
                "VALUES (?, 1, ?) ON CONFLICT(conversation_id) DO UPDATE SET "
                "llm_calls = llm_calls + 1, updated_at = excluded.updated_at",
                (conversation_id, _now()),
            )
        return True

    def llm_calls(self, conversation_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT llm_calls FROM coach_budget WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["llm_calls"]) if row else 0


_STORE: CoachStore | None = None


def get_coach_store() -> CoachStore:
    global _STORE
    if _STORE is None:
        _STORE = CoachStore()
    return _STORE
