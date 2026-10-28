"""Full Conversation & Session Persistence Store for RL-Native Agent IDE v4.1.

Persists every conversation in SQLite (`runs/conversations.db`) + workspace folder
(`workspaces/<conversation_id>/`) starting from the very first `/api/v4/architect/chat` message.
Unified `conversation_id` == `session_id`.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_title_from_prompt(prompt: str) -> str:
    """Auto-generates a clean title for a conversation from user prompt."""
    if not prompt or not prompt.strip():
        return "New Agent"
    from rl_ide.architect.intake import is_tictactoe_intent
    if is_tictactoe_intent(prompt):
        return "Tic-Tac-Toe Self-Play Agent"

    clean = re.sub(r"^(train|build|create|make|synthesize)\s+(an?\s+)?", "", prompt.strip(), flags=re.IGNORECASE)
    clean = clean.split("\n")[0].strip()
    if not clean:
        return "New Agent"
    clean = clean[0].upper() + clean[1:]
    if len(clean) > 40:
        clean = clean[:37] + "..."
    return clean


class ConversationStore:
    """Manages SQLite and disk persistence for conversational synthesis sessions."""

    def __init__(
        self,
        db_path: Path | str = "runs/conversations.db",
        workspaces_dir: Path | str = "workspaces",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir = Path(workspaces_dir)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chat_history TEXT NOT NULL,
                    dataset_id TEXT,
                    blueprint_spec TEXT,
                    training_params TEXT,
                    current_round INTEGER DEFAULT 1,
                    best_metric REAL DEFAULT 0.0,
                    rounds_history TEXT,
                    episodes TEXT,
                    env_code TEXT,
                    arena_html TEXT,
                    latest_model_zip_path TEXT,
                    arena_locked INTEGER DEFAULT 0
                );
                """
            )
            # Automatic schema migration for existing databases
            cur = conn.execute("PRAGMA table_info(conversations);")
            columns = [row["name"] for row in cur.fetchall()]
            if "arena_locked" not in columns:
                try:
                    conn.execute("ALTER TABLE conversations ADD COLUMN arena_locked INTEGER DEFAULT 0;")
                except Exception:
                    pass
            conn.commit()

    def create_conversation(
        self,
        title: str = "New Agent",
        conversation_id: str | None = None,
        dataset_id: str | None = None,
        training_params: dict[str, Any] | None = None,
        arena_locked: bool = False,
    ) -> dict[str, Any]:
        """Creates a new blank conversation entry and initializes its workspace directory."""
        cid = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        now = utc_now_iso()
        t_params = training_params or {
            "max_auto_tune_rounds": 5,
            "episodes_per_round": 200,
            "target_metric_threshold": 90.0,
        }

        ws_dir = self.workspaces_dir / cid
        ws_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "conversation_id": cid,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "status": "clarifying",
            "chat_history": json.dumps([]),
            "dataset_id": dataset_id,
            "blueprint_spec": None,
            "training_params": json.dumps(t_params),
            "current_round": 1,
            "best_metric": 0.0,
            "rounds_history": json.dumps([]),
            "episodes": json.dumps([]),
            "env_code": None,
            "arena_html": None,
            "latest_model_zip_path": None,
            "arena_locked": 1 if arena_locked else 0,
        }

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, title, created_at, updated_at, status, chat_history,
                    dataset_id, blueprint_spec, training_params, current_round, best_metric,
                    rounds_history, episodes, env_code, arena_html, latest_model_zip_path,
                    arena_locked
                ) VALUES (
                    :conversation_id, :title, :created_at, :updated_at, :status, :chat_history,
                    :dataset_id, :blueprint_spec, :training_params, :current_round, :best_metric,
                    :rounds_history, :episodes, :env_code, :arena_html, :latest_model_zip_path,
                    :arena_locked
                )
                ON CONFLICT(conversation_id) DO NOTHING;
                """,
                record,
            )
            conn.commit()

        return self.get_conversation(cid) or record

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieves and deserializes the full conversation state from SQLite."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            d = dict(row)
            d["chat_history"] = json.loads(d["chat_history"]) if d["chat_history"] else []
            d["blueprint_spec"] = json.loads(d["blueprint_spec"]) if d["blueprint_spec"] else None
            d["training_params"] = json.loads(d["training_params"]) if d["training_params"] else {
                "max_auto_tune_rounds": 5,
                "episodes_per_round": 200,
                "target_metric_threshold": 90.0,
            }
            d["rounds_history"] = json.loads(d["rounds_history"]) if d["rounds_history"] else []
            d["episodes"] = json.loads(d["episodes"]) if d["episodes"] else []
            d["arena_locked"] = bool(d.get("arena_locked", False))
            return d

    def list_conversations(self) -> list[dict[str, Any]]:
        """Returns metadata summary of all saved conversations sorted by updated_at DESC."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT conversation_id, title, status, best_metric, training_params,
                       current_round, updated_at, arena_locked
                FROM conversations
                ORDER BY updated_at DESC;
                """
            )
            rows = cur.fetchall()
            summaries = []
            for row in rows:
                t_params = {}
                if row["training_params"]:
                    try:
                        t_params = json.loads(row["training_params"])
                    except Exception:
                        pass
                row_dict = dict(row)
                summaries.append({
                    "conversation_id": row_dict["conversation_id"],
                    "title": row_dict["title"],
                    "status": row_dict["status"],
                    "best_metric": float(row_dict["best_metric"] or 0.0),
                    "target_metric_threshold": float(t_params.get("target_metric_threshold", 90.0)),
                    "current_round": int(row_dict["current_round"] or 1),
                    "max_auto_tune_rounds": int(t_params.get("max_auto_tune_rounds", 5)),
                    "arena_locked": bool(row_dict.get("arena_locked", False)),
                    "updated_at": row_dict["updated_at"],
                })
            return summaries

    def save_or_update(
        self,
        conversation_id: str,
        title: str | None = None,
        status: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        dataset_id: str | None = None,
        blueprint_spec: dict[str, Any] | None = None,
        training_params: dict[str, Any] | None = None,
        current_round: int | None = None,
        best_metric: float | None = None,
        rounds_history: list[dict[str, Any]] | None = None,
        episodes: list[dict[str, Any]] | None = None,
        env_code: str | None = None,
        arena_html: str | None = None,
        latest_model_zip_path: str | None = None,
        arena_locked: bool | None = None,
    ) -> dict[str, Any]:
        """Upserts conversation state, serializing lists/dicts to JSON."""
        existing = self.get_conversation(conversation_id)
        now = utc_now_iso()

        # Ensure workspace exists
        ws_dir = self.workspaces_dir / conversation_id
        ws_dir.mkdir(parents=True, exist_ok=True)

        if not existing:
            # Create fresh
            effective_title = title or "New Agent"
            effective_status = status or "clarifying"
            effective_chat = chat_history or []
            effective_training_params = training_params or {
                "max_auto_tune_rounds": 5,
                "episodes_per_round": 200,
                "target_metric_threshold": 90.0,
            }
            effective_locked = 1 if arena_locked else 0
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO conversations (
                        conversation_id, title, created_at, updated_at, status, chat_history,
                        dataset_id, blueprint_spec, training_params, current_round, best_metric,
                        rounds_history, episodes, env_code, arena_html, latest_model_zip_path,
                        arena_locked
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        effective_title,
                        now,
                        now,
                        effective_status,
                        json.dumps(effective_chat),
                        dataset_id,
                        json.dumps(blueprint_spec) if blueprint_spec else None,
                        json.dumps(effective_training_params),
                        current_round or 1,
                        best_metric or 0.0,
                        json.dumps(rounds_history or []),
                        json.dumps(episodes or []),
                        env_code,
                        arena_html,
                        latest_model_zip_path,
                        effective_locked,
                    ),
                )
                conn.commit()
        else:
            effective_title = title if title is not None else existing["title"]
            effective_status = status if status is not None else existing["status"]
            effective_chat = chat_history if chat_history is not None else existing["chat_history"]
            effective_dataset = dataset_id if dataset_id is not None else existing["dataset_id"]
            effective_blueprint = blueprint_spec if blueprint_spec is not None else existing["blueprint_spec"]
            effective_training_params = (
                training_params if training_params is not None else existing["training_params"]
            )
            effective_round = current_round if current_round is not None else existing["current_round"]
            effective_metric = (
                max(best_metric, existing.get("best_metric", 0.0))
                if best_metric is not None
                else existing.get("best_metric", 0.0)
            )
            effective_rounds_hist = rounds_history if rounds_history is not None else existing["rounds_history"]
            effective_episodes = episodes if episodes is not None else existing["episodes"]
            effective_code = env_code if env_code is not None else existing["env_code"]
            effective_arena = arena_html if arena_html is not None else existing["arena_html"]
            effective_model_path = (
                latest_model_zip_path if latest_model_zip_path is not None else existing["latest_model_zip_path"]
            )
            effective_locked = (
                (1 if arena_locked else 0)
                if arena_locked is not None
                else (1 if existing.get("arena_locked") else 0)
            )

            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE conversations SET
                        title = ?,
                        updated_at = ?,
                        status = ?,
                        chat_history = ?,
                        dataset_id = ?,
                        blueprint_spec = ?,
                        training_params = ?,
                        current_round = ?,
                        best_metric = ?,
                        rounds_history = ?,
                        episodes = ?,
                        env_code = ?,
                        arena_html = ?,
                        latest_model_zip_path = ?,
                        arena_locked = ?
                    WHERE conversation_id = ?
                    """,
                    (
                        effective_title,
                        now,
                        effective_status,
                        json.dumps(effective_chat),
                        effective_dataset,
                        json.dumps(effective_blueprint) if effective_blueprint else None,
                        json.dumps(effective_training_params),
                        effective_round,
                        effective_metric,
                        json.dumps(effective_rounds_hist),
                        json.dumps(effective_episodes),
                        effective_code,
                        effective_arena,
                        effective_model_path,
                        effective_locked,
                        conversation_id,
                    ),
                )
                conn.commit()

        # Save files to workspace folder if code/arena provided
        if env_code:
            try:
                (ws_dir / "env_module.py").write_text(env_code, encoding="utf-8")
            except Exception:
                pass
        if arena_html:
            try:
                (ws_dir / "arena.html").write_text(arena_html, encoding="utf-8")
            except Exception:
                pass

        return self.get_conversation(conversation_id)  # type: ignore

    def append_chat_message(self, conversation_id: str, message: dict[str, Any]) -> dict[str, Any]:
        """Appends a single message (user prompt, clarifying question batch, assistant reply) to chat_history."""
        existing = self.get_conversation(conversation_id)
        if not existing:
            self.create_conversation(conversation_id=conversation_id)
            existing = self.get_conversation(conversation_id)

        history = existing["chat_history"] if existing else []
        history.append(message)
        return self.save_or_update(conversation_id, chat_history=history)

    def update_params(self, conversation_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Updates max_auto_tune_rounds, episodes_per_round, target_metric_threshold."""
        existing = self.get_conversation(conversation_id)
        if not existing:
            raise KeyError(f"Conversation '{conversation_id}' not found.")

        current_params = existing.get("training_params", {}) or {}
        for k in ["max_auto_tune_rounds", "episodes_per_round", "target_metric_threshold"]:
            if k in params:
                current_params[k] = params[k]

        return self.save_or_update(conversation_id, training_params=current_params)

    def lock_arena(self, conversation_id: str, locked: bool = True) -> dict[str, Any]:
        """Locks or unlocks the visual arena configuration."""
        existing = self.get_conversation(conversation_id)
        if not existing:
            raise KeyError(f"Conversation '{conversation_id}' not found.")
        return self.save_or_update(conversation_id, arena_locked=locked)

    def delete(self, conversation_id: str) -> bool:
        """Deletes conversation record and associated workspace + runs files."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            conn.commit()

        # Remove workspace directory
        ws_dir = self.workspaces_dir / conversation_id
        if ws_dir.exists():
            try:
                shutil.rmtree(ws_dir, ignore_errors=True)
            except Exception:
                pass

        # Remove runs directory
        runs_dir = Path("runs") / conversation_id
        if runs_dir.exists():
            try:
                shutil.rmtree(runs_dir, ignore_errors=True)
            except Exception:
                pass

        return True


# Global Singleton Store
_global_conversation_store: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    global _global_conversation_store
    if _global_conversation_store is None:
        _global_conversation_store = ConversationStore()
    return _global_conversation_store
