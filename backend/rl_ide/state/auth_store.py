"""Tenant identity: hashed API tokens that resolve to an owner id.

Until now ownership came from an `x-owner-id` header that any caller could set, so every "tenant" was
a label rather than a boundary (G4). A hosted deployment needs the caller to *prove* which tenant it
is before any query is scoped by it.

Design choices, and why:

* **Only the hash is stored.** A leaked database must not yield working credentials. Tokens are 32
  random bytes rendered urlsafe; the row keeps `sha256(token)`.
* **Verification is constant-time** and the lookup is by hash, so a wrong token cannot be probed by
  timing or by prefix.
* **The plaintext is returned exactly once**, at creation.
* **The bootstrap token can create tokens but is not itself an owner**: it is a deployment secret
  (`RL_IDE_ADMIN_TOKEN`), not a tenant.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl_ide.state.sqlite_utils import bounded_connection

TOKEN_PREFIX = "rlt_"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class ApiTokenStore:
    """Issues, verifies and revokes tenant tokens in the same database as the data they scope."""

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
                CREATE TABLE IF NOT EXISTS api_tokens (
                    token_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    revoked INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_tokens_owner ON api_tokens(owner_id);"
            )
            conn.commit()

    # ------------------------------------------------------------------ issue
    def create(self, owner_id: str, label: str = "") -> dict[str, Any]:
        """Mints a token. The plaintext is in the return value and nowhere else."""
        owner = (owner_id or "").strip()
        if not owner:
            raise ValueError("owner_id is required")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        token_id = f"tok_{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO api_tokens (token_id, owner_id, label, token_hash, created_at, revoked)
                VALUES (?, ?, ?, ?, ?, 0);
                """,
                (token_id, owner, label, hash_token(token), _now()),
            )
            conn.commit()
        return {
            "token_id": token_id,
            "owner_id": owner,
            "label": label,
            "token": token,
            "created_at": _now(),
        }

    # ----------------------------------------------------------------- verify
    def verify(self, token: str) -> str | None:
        """Returns the owner id for a valid token, or None. Records last use."""
        if not token:
            return None
        digest = hash_token(token.strip())
        with self._conn() as conn:
            row = conn.execute(
                "SELECT token_id, owner_id, token_hash, revoked FROM api_tokens WHERE token_hash = ?;",
                (digest,),
            ).fetchone()
            if row is None:
                return None
            # Belt and braces: the query already matched the digest, but compare in constant time so
            # the code cannot be "improved" into an early-exit comparison later.
            if not hmac.compare_digest(str(row["token_hash"]), digest):
                return None
            if int(row["revoked"] or 0):
                return None
            conn.execute(
                "UPDATE api_tokens SET last_used_at = ? WHERE token_id = ?;",
                (_now(), row["token_id"]),
            )
            conn.commit()
            return str(row["owner_id"])

    # ----------------------------------------------------------------- manage
    def list(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        """Metadata only — never the token or its hash."""
        with self._conn() as conn:
            if owner_id:
                rows = conn.execute(
                    "SELECT token_id, owner_id, label, created_at, last_used_at, revoked "
                    "FROM api_tokens WHERE owner_id = ? ORDER BY created_at DESC;",
                    (str(owner_id),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT token_id, owner_id, label, created_at, last_used_at, revoked "
                    "FROM api_tokens ORDER BY created_at DESC;"
                ).fetchall()
        return [dict(row) for row in rows]

    def revoke(self, token_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("UPDATE api_tokens SET revoked = 1 WHERE token_id = ?;", (token_id,))
            conn.commit()
            return cur.rowcount > 0

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM api_tokens WHERE revoked = 0;").fetchone()
        return int(row["n"] if row else 0)


_global_token_store: ApiTokenStore | None = None


def get_token_store() -> ApiTokenStore:
    """Tokens live beside the data they scope, so a relocated state directory moves both.

    Bound to the conversation store's database rather than a second hardcoded path: otherwise a test
    (or a second deployment) that points the conversation store elsewhere would still read the
    developer's real tokens, and every request would suddenly require authentication.
    """
    global _global_token_store
    if _global_token_store is None:
        from rl_ide.state.conversation_store import get_conversation_store

        _global_token_store = ApiTokenStore(db_path=get_conversation_store().db_path)
    return _global_token_store


def reset_token_store_for_tests(store: "ApiTokenStore | None" = None) -> None:
    global _global_token_store
    _global_token_store = store
