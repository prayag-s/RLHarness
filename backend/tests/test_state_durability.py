"""Local persistence must not grow without bound.

A development run left a 2.4 GB write-ahead log beside an 88 MB database and filled the disk,
because ``with sqlite3.connect(...)`` only ends a transaction — it never closes the connection — and
SQLite reuses a WAL rather than truncating it. Both are fixed in ``rl_ide.state.sqlite_utils`` and
asserted here, because "the disk filled up" is not a failure any other test would catch.
"""

from __future__ import annotations

import pytest

from rl_ide.state.conversation_store import ConversationStore
from rl_ide.state.sqlite_utils import (
    JOURNAL_SIZE_LIMIT_BYTES,
    bounded_connection,
    checkpoint,
    open_connection,
)


def _wal_size(db_path) -> int:
    wal = db_path.with_suffix(db_path.suffix + "-wal")
    return wal.stat().st_size if wal.exists() else 0


def test_connections_are_closed_so_the_wal_can_be_truncated(tmp_path):
    db = tmp_path / "conv.db"
    store = ConversationStore(db_path=db, workspaces_dir=tmp_path / "ws")
    store.create_conversation(conversation_id="c1", owner_id="o1")
    for round_idx in range(20):
        store.save_or_update("c1", best_metric=float(round_idx), chat_history=[{"i": round_idx}])

    # No connection may be left open when the store is done: a leaked reader blocks the checkpoint.
    busy, _log_pages, _checkpointed = checkpoint(db)
    assert busy == 0, "a leaked connection still holds the database open"
    assert _wal_size(db) == 0, "the WAL survived a truncating checkpoint"


def test_journal_size_limit_is_applied(tmp_path):
    conn = open_connection(tmp_path / "cfg.db")
    try:
        limit = conn.execute("PRAGMA journal_size_limit;").fetchone()[0]
    finally:
        conn.close()
    assert int(limit) == JOURNAL_SIZE_LIMIT_BYTES


def test_bounded_connection_rolls_back_on_error(tmp_path):
    db = tmp_path / "rb.db"
    with bounded_connection(db) as conn:
        conn.execute("CREATE TABLE t (v TEXT);")
        conn.commit()

    with pytest.raises(RuntimeError):
        with bounded_connection(db) as conn:
            conn.execute("INSERT INTO t (v) VALUES ('leaked');")
            raise RuntimeError("boom")

    with bounded_connection(db) as conn:
        rows = conn.execute("SELECT v FROM t;").fetchall()
    assert rows == []


def test_delete_reclaims_disk_including_the_wal(tmp_path):
    db = tmp_path / "gc.db"
    store = ConversationStore(db_path=db, workspaces_dir=tmp_path / "ws")
    store.create_conversation(conversation_id="c1", owner_id="o1")
    # A conversation's episodes are the biggest rows in the schema; they are what grew the WAL.
    store.save_or_update(
        "c1",
        episodes=[{"episode_id": f"e{i}", "payload": "x" * 2048} for i in range(500)],
        best_metric=1.0,
    )
    populated = db.stat().st_size
    assert populated > 500_000

    assert store.delete("c1", owner_id="o1") is True
    assert _wal_size(db) == 0, "delete must not leave the WAL behind"
    assert db.stat().st_size < populated, "VACUUM did not reclaim the deleted pages"
    # The table itself must still exist for the next conversation.
    store.create_conversation(conversation_id="c2", owner_id="o1")
    assert store.get_conversation("c2", owner_id="o1") is not None


def test_repeated_writes_do_not_accumulate_a_wal(tmp_path):
    """Every operation closes its connection, so the WAL never reaches a high-water mark."""
    db = tmp_path / "loop.db"
    store = ConversationStore(db_path=db, workspaces_dir=tmp_path / "ws")
    store.create_conversation(conversation_id="c1", owner_id="o1")

    peak = 0
    for i in range(200):
        store.save_or_update("c1", episodes=[{"episode_id": f"e{i}", "payload": "y" * 1024}])
        peak = max(peak, _wal_size(db))

    assert peak <= JOURNAL_SIZE_LIMIT_BYTES, f"WAL reached {peak} bytes"
    assert _wal_size(db) == 0
