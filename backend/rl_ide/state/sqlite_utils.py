"""Shared SQLite plumbing for the studio's local stores.

Both the conversation store and the job store share one database file and run in WAL mode so a
training consumer and the API can write concurrently. Two details of that mode bit us for real and
belong here rather than in each store:

* ``with sqlite3.connect(...) as conn`` does **not** close the connection — it only commits or
  rolls back. Connections must be closed explicitly or the WAL cannot be reset.
* WAL files are reused rather than truncated by default, so a checkpoint that is ever blocked by a
  reader leaves the file at its high-water mark forever. One development run reached a 2.4 GB WAL
  and filled the disk. ``journal_size_limit`` bounds that.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Iterator

# Keep the write-ahead log small enough that a burst of episode writes can never fill the disk.
JOURNAL_SIZE_LIMIT_BYTES = 64 * 1024 * 1024
BUSY_TIMEOUT_MS = 30_000


def open_connection(db_path: Path | str) -> sqlite3.Connection:
    """Opens a connection with the pragmas every studio store needs."""
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    # -1 (the default) means "never truncate"; a bounded value lets a checkpoint reclaim space.
    conn.execute(f"PRAGMA journal_size_limit={JOURNAL_SIZE_LIMIT_BYTES};")
    return conn


@contextlib.contextmanager
def bounded_connection(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """Yields a connection and always closes it, so the WAL can be checkpointed."""
    conn = open_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        with contextlib.suppress(Exception):
            conn.rollback()
        raise
    finally:
        conn.close()


def checkpoint(db_path: Path | str, mode: str = "TRUNCATE") -> tuple[int, int, int]:
    """Checkpoints and truncates the WAL. Returns ``(busy, log_pages, checkpointed_pages)``.

    ``busy == 1`` means another connection held a read transaction: the checkpoint ran as far as it
    could and the caller may try again later. Callers must not treat that as an error.
    """
    with contextlib.closing(open_connection(db_path)) as conn:
        row = conn.execute(f"PRAGMA wal_checkpoint({mode});").fetchone()
    return (int(row[0]), int(row[1]), int(row[2])) if row else (0, 0, 0)
