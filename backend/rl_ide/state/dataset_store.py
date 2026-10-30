"""Durable, owner-scoped storage for ingested datasets.

Datasets previously lived only in a process dict (`ingested_datasets_store`), so an upload vanished
on restart and every caller could read every dataset. Rows are stored as compressed JSON in
`runs/datasets/<id>.json.gz` with metadata in SQLite; the DataFrame is rebuilt on load.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rl_ide.ingest.loader import IngestedDataset


class DatasetStore:
    """Persists ingested datasets with an owner column, and rebuilds them on demand."""

    def __init__(
        self,
        db_path: Path | str = "runs/datasets.db",
        files_dir: Path | str = "runs/datasets",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.files_dir = Path(files_dir)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, IngestedDataset] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    format TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'tabular',
                    created_at TEXT NOT NULL,
                    num_rows INTEGER DEFAULT 0,
                    payload_path TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_datasets_owner ON datasets(owner_id);"
            )
            conn.commit()

    # ------------------------------------------------------------------ write
    def save(self, dataset: IngestedDataset, owner_id: str = "local") -> IngestedDataset:
        """Persists the dataset (rows + derived metadata) and caches it in memory."""
        payload_path = self.files_dir / f"{dataset.dataset_id}.json.gz"
        frame = getattr(dataset, "df", None)
        rows = frame.to_dict(orient="records") if frame is not None else []
        with gzip.open(payload_path, "wt", encoding="utf-8") as handle:
            json.dump({"columns": [str(c) for c in (frame.columns if frame is not None else [])],
                       "rows": rows}, handle, default=str)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, owner_id, name, source_type, format, role, created_at,
                    num_rows, payload_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    name = excluded.name,
                    num_rows = excluded.num_rows,
                    payload_path = excluded.payload_path;
                """,
                (
                    dataset.dataset_id,
                    str(owner_id or "local"),
                    dataset.name,
                    dataset.source_type,
                    dataset.format,
                    dataset.role,
                    dataset.created_at or datetime.now(timezone.utc).isoformat(),
                    len(rows),
                    str(payload_path),
                ),
            )
            conn.commit()
        self._cache[dataset.dataset_id] = dataset
        return dataset

    # ------------------------------------------------------------------- read
    def owner_of(self, dataset_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT owner_id FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        return str(row["owner_id"]) if row else None

    def get(self, dataset_id: str, owner_id: str | None = None) -> IngestedDataset | None:
        """Loads a dataset, enforcing ownership when an owner is supplied."""
        if not dataset_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        if row is None:
            return None
        if owner_id is not None and str(row["owner_id"]) != str(owner_id):
            return None

        cached = self._cache.get(dataset_id)
        if cached is not None:
            return cached

        payload_path = Path(row["payload_path"])
        if not payload_path.exists():
            return None
        with gzip.open(payload_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        frame = pd.DataFrame(payload.get("rows") or [], columns=payload.get("columns") or None)
        dataset = IngestedDataset(
            dataset_id=row["dataset_id"],
            name=row["name"],
            source_type=row["source_type"],
            format=row["format"],
            role=row["role"],
            df=frame,
            created_at=row["created_at"],
        )
        self._cache[dataset_id] = dataset
        return dataset

    def list(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT dataset_id, owner_id, name, format, role, created_at, num_rows FROM datasets"
        params: tuple[Any, ...] = ()
        if owner_id is not None:
            query += " WHERE owner_id = ?"
            params = (str(owner_id),)
        query += " ORDER BY created_at DESC;"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    # ----------------------------------------------------------------- delete
    def delete(self, dataset_id: str, owner_id: str | None = None) -> bool:
        if owner_id is not None and self.owner_of(dataset_id) != str(owner_id):
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_path FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            conn.commit()
        if row and Path(row["payload_path"]).exists():
            try:
                Path(row["payload_path"]).unlink()
            except OSError:
                pass
        self._cache.pop(dataset_id, None)
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
            try:
                conn.execute("VACUUM;")
            finally:
                conn.close()
        except Exception:
            pass
        return row is not None


_global_store: DatasetStore | None = None


def get_dataset_store() -> DatasetStore:
    global _global_store
    if _global_store is None:
        _global_store = DatasetStore()
    return _global_store


def reset_dataset_store(store: DatasetStore | None = None) -> None:
    """Test helper: replaces the process-wide dataset store."""
    global _global_store
    _global_store = store
