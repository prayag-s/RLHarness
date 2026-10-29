"""Durable storage for uploaded dataset records used by synthesized environments.

Datasets are uploaded in-memory; a resumed conversation must still be able to instantiate its
classification environment with the same records, so they are persisted per conversation.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

RECORDS_FILENAME = "dataset_records.json"


def save_records(workspace_dir: str | Path, records: list[dict[str, Any]] | None) -> None:
    """Persists the records for a conversation (no-op when there is nothing to persist)."""
    if not records:
        return
    path = Path(workspace_dir) / RECORDS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, default=str), encoding="utf-8")


def load_records(workspace_dir: str | Path) -> list[dict[str, Any]] | None:
    """Loads persisted records, returning None when unavailable or unreadable."""
    path = Path(workspace_dir) / RECORDS_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None
    except Exception:
        return None
