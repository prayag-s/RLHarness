"""Versioned Task State Store backed by SQLite and local run snapshots."""

from __future__ import annotations
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from rl_ide.compiler.models import TaskConfig
from rl_ide.reward.dsl import RewardDSL


class TaskVersion(BaseModel):
    """Immutable snapshot of a task configuration, reward DSL, and checkpoint."""
    task_id: str
    version: int
    version_tag: str
    task_config: TaskConfig
    reward_dsl: RewardDSL
    applied_critique: str | None = None
    policy_checkpoint_path: str | None = None
    locked_episode_ids: list[int] = Field(default_factory=list)
    created_at: str
    snapshot_dir: str


class TaskStateStore:
    """Manages versioned persistence and rollback for tasks in SQLite and filesystem."""

    def __init__(self, base_dir: str | Path = "runs", db_name: str = "tasks.db"):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / db_name
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    current_version INTEGER NOT NULL,
                    name TEXT,
                    adapter_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_versions (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    version_tag TEXT NOT NULL,
                    applied_critique TEXT,
                    policy_checkpoint_path TEXT,
                    locked_episode_ids TEXT NOT NULL,
                    snapshot_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, version),
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );
            """)

    def save_version(
        self,
        task_id: str,
        task_config: TaskConfig,
        reward_dsl: RewardDSL,
        applied_critique: str | None = None,
        policy_checkpoint_path: str | None = None,
        locked_episode_ids: list[int] | None = None,
    ) -> TaskVersion:
        """Saves a new immutable version snapshot to disk and records metadata in SQLite."""
        locked_ids = list(locked_episode_ids or [])
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            # Query max version
            cursor = conn.execute(
                "SELECT MAX(version) as max_v FROM task_versions WHERE task_id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
            current_max = row["max_v"] if row and row["max_v"] is not None else 0
            new_version = current_max + 1
            version_tag = f"v{new_version}"

            # Create version directory
            version_dir = self.base_dir / task_id / version_tag
            version_dir.mkdir(parents=True, exist_ok=True)

            # Persist YAML configs
            config_file = version_dir / "task_config.yaml"
            config_file.write_text(task_config.to_yaml(), encoding="utf-8")

            reward_file = version_dir / "reward_dsl.yaml"
            reward_file.write_text(reward_dsl.to_yaml(), encoding="utf-8")

            # Copy checkpoint if provided
            saved_checkpoint_path: str | None = None
            if policy_checkpoint_path:
                src_ckpt = Path(policy_checkpoint_path)
                if src_ckpt.exists() and src_ckpt.is_file():
                    dest_ckpt = version_dir / src_ckpt.name
                    if src_ckpt.resolve() != dest_ckpt.resolve():
                        shutil.copy2(src_ckpt, dest_ckpt)
                    saved_checkpoint_path = str(dest_ckpt)
                else:
                    saved_checkpoint_path = policy_checkpoint_path

            # Update database
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks (task_id, current_version, name, adapter_name, created_at, updated_at)
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    COALESCE((SELECT created_at FROM tasks WHERE task_id = ?), ?),
                    ?
                )
                """,
                (
                    task_id,
                    new_version,
                    task_config.name,
                    task_config.adapter_name,
                    task_id,
                    now_iso,
                    now_iso,
                ),
            )

            conn.execute(
                """
                INSERT INTO task_versions (
                    task_id, version, version_tag, applied_critique,
                    policy_checkpoint_path, locked_episode_ids, snapshot_dir, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    new_version,
                    version_tag,
                    applied_critique,
                    saved_checkpoint_path,
                    json.dumps(locked_ids),
                    str(version_dir),
                    now_iso,
                ),
            )

        return TaskVersion(
            task_id=task_id,
            version=new_version,
            version_tag=version_tag,
            task_config=task_config,
            reward_dsl=reward_dsl,
            applied_critique=applied_critique,
            policy_checkpoint_path=saved_checkpoint_path,
            locked_episode_ids=locked_ids,
            created_at=now_iso,
            snapshot_dir=str(version_dir),
        )

    def get_version(self, task_id: str, version: int) -> TaskVersion:
        """Retrieves a specific version snapshot."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT task_id, version, version_tag, applied_critique,
                       policy_checkpoint_path, locked_episode_ids, snapshot_dir, created_at
                FROM task_versions
                WHERE task_id = ? AND version = ?
                """,
                (task_id, version),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Version {version} not found for task '{task_id}'")

            snapshot_dir = Path(row["snapshot_dir"])
            config_file = snapshot_dir / "task_config.yaml"
            reward_file = snapshot_dir / "reward_dsl.yaml"

            if not config_file.exists() or not reward_file.exists():
                raise FileNotFoundError(f"Snapshot files missing in {snapshot_dir}")

            task_config = TaskConfig.from_yaml(config_file.read_text(encoding="utf-8"))
            reward_dsl = RewardDSL.from_yaml(reward_file.read_text(encoding="utf-8"))
            locked_ids = json.loads(row["locked_episode_ids"])

            return TaskVersion(
                task_id=row["task_id"],
                version=row["version"],
                version_tag=row["version_tag"],
                task_config=task_config,
                reward_dsl=reward_dsl,
                applied_critique=row["applied_critique"],
                policy_checkpoint_path=row["policy_checkpoint_path"],
                locked_episode_ids=locked_ids,
                created_at=row["created_at"],
                snapshot_dir=str(snapshot_dir),
            )

    def get_current_version(self, task_id: str) -> TaskVersion | None:
        """Retrieves the active current version of the task."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT current_version FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self.get_version(task_id, row["current_version"])

    def list_versions(self, task_id: str) -> list[TaskVersion]:
        """Lists all version snapshots for a task in chronological order."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT version FROM task_versions WHERE task_id = ? ORDER BY version ASC",
                (task_id,),
            )
            rows = cursor.fetchall()
            return [self.get_version(task_id, r["version"]) for r in rows]

    def rollback(self, task_id: str, target_version: int) -> TaskVersion:
        """Rolls back the task state to target_version, restoring DSL, config, and checkpoint.
        
        Creates a new snapshot explicitly recording the rollback event to preserve full audit history.
        """
        target_ver = self.get_version(task_id, target_version)

        # Save as new version recording rollback
        restored = self.save_version(
            task_id=task_id,
            task_config=target_ver.task_config,
            reward_dsl=target_ver.reward_dsl,
            applied_critique=f"Rollback to v{target_version}",
            policy_checkpoint_path=target_ver.policy_checkpoint_path,
            locked_episode_ids=target_ver.locked_episode_ids,
        )

        return restored
