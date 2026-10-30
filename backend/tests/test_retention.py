"""Time-based retention (G9).

Deletion used to require someone to ask. Under a DPA the policy itself has to erase data, so these
tests pin the boundaries: nothing is deleted when retention is unconfigured, dry runs delete nothing,
expired rows go, fresh rows stay, and a single failure does not abort the sweep.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rl_ide.state.conversation_store import ConversationStore
from rl_ide.state.dataset_store import DatasetStore
from rl_ide.state.retention import (
    RetentionPolicy,
    RetentionSweeper,
    SweepReport,
    sweep_stale_workspaces,
)

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def _ago(days: int) -> str:
    return (NOW - timedelta(days=days)).isoformat()


@pytest.fixture()
def store(tmp_path):
    return ConversationStore(db_path=tmp_path / "c.db", workspaces_dir=tmp_path / "ws")


def _age(store: ConversationStore, conv_id: str, days: int) -> None:
    """Backdates a conversation's updated_at, which is what the policy reads."""
    with store._get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?;",
            (_ago(days), conv_id),
        )
        conn.commit()


def _sweeper(store, **policy_kwargs) -> RetentionSweeper:
    return RetentionSweeper(
        RetentionPolicy(**policy_kwargs),
        conversation_store_factory=lambda: store,
        now=lambda: NOW,
    )


def test_an_unconfigured_policy_deletes_nothing(store):
    store.create_conversation(conversation_id="old", owner_id="acme")
    _age(store, "old", 400)

    report = _sweeper(store).sweep()

    assert report.total == 0
    assert store.get_conversation("old") is not None, "a desktop studio must never eat user work"


def test_expired_conversations_are_deleted_and_fresh_ones_kept(store):
    for cid, days in (("ancient", 40), ("recent", 2)):
        store.create_conversation(conversation_id=cid, owner_id="acme")
        _age(store, cid, days)

    report = _sweeper(store, conversation_days=30).sweep()

    assert report.deleted["conversations"] == ["ancient"]
    assert store.get_conversation("ancient") is None
    assert store.get_conversation("recent") is not None


def test_dry_run_reports_without_deleting(store):
    store.create_conversation(conversation_id="ancient", owner_id="acme")
    _age(store, "ancient", 40)

    report = _sweeper(store, conversation_days=30).sweep(dry_run=True)

    assert report.dry_run is True
    assert report.deleted["conversations"] == ["ancient"]
    assert store.get_conversation("ancient") is not None, "a dry run must change nothing"


def test_datasets_expire_on_their_own_clock(tmp_path):
    from rl_ide.ingest import load_bytes

    ds_store = DatasetStore(db_path=tmp_path / "d.db", files_dir=tmp_path / "payloads")
    dataset = load_bytes(filename="rows.csv", content=b"a,b\n1,2\n3,4\n")
    ds_store.save(dataset, owner_id="acme")
    with ds_store._connect() as conn:
        conn.execute(
            "UPDATE datasets SET created_at = ? WHERE dataset_id = ?;",
            (_ago(90), dataset.dataset_id),
        )
        conn.commit()

    sweeper = RetentionSweeper(
        RetentionPolicy(conversation_days=1, dataset_days=30),
        dataset_store_factory=lambda: ds_store,
        now=lambda: NOW,
    )
    report = sweeper.sweep()

    assert report.deleted.get("datasets") == [dataset.dataset_id]
    assert ds_store.get(dataset.dataset_id, owner_id="acme") is None


def test_a_failure_in_one_artefact_does_not_stop_the_sweep(store):
    for cid in ("a", "b"):
        store.create_conversation(conversation_id=cid, owner_id="acme")
        _age(store, cid, 40)

    original = store.delete
    calls: list[str] = []

    def flaky(conv_id, owner_id=None):
        calls.append(conv_id)
        if conv_id == "a":
            raise RuntimeError("locked")
        return original(conv_id, owner_id)

    store.delete = flaky  # type: ignore[assignment]
    report = _sweeper(store, conversation_days=30).sweep()

    assert calls == ["a", "b"], "the sweep must continue past a failure"
    assert report.deleted["conversations"] == ["b"]
    assert any("a" in err for err in report.errors)


def test_policy_reads_from_the_environment(monkeypatch):
    monkeypatch.delenv("RL_IDE_RETENTION_DAYS", raising=False)
    assert RetentionPolicy.from_env().enabled is False

    monkeypatch.setenv("RL_IDE_RETENTION_DAYS", "30")
    policy = RetentionPolicy.from_env()
    assert policy.enabled is True
    assert (policy.conversation_days, policy.dataset_days, policy.bundle_days) == (30, 30, 30)

    # Per-artefact overrides win over the shared knob.
    monkeypatch.setenv("RL_IDE_RETENTION_DATASET_DAYS", "7")
    assert RetentionPolicy.from_env().dataset_days == 7

    # Zero / negative means "keep forever" for that kind, not "delete everything now".
    monkeypatch.setenv("RL_IDE_RETENTION_DATASET_DAYS", "0")
    assert RetentionPolicy.from_env().dataset_days is None


def test_orphan_workspaces_are_removed(tmp_path):
    base = tmp_path / "ws"
    orphan = base / "conv_orphan"
    orphan.mkdir(parents=True)
    (orphan / "arena.html").write_text("<html/>", encoding="utf-8")
    old = (NOW - timedelta(days=90)).timestamp()
    for path in [orphan, orphan / "arena.html"]:
        import os

        os.utime(path, (old, old))

    fresh = base / "conv_fresh"
    fresh.mkdir(parents=True)
    (fresh / "arena.html").write_text("<html/>", encoding="utf-8")

    removed = sweep_stale_workspaces(base, max_age_days=30)

    assert removed == 1
    assert not orphan.exists()
    assert fresh.exists()


def test_sweep_report_serialises_for_the_api():
    report = SweepReport(dry_run=True)
    report.add("conversations", "c1")
    report.errors.append("boom")
    payload = report.as_dict()
    assert payload["dry_run"] is True
    assert payload["total"] == 1
    assert payload["deleted"] == {"conversations": ["c1"]}
    assert payload["errors"] == ["boom"]
