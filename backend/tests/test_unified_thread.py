"""The unified thread: one ordered transcript, one owner per turn.

These tests pin the two properties the UI depends on and cannot enforce by itself: sequence numbers
are gap-free and monotonic per conversation, and a second party cannot open a turn while one is
running.
"""

from __future__ import annotations

import pytest

from rl_ide.state.thread_store import ThreadStore, TurnInProgressError


@pytest.fixture()
def store(tmp_path):
    return ThreadStore(tmp_path / "conversations.db")


def _entry(**overrides):
    base = {
        "actor": "studio",
        "kind": "answer",
        "status": "success",
        "title": "Studio",
        "body": "hello",
    }
    base.update(overrides)
    return base


def test_append_assigns_monotonic_sequence(store):
    store.append("c1", [_entry(body="one"), _entry(body="two")], turn_id="t1")
    store.append("c1", [_entry(body="three")], turn_id="t2")

    entries = store.entries("c1")

    assert [e["seq"] for e in entries] == [1, 2, 3]
    assert [e["body"] for e in entries] == ["one", "two", "three"]
    assert [e["turn_id"] for e in entries] == ["t1", "t1", "t2"]


def test_sequences_are_per_conversation(store):
    store.append("c1", [_entry()], turn_id="t1")
    store.append("c2", [_entry()], turn_id="t2")

    assert [e["seq"] for e in store.entries("c1")] == [1]
    assert [e["seq"] for e in store.entries("c2")] == [1]


def test_tail_keeps_ascending_order(store):
    store.append("c1", [_entry(body=str(i)) for i in range(10)], turn_id="t1")

    tail = store.tail("c1", limit=3)

    assert [e["body"] for e in tail] == ["7", "8", "9"]


def test_only_one_party_can_own_a_turn(store):
    turn = store.begin_turn("c1", actor="user")
    assert turn["state"] == "running"

    with pytest.raises(TurnInProgressError) as excinfo:
        store.begin_turn("c1", actor="trainer")
    assert excinfo.value.active_turn_id == turn["turn_id"]

    store.close_turn("c1", turn_id=turn["turn_id"], state="idle")
    second = store.begin_turn("c1", actor="user")
    assert second["state"] == "running"
    assert second["turn_id"] != turn["turn_id"]


def test_closing_a_superseded_turn_does_not_unlock_the_new_one(store):
    first = store.begin_turn("c1", actor="user")
    store.close_turn("c1", turn_id=first["turn_id"], state="idle")
    second = store.begin_turn("c1", actor="trainer")

    store.close_turn("c1", turn_id=first["turn_id"], state="idle")

    assert store.state("c1")["state"] == "running"
    assert store.state("c1")["turn_id"] == second["turn_id"]


def test_decision_required_states_accept_a_new_turn(store):
    for state in ("idle", "decision_required", "blocked"):
        turn = store.begin_turn("c1", actor="studio")
        store.close_turn("c1", turn_id=turn["turn_id"], state=state)
        assert store.state("c1")["state"] == state

        reopened = store.begin_turn("c1", actor="user")
        assert reopened["state"] == "running"
        store.close_turn("c1", turn_id=reopened["turn_id"], state="idle")


def test_reconcile_reclaims_only_stale_turns(store):
    store.begin_turn("c1", actor="trainer")

    assert store.reconcile_interrupted(max_age_seconds=3600) == 0
    assert store.state("c1")["state"] == "running"

    assert store.reconcile_interrupted(max_age_seconds=-1) == 1
    state = store.state("c1")
    assert state["state"] == "idle"
    assert state["outcome"] == "interrupted"


def test_legacy_history_migrates_once_with_honest_actors(store):
    history = [
        {"role": "user", "content": "train an agent", "timestamp": "2026-01-01T00:00:00Z"},
        {"role": "user_answers", "answers": {"reward": "win"}, "timestamp": "2026-01-01T00:00:01Z"},
        {"role": "assistant", "content": "which environment?", "timestamp": "2026-01-01T00:00:02Z"},
        {"role": "assistant", "content": "blueprint ready", "timestamp": "2026-01-01T00:00:03Z"},
    ]

    assert store.migrate_legacy("c1", history) == 4
    entries = store.entries("c1")
    assert [(e["actor"], e["kind"]) for e in entries] == [
        ("user", "instruction"),
        ("user", "answer"),
        ("studio", "answer"),
        ("studio", "answer"),
    ]

    # A second load must not duplicate the history.
    assert store.migrate_legacy("c1", history) == 0
    assert store.count("c1") == 4


def test_legacy_questions_are_not_labelled_as_plain_answers(store):
    store.migrate_legacy(
        "c1",
        [
            {
                "role": "assistant",
                "content": "Three things to settle",
                "questions": [{"question": "reward?"}],
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ],
    )

    assert store.entries("c1")[0]["kind"] == "question"


def test_delete_removes_entries_and_turn_state(store):
    store.begin_turn("c1", actor="user")
    store.append("c1", [_entry()], turn_id="t1")

    store.delete("c1")

    assert store.entries("c1") == []
    assert store.state("c1")["state"] == "idle"


def test_entry_anchors_and_actions_round_trip(store):
    store.append(
        "c1",
        [
            _entry(
                actor="gatekeeper",
                kind="scorecard",
                anchor={"round": 3, "metric": 91.5},
                actions=[{"label": "Review evidence", "kind": "open_trust"}],
            )
        ],
        turn_id="t1",
    )

    entry = store.entries("c1")[0]
    assert entry["anchor"] == {"round": 3, "metric": 91.5}
    assert entry["actions"] == [{"label": "Review evidence", "kind": "open_trust"}]
