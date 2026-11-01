"""The unified thread at the API boundary: one turn at a time, durable, replayable.

These tests drive the endpoints the UI actually calls. The studio agent and the LLM gate are
stubbed so each test is about turn ownership and transcript ordering, not about synthesis.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rl_ide.api import server
from rl_ide.architect.omni_agent import StudioChatResponse
from rl_ide.state.conversation_store import ConversationStore
from rl_ide.state.thread_store import ThreadStore


@pytest.fixture
def studio(tmp_path, monkeypatch):
    conv_store = ConversationStore(
        db_path=tmp_path / "conversations.db", workspaces_dir=tmp_path / "workspaces"
    )
    thread_store = ThreadStore(tmp_path / "conversations.db")

    monkeypatch.setattr(server, "get_conversation_store", lambda: conv_store)
    monkeypatch.setattr("rl_ide.state.thread_store._global_thread_store", thread_store)
    monkeypatch.setattr(server, "get_thread_store", lambda: thread_store)
    monkeypatch.setattr(server, "require_verified_llm", lambda: None)

    class _Agent:
        def process_message(self, conversation_id, user_message, **kwargs):
            return StudioChatResponse(
                conversation_id=conversation_id,
                status="clarifying",
                assistant_message=f"Understood: {user_message}",
                questions=[],
            )

    monkeypatch.setattr(server, "global_omni_agent", _Agent())
    return TestClient(server.app), conv_store, thread_store


def test_a_chat_turn_is_recorded_as_one_ordered_thread(studio):
    client, conv_store, thread_store = studio
    conv = conv_store.create_conversation(title="Triage agent")

    resp = client.post(
        f"/api/v4/conversations/{conv['conversation_id']}/chat",
        json={"user_message": "Train an agent to triage tickets"},
    )
    assert resp.status_code == 200, resp.text

    entries = thread_store.entries(conv["conversation_id"])
    # You, then the studio's progress stage, then the studio's reply — one turn, one order.
    assert [(e["actor"], e["kind"]) for e in entries] == [
        ("user", "instruction"),
        ("studio", "stage"),
        ("studio", "answer"),
    ]
    assert [e["seq"] for e in entries] == [1, 2, 3]
    assert entries[0]["body"] == "Train an agent to triage tickets"
    assert "Understood" in entries[2]["body"]
    assert len({e["turn_id"] for e in entries}) == 1
    assert thread_store.state(conv["conversation_id"])["state"] == "idle"


def test_the_second_party_is_refused_while_a_turn_is_running(studio):
    client, conv_store, thread_store = studio
    conv = conv_store.create_conversation(title="Busy")
    cid = conv["conversation_id"]

    thread_store.begin_turn(cid, actor="trainer")

    resp = client.post(
        f"/api/v4/conversations/{cid}/chat",
        json={"user_message": "me too"},
    )
    assert resp.status_code == 409, resp.text
    assert "mid-turn" in resp.json()["detail"]

    # Nothing was appended for the refused party.
    assert thread_store.entries(cid) == []


def test_stop_hands_the_conversation_back(studio):
    client, conv_store, thread_store = studio
    conv = conv_store.create_conversation(title="Stop me")
    cid = conv["conversation_id"]
    thread_store.begin_turn(cid, actor="trainer")

    resp = client.post(f"/api/v4/conversations/{cid}/turn/stop")

    assert resp.status_code == 200
    assert resp.json()["turn"]["state"] == "idle"
    entries = thread_store.entries(cid)
    assert entries[-1]["title"] == "Stopped"
    assert thread_store.state(cid)["state"] == "idle"


def test_thread_endpoint_replays_in_order_and_supports_a_cursor(studio):
    client, conv_store, thread_store = studio
    conv = conv_store.create_conversation(title="Replay")
    cid = conv["conversation_id"]
    thread_store.append(
        cid,
        [
            {"actor": "user", "kind": "instruction", "title": "You", "body": "one"},
            {"actor": "studio", "kind": "answer", "title": "Studio", "body": "two"},
            {"actor": "trainer", "kind": "round", "title": "Round 1", "body": "three"},
        ],
        turn_id="t1",
    )

    full = client.get(f"/api/v4/conversations/{cid}/thread")
    assert full.status_code == 200
    assert [e["body"] for e in full.json()["entries"]] == ["one", "two", "three"]
    assert full.json()["turn"]["state"] == "idle"

    tail = client.get(f"/api/v4/conversations/{cid}/thread", params={"since_seq": 1})
    assert [e["body"] for e in tail.json()["entries"]] == ["two", "three"]


def test_loading_a_conversation_includes_its_thread(studio):
    client, conv_store, thread_store = studio
    conv = conv_store.create_conversation(title="Load")
    cid = conv["conversation_id"]
    thread_store.append(
        cid,
        [{"actor": "user", "kind": "instruction", "title": "You", "body": "hello"}],
        turn_id="t1",
    )

    payload = client.get(f"/api/v4/conversations/{cid}").json()

    assert payload["thread"][0]["body"] == "hello"
    assert payload["turn"]["state"] == "idle"


def test_a_legacy_conversation_is_migrated_without_relabelling(studio):
    client, conv_store, thread_store = studio
    conv_store.create_conversation(title="Legacy")
    cid = "conv_legacy"
    conv_store.save_or_update(
        conversation_id=cid,
        title="Legacy",
        status="clarifying",
        chat_history=[
            {"role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00Z"},
            {"role": "assistant", "content": "hi there", "timestamp": "2026-01-01T00:00:01Z"},
        ],
    )

    payload = client.get(f"/api/v4/conversations/{cid}").json()

    assert [(e["actor"], e["body"]) for e in payload["thread"]] == [
        ("user", "hello"),
        ("studio", "hi there"),
    ]
    assert all(e["title"] != "Architect Evaluation" for e in payload["thread"])


def test_deleting_a_conversation_deletes_its_thread(studio):
    client, conv_store, thread_store = studio
    conv = conv_store.create_conversation(title="Remove")
    cid = conv["conversation_id"]
    thread_store.append(
        cid,
        [{"actor": "user", "kind": "instruction", "title": "You", "body": "gone"}],
        turn_id="t1",
    )

    assert client.delete(f"/api/v4/conversations/{cid}").status_code == 200
    assert thread_store.entries(cid) == []
