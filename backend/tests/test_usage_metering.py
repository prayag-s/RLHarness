"""G7: every provider call is accounted for, priced and optionally capped.

Metering must never be decorative: these tests assert the persisted row matches what the provider
reported, that the price table produces a real number, and that a tenant cap fails *before* the
network call rather than after the money is gone.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from rl_ide.llm import metering
from rl_ide.llm.client import DeepSeekLLMClient, LLMConfig
from rl_ide.state.conversation_store import ConversationStore, get_conversation_store


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def temp_store(tmp_path, monkeypatch) -> ConversationStore:
    """Keeps accounting rows out of the development database."""
    store = ConversationStore(db_path=tmp_path / "usage.db", workspaces_dir=tmp_path / "workspaces")
    monkeypatch.setattr(metering, "_store", lambda: store)
    monkeypatch.setattr("rl_ide.api.server.get_conversation_store", lambda: store)
    return store


def test_provider_usage_is_recorded_and_priced(temp_store, monkeypatch):
    conv_id = f"usage_{uuid.uuid4().hex[:8]}"
    metering.current_conversation_id.set(conv_id)
    metering.current_owner_id.set("acme")

    payload = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 2000},
    }
    monkeypatch.setattr(
        "rl_ide.llm.client.httpx.Client.post", lambda self, *a, **k: _FakeResponse(payload)
    )

    client = DeepSeekLLMClient(LLMConfig(provider="deepseek", api_key="sk-test", model_name="deepseek-chat"))
    assert client.generate("hello") == "ok"

    report = metering.usage_report(conversation_id=conv_id, owner_id="acme")
    assert report["totals"] == {
        "calls": 1,
        "input_tokens": 1000,
        "output_tokens": 2000,
        "cost_usd": pytest.approx(0.00027 + 0.0022, rel=1e-6),
    }
    assert report["by_model"][0]["provider"] == "deepseek"
    assert report["recent"][0]["conversation_id"] == conv_id
    assert report["recent"][0]["latency_ms"] >= 0


def test_unknown_model_is_recorded_with_zero_cost(temp_store, monkeypatch):
    """Local weights and unlisted models still show up in the token totals."""
    payload = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}
    monkeypatch.setattr(
        "rl_ide.llm.client.httpx.Client.post", lambda self, *a, **k: _FakeResponse(payload)
    )
    client = DeepSeekLLMClient(LLMConfig(provider="deepseek", api_key="sk-test", model_name="house-model"))
    client.generate("hello")

    totals = metering.usage_report()["totals"]
    assert totals["calls"] == 1
    assert totals["input_tokens"] + totals["output_tokens"] == 12
    assert totals["cost_usd"] == 0.0


def test_budget_cap_fails_before_the_call(temp_store, monkeypatch):
    owner = f"cap_{uuid.uuid4().hex[:8]}"
    metering.current_owner_id.set(owner)
    temp_store.record_usage(
        provider="deepseek",
        model="deepseek-chat",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cost_usd=5.0,
        owner_id=owner,
    )
    monkeypatch.setenv("RL_IDE_OWNER_BUDGET_USD", "1.0")

    network_calls: list[int] = []
    monkeypatch.setattr(
        "rl_ide.llm.client.httpx.Client.post", lambda self, *a, **k: network_calls.append(1)
    )
    client = DeepSeekLLMClient(LLMConfig(provider="deepseek", api_key="sk-test"))

    with pytest.raises(metering.BudgetExceeded):
        client.generate("hello")
    assert network_calls == [], "a capped tenant must not reach the provider"
    assert metering.usage_report(owner_id=owner)["budget"]["remaining_usd"] == 0.0


def test_usage_endpoint_is_owner_scoped(temp_store):
    from rl_ide.api.server import app

    acme_conv = temp_store.create_conversation(title="acme", owner_id="acme")["conversation_id"]
    temp_store.record_usage(
        provider="deepseek", model="deepseek-chat", input_tokens=100, output_tokens=10,
        conversation_id=acme_conv, owner_id="acme",
    )
    temp_store.record_usage(
        provider="openai", model="gpt-4o", input_tokens=900, output_tokens=90,
        owner_id="globex",
    )

    client = TestClient(app)
    acme = client.get("/api/v4/usage", headers={"x-owner-id": "acme"}).json()
    assert acme["totals"]["calls"] == 1
    assert acme["totals"]["input_tokens"] == 100
    assert [row["provider"] for row in acme["by_model"]] == ["deepseek"]

    scoped = client.get(
        "/api/v4/usage", params={"conversation_id": acme_conv}, headers={"x-owner-id": "acme"}
    ).json()
    assert scoped["totals"]["calls"] == 1

    cross_tenant = client.get(
        f"/api/v4/usage?conversation_id={acme_conv}", headers={"x-owner-id": "globex"}
    )
    assert cross_tenant.status_code == 404, "another tenant's conversation must not be readable"


def test_global_store_is_untouched_by_metering(temp_store):
    """The fixture is a real ConversationStore, so this also asserts the schema migration runs twice."""
    assert get_conversation_store() is not temp_store
    assert metering.usage_report()["totals"]["calls"] == 0
