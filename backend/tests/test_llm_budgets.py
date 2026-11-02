"""Cost-control contract: every call declares a budget, and the bill can be attributed.

Reasoning tokens are billed as output and DeepSeek enables them by default. These tests pin the
two guarantees that keep a run cheap: a role never gets an unbounded output budget, and a role that
only classifies never pays for chain-of-thought.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rl_ide.llm import metering
from rl_ide.llm.client import DeepSeekLLMClient, LLMConfig
from rl_ide.llm.roles import max_tokens_for, profile_for, timeout_for
from rl_ide.state.conversation_store import ConversationStore


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def captured(monkeypatch) -> list[dict]:
    """Captures the request body the client would send, without touching the network."""
    bodies: list[dict] = []

    def _post(self, url, headers=None, json=None, **kwargs):  # noqa: ANN001, ANN002, ANN003
        bodies.append(json)
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "completion_tokens_details": {"reasoning_tokens": 25},
                    "prompt_cache_hit_tokens": 60,
                },
            }
        )

    monkeypatch.setattr("rl_ide.llm.client.httpx.Client.post", _post)
    return bodies


def _client() -> DeepSeekLLMClient:
    return DeepSeekLLMClient(
        LLMConfig(provider="deepseek", api_key="sk-test", model_name="deepseek-flash")
    )


def test_classification_roles_do_not_pay_for_reasoning(captured):
    _client().generate("plan this", role="chat_planner")
    body = captured[-1]
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    assert body["max_tokens"] <= 700


def test_code_roles_emit_code_instead_of_reasoning(captured):
    """A module plus chain-of-thought does not fit one ceiling.

    deepseek-chat ignores `reasoning_effort`: an env synthesis call spent all 8000 tokens on
    reasoning (`reasoning_tokens == output_tokens`) and returned an empty, truncated answer.
    """
    _client().generate("write the env", role="env_synthesizer")
    body = captured[-1]
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    assert body["max_tokens"] == profile_for("env_synthesizer").max_tokens
    assert body["max_tokens"] <= 8000

    _client().generate("repair the env", role="verifier_repair")
    assert captured[-1]["thinking"] == {"type": "disabled"}


def test_configured_token_limit_cannot_be_exceeded(captured):
    """A large provider config must not become a large per-call budget."""
    client = DeepSeekLLMClient(
        LLMConfig(provider="deepseek", api_key="sk-test", model_name="deepseek-flash", max_tokens=384000)
    )
    client.generate("write the env", role="env_synthesizer")
    assert captured[-1]["max_tokens"] == profile_for("env_synthesizer").max_tokens
    assert max_tokens_for("env_synthesizer", 500) == 500


def test_global_switch_disables_reasoning_everywhere(captured, monkeypatch):
    monkeypatch.setenv("RL_IDE_LLM_THINKING", "0")
    _client().generate("write the env", role="env_synthesizer")
    body = captured[-1]
    assert body["thinking"] == {"type": "disabled"}
    assert profile_for("env_synthesizer").thinking is False


def test_a_large_budget_gets_time_to_finish():
    """A ceiling the request cannot reach is just a paid timeout."""
    assert timeout_for("env_synthesizer", None, 120.0) == 300.0
    assert timeout_for("verifier_repair", None, 120.0) == 240.0
    assert timeout_for("chat_planner", None, 120.0) == 120.0
    assert timeout_for("env_synthesizer", 45.0, 120.0) == 45.0


def test_unknown_role_is_still_bounded(captured):
    _client().generate("anything")
    assert captured[-1]["max_tokens"] == profile_for(None).max_tokens


def test_reasoning_and_cached_tokens_are_recorded(tmp_path, monkeypatch):
    store = ConversationStore(db_path=tmp_path / "usage.db", workspaces_dir=tmp_path / "ws")
    monkeypatch.setattr(metering, "_store", lambda: store)

    row = metering.capture_response(
        "deepseek",
        "deepseek-flash",
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 400,
                "completion_tokens_details": {"reasoning_tokens": 250},
                "prompt_cache_hit_tokens": 600,
            }
        },
        started_at=0.0,
        role="env_synthesizer",
    )

    assert row["role"] == "env_synthesizer"
    assert row["reasoning_tokens"] == 250
    assert row["cached_input_tokens"] == 600

    summary = store.usage_summary(limit=5)
    assert summary["efficiency"]["reasoning_tokens"] == 250
    assert summary["efficiency"]["cache_hit_rate"] == 0.6
    assert summary["by_role"][0]["role"] == "env_synthesizer"
    assert summary["recent"][0]["reasoning_tokens"] == 250


def test_offpeak_pricing_is_half_of_peak():
    peak = datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc)      # Monday, inside 01:00-04:00
    offpeak = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)   # Monday, outside peak windows
    assert metering.is_deepseek_offpeak(peak) is False
    assert metering.is_deepseek_offpeak(offpeak) is True
    assert metering.is_deepseek_offpeak(datetime(2026, 11, 7, 2, 0, tzinfo=timezone.utc)) is True

    peak_cost = metering.estimate_cost("deepseek", "deepseek-flash", 1_000_000, 0, at=peak)
    offpeak_cost = metering.estimate_cost("deepseek", "deepseek-flash", 1_000_000, 0, at=offpeak)
    assert peak_cost == pytest.approx(0.30)
    assert offpeak_cost == pytest.approx(0.15)


def test_cached_input_is_priced_below_a_miss():
    at = datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc)
    miss = metering.estimate_cost("deepseek", "deepseek-flash", 1_000_000, 0, at=at)
    hit = metering.estimate_cost(
        "deepseek", "deepseek-flash", 1_000_000, 0, cached_input_tokens=1_000_000, at=at
    )
    assert hit < miss / 10
