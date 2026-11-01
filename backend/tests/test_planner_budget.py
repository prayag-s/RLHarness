"""The per-message planner must not be the default cost of chatting.

Every chat message used to spend a provider call to classify intent, even though the local grammar
already routes the same requests. ``auto`` keeps the model for the one case the grammar cannot
decide: a directive that matches no target at all.
"""

from __future__ import annotations

import json

import pytest

from rl_ide.architect.models import ExecutionTarget
from rl_ide.architect.omni_agent import StudioOmniAgent, planner_mode
from rl_ide.llm.client import BaseLLMClient, LLMConfig


class _RecordingClient(BaseLLMClient):
    def __init__(self, answer: dict | None = None):
        super().__init__(LLMConfig(provider="deepseek", api_key="sk-test", model_name="deepseek-flash"))
        self.answer = answer or {
            "supported": True,
            "unsupported_reason": None,
            "scope_classification": "arena_only",
            "targets": ["arena_visual_ui"],
            "requires_retrain": False,
            "surgical_instructions": "Restyle the arena.",
            "user_facing_summary": "Restyling.",
        }
        self.calls: list[str | None] = []

    def generate(self, prompt, system_prompt=None, timeout=None, role=None):  # noqa: ANN001
        self.calls.append(role)
        return json.dumps(self.answer)

    def generate_json(self, prompt, system_prompt=None, timeout=None, role=None):  # noqa: ANN001
        self.calls.append(role)
        return self.answer


def _agent(client: BaseLLMClient) -> StudioOmniAgent:
    return StudioOmniAgent(llm_client=client)


def test_planner_mode_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("RL_IDE_PLANNER_LLM", raising=False)
    assert planner_mode() == "auto"
    monkeypatch.setenv("RL_IDE_PLANNER_LLM", "off")
    assert planner_mode() == "off"


def test_a_routable_message_costs_nothing(monkeypatch):
    monkeypatch.delenv("RL_IDE_PLANNER_LLM", raising=False)
    client = _RecordingClient()
    plan = _agent(client).plan_chat_action("randomize the first move among good options")

    assert client.calls == []
    assert plan.targets == [ExecutionTarget.INFERENCE_STRATEGY]
    assert plan.requires_retrain is False


def test_an_unroutable_directive_asks_the_model_once(monkeypatch):
    monkeypatch.delenv("RL_IDE_PLANNER_LLM", raising=False)
    client = _RecordingClient()
    plan = _agent(client).plan_chat_action("do the thing we discussed yesterday")

    assert client.calls == ["chat_planner"]
    assert plan.targets == [ExecutionTarget.ARENA_VISUAL_UI]


def test_off_never_spends_a_call(monkeypatch):
    monkeypatch.setenv("RL_IDE_PLANNER_LLM", "off")
    client = _RecordingClient()
    plan = _agent(client).plan_chat_action("do the thing we discussed yesterday")

    assert client.calls == []
    assert plan.targets == []


def test_always_restores_the_previous_behaviour(monkeypatch):
    monkeypatch.setenv("RL_IDE_PLANNER_LLM", "always")
    client = _RecordingClient()
    plan = _agent(client).plan_chat_action("randomize the first move among good options")

    assert client.calls == ["chat_planner"]
    assert plan.targets == [ExecutionTarget.ARENA_VISUAL_UI]


def test_unsupported_capabilities_still_short_circuit(monkeypatch):
    monkeypatch.setenv("RL_IDE_PLANNER_LLM", "always")
    client = _RecordingClient()
    plan = _agent(client).plan_chat_action("stream 3D WebGL animations to Discord")

    assert client.calls == []
    assert plan.supported is False
    assert ExecutionTarget.UNSUPPORTED_CAPABILITY in plan.targets
