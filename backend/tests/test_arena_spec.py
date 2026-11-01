"""Spec-driven arena synthesis: the model styles, code renders.

Asking a model for a whole HTML document costs several thousand output tokens per build and again
per restyle, and the document can only be kept if a JavaScript engine validates it. Spec mode asks
for a ~150-token styling contract instead, and the arena we already render from the blueprint
applies it, so every document keeps the hooks the iframe bridge depends on.
"""

from __future__ import annotations

import json

import pytest

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.llm.client import BaseLLMClient, LLMConfig
from rl_ide.synthesizer.arena_builder import (
    JITArenaSynthesizer,
    arena_mode,
    read_arena_spec,
    synthesize_universal_arena_html,
)

SPEC = {
    "headline": "Queue Arena",
    "accent": "violet",
    "theme": "mono",
    "scoreboard": True,
    "stage": "large",
    "caption": "four routing queues",
    "chips": ["SLA", "Backlog"],
}

HOOKS = ("ARENA_UPDATE", "startMockAnimationLoop", "sendInteractiveAction", "renderDomainState")


class _SpecClient(BaseLLMClient):
    """Answers every call with a JSON spec, and records what it was asked for."""

    def __init__(self, payload: dict | None = None):
        super().__init__(LLMConfig(provider="deepseek", api_key="sk-test", model_name="deepseek-flash"))
        self.payload = payload if payload is not None else SPEC
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt, system_prompt=None, timeout=None, role=None):  # noqa: ANN001
        self.calls.append((prompt, role))
        return json.dumps(self.payload)


def _blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Ticket Router",
        domain_summary="Route tickets",
        data_source_type="simulated",
        observation_features=["queue", "age"],
        action_labels=["Tier 1", "Senior", "Retention"],
        arena_visual_description="dispatch desk",
        interactive_mode_description="route a ticket",
        success_metric_name="Routing Accuracy (%)",
    )


def test_spec_mode_is_the_default(monkeypatch):
    monkeypatch.delenv("RL_IDE_ARENA_MODE", raising=False)
    assert arena_mode() == "spec"
    monkeypatch.setenv("RL_IDE_ARENA_MODE", "nonsense")
    assert arena_mode() == "spec"


def test_the_model_is_asked_for_a_contract_not_a_document():
    client = _SpecClient()
    outcome: dict = {}
    html = synthesize_universal_arena_html(
        _blueprint(), {"queue": ["a", "b"]}, client=client, outcome=outcome
    )

    assert outcome["source"] == "llm"
    assert len(client.calls) == 1
    prompt, role = client.calls[0]
    assert role == "arena_ui"
    assert "Return strict JSON only" in prompt
    assert "arena.html" not in prompt

    for hook in HOOKS:
        assert hook in html
    assert "Queue Arena" in html
    assert "violet" in html
    assert "font-mono" in html
    assert "SLA" in html


def test_the_spec_survives_a_round_trip():
    client = _SpecClient()
    html = synthesize_universal_arena_html(_blueprint(), {"queue": []}, client=client)
    assert read_arena_spec(html)["accent"] == "violet"


def test_a_restyle_returns_a_new_contract_and_rerenders(monkeypatch):
    client = _SpecClient()
    synth = JITArenaSynthesizer(client)
    current = synth.generate_arena_html(_blueprint(), {"queue": []})

    client.payload = {**SPEC, "accent": "amber", "headline": "Warm Queue"}
    updated = synth.update_arena(_blueprint(), current, "make it warm and amber")

    assert synth.last_source == "llm"
    assert "amber" in updated
    assert "Warm Queue" in updated
    assert "violet" not in updated

    prompt, role = client.calls[-1]
    assert role == "arena_ui"
    assert "violet" in prompt, "the previous contract must be sent instead of the whole document"
    assert "<html" not in prompt


def test_deterministic_mode_never_calls_the_provider(monkeypatch):
    monkeypatch.setenv("RL_IDE_ARENA_MODE", "deterministic")

    class _NeverCalled(_SpecClient):
        def generate(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("deterministic mode must not spend tokens")

    outcome: dict = {}
    html = synthesize_universal_arena_html(
        _blueprint(), {"queue": []}, client=_NeverCalled(), outcome=outcome
    )
    assert outcome["source"] == "fallback"
    for hook in HOOKS:
        assert hook in html


def test_an_invalid_answer_falls_back_without_losing_the_preview():
    client = _SpecClient()
    client.payload = {"accent": "not-a-colour", "theme": "loud"}
    outcome: dict = {}
    html = synthesize_universal_arena_html(
        _blueprint(), {"queue": []}, client=client, outcome=outcome
    )
    # The contract normalises to the defaults rather than injecting arbitrary CSS.
    for hook in HOOKS:
        assert hook in html
    assert outcome["source"] == "llm"
