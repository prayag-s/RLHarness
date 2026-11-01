"""The product speaks English, whatever language the user writes in.

Every role prompt carries the same rule, and the prompts that are assembled at call time (the
studio planner and the diagnostic assistant) are checked at the call site, because that is where a
future edit would silently drop it.
"""

from __future__ import annotations

import pytest

from rl_ide.architect.omni_agent import StudioOmniAgent
from rl_ide.coach.judge import COACH_JUDGE_SYSTEM_PROMPT
from rl_ide.llm.client import BaseLLMClient, LLMConfig
from rl_ide.llm.prompts import (
    ARENA_BUILDER_SYSTEM_PROMPT,
    ARENA_SYNTHESIZER_SYSTEM_PROMPT,
    AUTO_TUNER_SYSTEM_PROMPT,
    ENV_BUILDER_SYSTEM_PROMPT,
    ENV_SYNTHESIZER_SYSTEM_PROMPT,
    INTAKE_ARCHITECT_SYSTEM_PROMPT,
    with_english_rule,
)

RULE_MARKER = "Write every user-facing word in English"


@pytest.mark.parametrize(
    "prompt",
    [
        INTAKE_ARCHITECT_SYSTEM_PROMPT,
        ENV_SYNTHESIZER_SYSTEM_PROMPT,
        ENV_BUILDER_SYSTEM_PROMPT,
        ARENA_SYNTHESIZER_SYSTEM_PROMPT,
        ARENA_BUILDER_SYSTEM_PROMPT,
        AUTO_TUNER_SYSTEM_PROMPT,
        COACH_JUDGE_SYSTEM_PROMPT,
    ],
    ids=[
        "intake",
        "env-synthesizer",
        "env-builder",
        "arena-synthesizer",
        "arena-builder",
        "auto-tuner",
        "coach-judge",
    ],
)
def test_every_role_prompt_speaks_english(prompt: str):
    assert RULE_MARKER in prompt


def test_the_rule_is_added_once_and_keeps_the_prompt():
    once = with_english_rule("You are a role. Do the thing.")
    assert once.startswith("You are a role. Do the thing.")
    assert once.count("OUTPUT LANGUAGE") == 1
    assert with_english_rule(once) == once


class _RecordingClient(BaseLLMClient):
    """Minimal client that records what the studio agent actually sends."""

    def __init__(self):
        super().__init__(LLMConfig(provider="deepseek", api_key="sk-test-12345678"))
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []

    def generate(self, prompt, system_prompt=None, timeout=None, role=None):  # noqa: ANN001
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return "Here is what happened."

    def generate_json(self, prompt, system_prompt=None, timeout=None, role=None):  # noqa: ANN001
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return {}


def test_the_studio_assistant_asks_for_english():
    client = _RecordingClient()
    agent = StudioOmniAgent(llm_client=client)

    reply = agent._generate_diagnostic_explanation(
        {"title": "Triage agent", "best_metric": 91.0, "current_round": 2, "rounds_history": []},
        "warum ist das so?",
    )

    assert reply == "Here is what happened."
    assert RULE_MARKER in client.prompts[-1]
