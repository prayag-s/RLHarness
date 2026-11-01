"""Function-level patching replaces one function and leaves the module untouched.

The auto-tuner used to re-emit the entire environment module to change a reward term. That is both
the largest line on the bill and a fresh chance to break working code, so these tests pin that only
the target function can change.
"""

from __future__ import annotations

import ast

import pytest

from rl_ide.autopilot.auto_tuner import AutoTunerAgent
from rl_ide.synthesizer.patch import PatchError, extract_function_block, replace_function
from rl_ide.synthesizer.verifier import repair_attempts

MODULE = '''"""Synthesized environment."""

import numpy as np


class SynthesizedGymEnv:
    """Minimal environment used to exercise patching."""

    def __init__(self):
        self.moves = 0

    def compute_reward(self, state, action, next_state, info):
        # ORIGINAL BODY
        return float(state["x"]), {"win": float(state["x"])}

    def step(self, action):
        self.moves += 1
        return None
'''

PATCH = '''def compute_reward(self, state, action, next_state, info):
    win = 1.0 if next_state["done"] else 0.0
    return win, {"win": win}
'''


def test_only_the_target_function_changes():
    updated = replace_function(MODULE, PATCH, "compute_reward")

    assert "ORIGINAL BODY" not in updated
    assert 'next_state["done"]' in updated
    assert "def step(self, action):" in updated
    assert "class SynthesizedGymEnv:" in updated
    assert ast.parse(updated)
    # Everything outside the replaced function is byte-identical.
    before = MODULE.split("    def compute_reward")[0]
    assert updated.startswith(before)


def test_a_method_stays_a_method():
    updated = replace_function(MODULE, PATCH, "compute_reward")
    assert "\n    def compute_reward(self, state, action, next_state, info):" in updated


def test_patch_for_another_function_is_refused():
    with pytest.raises(PatchError):
        replace_function(MODULE, "def compute_other(self):\n    return 1\n", "compute_reward")


def test_patch_with_extra_statements_is_refused():
    with pytest.raises(PatchError):
        replace_function(MODULE, PATCH + "\nprint('side effect')\n", "compute_reward")


def test_unknown_target_is_refused():
    with pytest.raises(PatchError):
        replace_function(MODULE, PATCH, "no_such_method")


def test_function_block_is_extracted_from_a_fenced_answer():
    raw = "Here is the fix:\n```python\n" + PATCH + "```\nDone."
    block = extract_function_block(raw, "compute_reward")
    assert block is not None and block.startswith("def compute_reward")


def test_a_whole_module_answer_is_not_mistaken_for_a_patch():
    raw = "```python\n" + MODULE + "```"
    assert extract_function_block(raw, "compute_reward") is None


def test_auto_tuner_splices_a_function_only_answer():
    raw = "```python\n" + PATCH + "```"
    patched = AutoTunerAgent._apply_reward_patch(MODULE, raw)
    assert patched is not None
    assert "ORIGINAL BODY" not in patched
    assert "def step(self, action):" in patched


def test_repair_attempts_are_bounded_and_overridable(monkeypatch):
    monkeypatch.delenv("RL_IDE_VERIFY_MAX_REPAIRS", raising=False)
    monkeypatch.delenv("RL_IDE_TUNE_MAX_REPAIRS", raising=False)
    assert repair_attempts("synthesize") == 2
    assert repair_attempts("tune") == 1

    monkeypatch.setenv("RL_IDE_TUNE_MAX_REPAIRS", "0")
    assert repair_attempts("tune") == 0


# --- targeted verification repair -------------------------------------------

from rl_ide.architect.models import AgentBlueprintSpec  # noqa: E402
from rl_ide.llm.client import BaseLLMClient, LLMConfig  # noqa: E402
from rl_ide.synthesizer.verifier import SandboxSmokeTester, functions_in_traceback  # noqa: E402

TRACEBACK = '''Traceback (most recent call last):
  File "/tmp/attempt.py", line {line}, in step
    raise ValueError("boom")
ValueError: boom'''


class _RepairClient(BaseLLMClient):
    def __init__(self, answer: str):
        super().__init__(LLMConfig(provider="deepseek", api_key="sk-test", model_name="deepseek-flash"))
        self.answer = answer
        self.prompts: list[str] = []
        self.roles: list[str | None] = []

    def generate(self, prompt, system_prompt=None, timeout=None, role=None):  # noqa: ANN001
        self.prompts.append(prompt)
        self.roles.append(role)
        return self.answer


def _traceback_at(code: str, marker: str) -> str:
    line = next(i for i, text in enumerate(code.splitlines(), 1) if marker in text)
    return TRACEBACK.format(line=line)


def _verifier(client: BaseLLMClient) -> SandboxSmokeTester:
    return SandboxSmokeTester(llm_client=client)


def _blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="T",
        domain_summary="d",
        data_source_type="simulated",
        observation_features=["a"],
        action_labels=["x", "y"],
        arena_visual_description="stage",
        interactive_mode_description="click",
        success_metric_name="Win (%)",
    )


def test_traceback_names_the_innermost_function():
    code = "\n".join(
        [
            "class E:",
            "    def reset(self):",
            "        return 1",
            "",
            "    def step(self, action):",
            "        raise ValueError('boom')",
            "",
            "    def compute_reward(self, s, a, ns, info):",
            "        return 0.0, {}",
        ]
    )
    assert functions_in_traceback(_traceback_at(code, "raise ValueError"), code) == ["step"]


def test_repair_returns_only_the_failing_function():
    client = _RepairClient("```python\ndef step(self, action):\n    return None\n```")
    repaired = _verifier(client)._repair_code(
        MODULE, "boom", _traceback_at(MODULE, "self.moves += 1"), _blueprint(), None
    )

    assert "def step(self, action):\n        return None" in repaired
    assert "ORIGINAL BODY" in repaired, "the working reward function must not be re-rolled"
    assert "class SynthesizedGymEnv:" in repaired
    assert client.roles == ["verifier_repair"]
    assert "Return ONLY the replacement definition" in client.prompts[0]


def test_a_module_answer_is_accepted_without_a_second_call():
    client = _RepairClient("```python\n" + MODULE.replace("ORIGINAL BODY", "REPAIRED") + "```")
    repaired = _verifier(client)._repair_code(
        MODULE, "boom", _traceback_at(MODULE, "self.moves += 1"), _blueprint(), None
    )

    assert "REPAIRED" in repaired
    assert len(client.prompts) == 1


def test_an_unusable_answer_falls_back_to_a_whole_module_repair():
    client = _RepairClient("I could not find the bug.")
    _verifier(client)._repair_code(
        MODULE, "boom", _traceback_at(MODULE, "self.moves += 1"), _blueprint(), None
    )

    assert len(client.prompts) == 2
    assert "complete corrected Python code" in client.prompts[1]


def test_an_unmappable_traceback_skips_the_targeted_attempt():
    client = _RepairClient("```python\n" + MODULE + "```")
    syntax = '  File "/tmp/x.py", line 3\nSyntaxError: invalid syntax'
    _verifier(client)._repair_code(MODULE, "syntax", syntax, _blueprint(), None)

    assert len(client.prompts) == 1
    assert "complete corrected Python code" in client.prompts[0]
