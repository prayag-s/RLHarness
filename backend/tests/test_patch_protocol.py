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
