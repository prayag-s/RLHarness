"""Verification test suite for Blackjack 21 Card Agent Intake, Arena Synthesis, and Gym Environment."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from rl_ide.architect.intake import (
    is_blackjack_intent,
    is_blackjack_spec,
    IntakeArchitectAgent,
)
from rl_ide.synthesizer.arena_builder import JITArenaSynthesizer
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer
from rl_ide.synthesizer.verifier import SandboxSmokeTester
from rl_ide.architect.omni_agent import StudioOmniAgent


def test_blackjack_intent_and_blueprint_synthesis():
    """Verifies that Blackjack prompts produce a tailored Blackjack blueprint, not Autonomous Task Synthesizer."""
    prompt = "train a blackjack playing agent which winws everyttime"
    assert is_blackjack_intent(prompt)
    assert is_blackjack_intent("arena should have blackjack table etc")
    assert is_blackjack_intent("card game with dealer and hit or stand")

    agent = IntakeArchitectAgent()
    resp = agent._deterministic_intake(
        prompt=prompt,
        dataset=None,
        answers={},
        wants_defaults=True,
    )
    bp = resp.blueprint_spec
    assert bp is not None
    assert is_blackjack_spec(bp)
    assert bp.task_title == "Blackjack 21 Card Agent"
    assert "Hit" in " ".join(bp.action_labels)
    assert "Stand" in " ".join(bp.action_labels)
    assert "Double Down" in " ".join(bp.action_labels)
    assert "player_total" in bp.observation_features
    assert "dealer_upcard_value" in bp.observation_features


def test_blackjack_arena_synthesis():
    """Verifies that Blackjack generates an emerald felt casino table arena with cards, not a 2D rover canvas."""
    agent = IntakeArchitectAgent()
    resp = agent._deterministic_intake(
        prompt="train a blackjack playing agent",
        dataset=None,
        answers={},
        wants_defaults=True,
    )
    bp = resp.blueprint_spec

    arena_builder = JITArenaSynthesizer()
    with tempfile.TemporaryDirectory() as td:
        html, _ = arena_builder.synthesize(blueprint=bp, workspace_dir=td)
        assert "dealer-cards" in html
        assert "player-cards" in html
        assert "BLACKJACK PAYS 3 TO 2" in html
        assert "Hit (+1 Card)" in html
        assert "Stand (Hold Hand)" in html
        assert "Double Down (2x)" in html
        assert "startMockAnimationLoop" in html
        assert "ARENA_UPDATE" in html
        assert "Autonomous Task Synthesizer" not in html
        assert "sim-canvas" not in html


def test_blackjack_env_sandbox_smoke_test():
    """Verifies that synthesized Blackjack Gymnasium environment satisfies all 6 smoke test requirements."""
    agent = IntakeArchitectAgent()
    resp = agent._deterministic_intake(
        prompt="train a blackjack playing agent",
        dataset=None,
        answers={},
        wants_defaults=True,
    )
    bp = resp.blueprint_spec

    env_builder = JITEnvironmentSynthesizer()
    with tempfile.TemporaryDirectory() as td:
        code_str, _, env_cls = env_builder.synthesize(blueprint=bp, workspace_dir=td)
        assert "class SynthesizedGymEnv" in code_str

        tester = SandboxSmokeTester()
        smoke_res = tester.run_smoke_test(env_cls)
        assert smoke_res["verified"] is True
        assert smoke_res["obs_dim"] == 4
        assert smoke_res["n_actions"] == 3
        assert "player_cards" in smoke_res["sample_arena_state"]
        assert "dealer_cards" in smoke_res["sample_arena_state"]


def test_blackjack_omni_chat_flow():
    """Verifies that Omni-Agent creates and maintains Blackjack table across chat turns."""
    omni = StudioOmniAgent()
    cid = "test_bj_chat_regress"

    # Turn 1: Initial Blackjack prompt
    r1 = omni.process_message(
        conversation_id=cid,
        user_message="train a blackjack playing agent which winws everyttime"
    )
    assert r1.blueprint_spec is not None
    assert r1.blueprint_spec.task_title == "Blackjack 21 Card Agent"
    assert "dealer-cards" in r1.arena_html
    assert "player-cards" in r1.arena_html

    # Turn 2: User requests table / visual enhancement
    r2 = omni.process_message(
        conversation_id=cid,
        user_message="arena should have blackjack table etc"
    )
    assert "dealer-cards" in r2.arena_html
    assert "player-cards" in r2.arena_html
    assert "Autonomous Task Synthesizer" not in r2.arena_html
