"""Verification test suite for Universal Generative Pipeline & Agentic Chat Planner.

Tests strictly verify the elimination of hardcoded domain templates and static keyword lists:
1. Zero-hardcoding novel domain synthesis (Connect-4 / novel domain) without rover fallback.
2. Inference strategy patch for opening move randomization in interactive_step().
3. Multi-target code and UI synchronization across architectural layers.
4. Unsupported capability honest refusal with zero file mutations.
5. Codebase audit ensuring omni_agent routing delegates to agentic planner without keyword lists.
"""

import ast
import inspect
import uuid
from pathlib import Path
import numpy as np
import pytest

from rl_ide.architect.models import (
    DynamicTaskBlueprint,
    ExecutionTarget,
    ChatActionPlan,
)
from rl_ide.architect.intake import (
    synthesize_blueprint_from_prompt,
    IntakeArchitectAgent,
)
from rl_ide.synthesizer.env_builder import (
    synthesize_universal_env,
    JITEnvironmentSynthesizer,
    load_synthesized_env_class,
)
from rl_ide.synthesizer.arena_builder import (
    synthesize_universal_arena_html,
    JITArenaSynthesizer,
)
from rl_ide.synthesizer.verifier import SandboxSmokeTester
from rl_ide.architect.omni_agent import StudioOmniAgent, StudioActionType
from rl_ide.llm.client import FallbackCodeSynthesizer


def test_zero_hardcoding_novel_domain_synthesis(tmp_path):
    """Test 1: Novel domain synthesis without hardcoded templates or 2D rover fallback."""
    # Novel prompt: Connect 4
    prompt = "Train a Connect-4 grid agent that drops discs into 7 columns"
    blueprint = synthesize_blueprint_from_prompt(prompt)

    assert isinstance(blueprint, DynamicTaskBlueprint)
    assert "rover" not in blueprint.domain_title.lower()
    assert "waypoint" not in blueprint.domain_title.lower()

    # Assert observation features and actions are domain-specific
    obs_names = [f.name for f in blueprint.observation_features]
    action_labels = [a.label for a in blueprint.actions]

    assert not any("state_pos_x" in name for name in obs_names)
    assert not any("Move Left" in lbl for lbl in action_labels)
    assert len(action_labels) == 7
    assert any("col" in lbl.lower() for lbl in action_labels)

    # Synthesize Gymnasium environment
    env_code = synthesize_universal_env(blueprint, user_prompt=prompt)
    assert "class SynthesizedGymEnv" in env_code
    assert "get_render_state" in env_code
    assert "interactive_step" in env_code

    # Verify via SandboxSmokeTester
    smoke_tester = SandboxSmokeTester()
    verified, healed_code, logs = smoke_tester.verify_and_heal(
        code_str=env_code,
        blueprint=blueprint.to_agent_blueprint_spec(),
        workspace_dir=tmp_path,
    )
    assert verified is True
    env_cls = load_synthesized_env_class(tmp_path / "env_module.py")
    smoke_res = smoke_tester.run_smoke_test(env_cls)
    assert smoke_res.get("verified") is True

    # Synthesize Universal Arena HTML
    arena_html = synthesize_universal_arena_html(
        blueprint=blueprint,
        sample_render_state=smoke_res.get("sample_arena_state") or {"board": [[0] * 7 for _ in range(6)]},
    )

    assert "sim-canvas" not in arena_html
    assert "ARENA_UPDATE" in arena_html
    assert "startMockAnimationLoop" in arena_html
    assert ("c4-grid" in arena_html or "dropDisc" in arena_html or "Connect 4" in arena_html or "dynamic-telemetry-grid" in arena_html)

    # Test with arbitrary novel domain: Warehouse Elevator Dispatcher
    novel_prompt = "Train a warehouse elevator dispatcher agent managing 3 elevator cars across 10 floors"
    novel_bp = synthesize_blueprint_from_prompt(novel_prompt)
    assert isinstance(novel_bp, DynamicTaskBlueprint)
    assert "rover" not in novel_bp.domain_title.lower()

    novel_env_code = synthesize_universal_env(novel_bp, user_prompt=novel_prompt)
    novel_dir = tmp_path / "novel"
    verified2, _, logs2 = smoke_tester.verify_and_heal(
        code_str=novel_env_code,
        blueprint=novel_bp.to_agent_blueprint_spec(),
        workspace_dir=novel_dir,
    )
    assert verified2 is True
    novel_env_cls = load_synthesized_env_class(novel_dir / "env_module.py")
    smoke_res2 = smoke_tester.run_smoke_test(novel_env_cls)
    assert smoke_res2.get("verified") is True
    novel_arena_html = synthesize_universal_arena_html(novel_bp, sample_render_state=smoke_res2.get("sample_arena_state"))
    assert "sim-canvas" not in novel_arena_html
    assert "dynamic-telemetry-grid" in novel_arena_html or "ARENA_UPDATE" in novel_arena_html


def test_first_move_randomization_inference_patch(tmp_path):
    """Test 2: Opening move randomization routes to INFERENCE_STRATEGY and patches interactive_step."""
    agent = StudioOmniAgent()
    user_prompt = "if agent is first, then randomize its first move among good options"

    plan = agent.plan_chat_action(user_prompt)
    assert plan.supported is True
    assert ExecutionTarget.INFERENCE_STRATEGY in plan.targets

    # Synthesize TicTacToe environment to get live baseline source code
    intake_agent = IntakeArchitectAgent()
    intake_resp = intake_agent.interview("Train a competitive Tic-Tac-Toe agent", wants_defaults=True)
    bp = intake_resp.blueprint_spec

    env_synthesizer = JITEnvironmentSynthesizer()
    base_code, env_path, _ = env_synthesizer.synthesize(bp, workspace_dir=tmp_path)

    # Apply surgical patch to interactive_step
    patched_code = agent._patch_interactive_inference(base_code, user_prompt, plan=plan)
    assert patched_code != base_code
    assert "good_moves" in patched_code
    assert "np.random.choice(good_moves)" in patched_code

    # Write patched code and load class
    env_file = tmp_path / "env_module.py"
    env_file.write_text(patched_code, encoding="utf-8")
    env_cls = load_synthesized_env_class(env_file)
    env = env_cls()

    # Policy that recommends move 4 (center) with high probability
    def deterministic_policy(obs):
        probs = np.array([0.1, 0.05, 0.1, 0.05, 0.4, 0.05, 0.1, 0.05, 0.1], dtype=np.float32)
        return 4, probs

    # Run 30 simulated opening turns with agent going first
    chosen_openings = []
    for _ in range(30):
        res = env.interactive_step({"reset": True, "human_first": False}, deterministic_policy)
        chosen_openings.append(env.last_move)

    # Agent must select from good strategic openings and NOT make only a single static move
    distinct_moves = set(chosen_openings)
    assert len(distinct_moves) >= 2, f"Expected at least 2 distinct opening moves across 30 turns, got: {distinct_moves}"
    # Strategic openings in Tic-Tac-Toe are corners [0, 2, 6, 8] and center [4]
    valid_strategic_openings = {0, 2, 4, 6, 8}
    assert distinct_moves.issubset(valid_strategic_openings), f"Openings {distinct_moves} not in {valid_strategic_openings}"


def test_multi_target_code_and_ui_synchronization(tmp_path):
    """Test 3: Multi-target prompt synchronizes reward/env code and visual arena UI."""
    agent = StudioOmniAgent()
    compound_prompt = "Add a defensive corner reward bonus and make the visual arena display a golden scoreboard"

    plan = agent.plan_chat_action(compound_prompt)
    assert plan.supported is True
    assert ExecutionTarget.REWARD_SHAPING in plan.targets
    assert ExecutionTarget.ARENA_VISUAL_UI in plan.targets
    assert plan.requires_retrain is True

    # Test execution through process_message
    conv_id = f"test_multi_target_sync_{uuid.uuid4().hex[:6]}"
    resp1 = agent.process_message(
        conversation_id=conv_id,
        user_message="Train a competitive Tic-Tac-Toe agent",
        selected_options={"smart_defaults": "true"},
    )
    assert resp1.status == "arena_preview"

    # Send multi-target request
    resp2 = agent.process_message(
        conversation_id=conv_id,
        user_message=compound_prompt,
    )
    assert resp2.action_plan is not None
    assert ExecutionTarget.REWARD_SHAPING in resp2.action_plan.targets or ExecutionTarget.ARENA_VISUAL_UI in resp2.action_plan.targets
    assert resp2.arena_html is not None
    assert "scoreboard" in resp2.arena_html.lower() or "gold" in resp2.arena_html.lower()


def test_unsupported_capability_honest_refusal(tmp_path):
    """Test 4: Honest refusal with zero code mutations for unsupported capabilities."""
    agent = StudioOmniAgent()
    impossible_prompt = "Stream live 3D VR audio to a Discord voice channel"

    plan = agent.plan_chat_action(impossible_prompt)
    assert plan.supported is False
    assert ExecutionTarget.UNSUPPORTED_CAPABILITY in plan.targets

    conv_id = f"test_honest_refusal_{uuid.uuid4().hex[:6]}"
    # First create a valid session
    resp1 = agent.process_message(
        conversation_id=conv_id,
        user_message="Train a competitive Tic-Tac-Toe agent",
        selected_options={"smart_defaults": "true"},
    )
    env_file = agent.conv_store.workspaces_dir / conv_id / "env_module.py"
    original_code = env_file.read_text(encoding="utf-8") if env_file.exists() else ""

    # Send impossible request
    resp2 = agent.process_message(
        conversation_id=conv_id,
        user_message=impossible_prompt,
    )

    assert resp2.status == "unsupported"
    assert "I do not know how to do that yet" in resp2.assistant_message
    assert plan.unsupported_reason in resp2.assistant_message or "Discord" in resp2.assistant_message

    # Verify zero mutations
    if env_file.exists():
        current_code = env_file.read_text(encoding="utf-8")
        assert current_code == original_code

    # Test another impossible capability: Bloomberg FIX protocol
    bloomberg_prompt = "Connect this agent to a live Bloomberg terminal via FIX protocol"
    plan2 = agent.plan_chat_action(bloomberg_prompt)
    assert plan2.supported is False
    assert ExecutionTarget.UNSUPPORTED_CAPABILITY in plan2.targets


def test_no_keyword_routing_lists_in_omni_agent():
    """Test 5: Codebase audit asserting classify_intents delegates to _plan_chat_action without keyword lists."""
    omni_agent_file = Path(__file__).resolve().parent.parent / "rl_ide" / "architect" / "omni_agent.py"
    source_code = omni_agent_file.read_text(encoding="utf-8")

    # Parse AST of omni_agent.py
    tree = ast.parse(source_code)

    # Locate StudioOmniAgent class
    studio_agent_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "StudioOmniAgent":
            studio_agent_class = node
            break

    assert studio_agent_class is not None, "StudioOmniAgent class not found in omni_agent.py"

    # Locate classify_intents method
    classify_intents_node = None
    for item in studio_agent_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "classify_intents":
            classify_intents_node = item
            break

    assert classify_intents_node is not None, "classify_intents method not found in StudioOmniAgent"

    classify_source = ast.get_source_segment(source_code, classify_intents_node)

    # Audit that classify_intents calls _plan_chat_action
    assert "_plan_chat_action" in classify_source

    # Audit that static domain keyword lists are NOT in classify_intents
    banned_variable_names = ["arena_kws", "train_kws", "agent_kws", "env_kws"]
    for banned in banned_variable_names:
        assert banned not in classify_source, f"Static keyword list '{banned}' found in classify_intents!"

    # Audit that hardcoded blackjack intent checking is NOT in classify_intents or process_message
    process_message_node = None
    for item in studio_agent_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "process_message":
            process_message_node = item
            break

    assert process_message_node is not None
    process_source = ast.get_source_segment(source_code, process_message_node)
    assert "is_blackjack_intent" not in process_source, "Hardcoded is_blackjack_intent found in process_message!"
