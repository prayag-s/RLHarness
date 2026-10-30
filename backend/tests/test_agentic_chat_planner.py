"""Automated Verification Suite for Code-Aware Agentic Chat Planner & Multi-Layer Code Executor."""
import uuid
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from rl_ide.architect.models import ExecutionTarget, ChatActionPlan
from rl_ide.architect.omni_agent import StudioActionType
from rl_ide.synthesizer.env_builder import load_synthesized_env_class, instantiate_env
from rl_ide.api.server import app, global_store

# Trains, exports or spawns sandbox workers: part of the full suite, not the push gate.
pytestmark = pytest.mark.slow
@pytest.fixture
def client():
    return TestClient(app)
# -----------------------------------------------------------------------------
# Test 1: Inference Strategy Surgical Patching for First Move Randomization
# -----------------------------------------------------------------------------
def test_first_move_randomization_inference_patch(client: TestClient):
    conv_id = f"test_planner_first_move_{uuid.uuid4().hex[:6]}"
    # Step 1: Initialize conversation with Tic-Tac-Toe
    init_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train an unbeatable 3x3 Tic-Tac-Toe agent"},
    )
    assert init_resp.status_code == 200
    # Step 2: Lock Arena and generate env_module.py
    lock_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Looks good, lock arena and start training for 1 round with 2 episodes",
            "training_params": {"max_auto_tune_rounds": 1, "episodes_per_round": 2},
            "run_sync": True,
        },
    )
    assert lock_resp.status_code == 200
    ws_env = Path("workspaces") / conv_id / "env_module.py"
    assert ws_env.exists(), "env_module.py must exist after initialization"
    original_code = ws_env.read_text(encoding="utf-8")
    # Step 3: Send exact prompt requesting opening move randomization
    prompt = "if agent is first, then randomize its first move among good options"
    patch_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    # Assert plan properties
    action_plan = data.get("action_plan")
    assert action_plan is not None, "Response must include action_plan"
    assert action_plan["supported"] is True, "Plan must be marked supported"
    assert ExecutionTarget.INFERENCE_STRATEGY.value in action_plan["targets"]
    assert action_plan["scope_classification"] in ("arena_only", "both")
    # Step 4: Verify interactive_step() in env_module.py was updated
    updated_code = ws_env.read_text(encoding="utf-8")
    assert updated_code != original_code, "env_module.py code must be modified"
    assert "_good" in updated_code and "np.random.choice" in updated_code, (
        "interactive_step() must be patched with legal-action candidate sampling"
    )
    assert "range(9)" not in updated_code.split("def interactive_step")[1].split("def ")[0], (
        "The inference patch must not assume a 9-action board"
    )
    # Step 5: Test Live Behavior - Run 30 consecutive simulated opening turns
    env_cls = load_synthesized_env_class(ws_env)
    env = instantiate_env(env_cls)
    def peaked_mock_policy(obs):
        # Peaked policy that strongly favors center cell 4 (pure argmax would deterministically pick 4)
        probs = np.zeros(9, dtype=np.float32)
        probs[4] = 1.0
        return 4, probs
    n_actions = env.action_space.n
    selected_opening_cells = set()
    for _ in range(30):
        env.interactive_step(
            {"reset": True, "human_first": False, "human_symbol": "O"},
            peaked_mock_policy,
        )
        agent_move = int(getattr(env, "last_move"))
        assert 0 <= agent_move < n_actions, (
            f"Agent opening move {agent_move} is not a legal action (n_actions={n_actions})"
        )
        selected_opening_cells.add(agent_move)
    assert len(selected_opening_cells) >= 2, (
        f"Expected at least 2 distinct legal opening moves, but observed {selected_opening_cells}. "
        "Argmax determinism was not replaced by stochastic selection!"
    )
# -----------------------------------------------------------------------------
# Test 2: Multi-Layer Execution Chaining (REWARD_SHAPING + INFERENCE_STRATEGY)
# -----------------------------------------------------------------------------
def test_multi_layer_arena_and_training_patch(client: TestClient):
    conv_id = f"test_planner_multi_layer_{uuid.uuid4().hex[:6]}"
    # Step 1: Initialize conversation
    init_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train a competitive Tic-Tac-Toe agent"},
    )
    assert init_resp.status_code == 200
    # Step 2: Lock Arena and generate env_module.py
    lock_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Lock arena and start training for 1 round with 2 episodes",
            "training_params": {"max_auto_tune_rounds": 1, "episodes_per_round": 2},
            "run_sync": True,
        },
    )
    assert lock_resp.status_code == 200
    ws_env = Path("workspaces") / conv_id / "env_module.py"
    assert ws_env.exists()
    # Step 3: Send compound prompt touching both reward shaping and arena inference
    compound_prompt = "Reward corner openings during training and also randomize the agent's first move in the arena"
    patch_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": compound_prompt},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    # Assert plan contains both targets
    action_plan = data.get("action_plan")
    assert action_plan is not None
    assert action_plan["supported"] is True
    assert ExecutionTarget.REWARD_SHAPING.value in action_plan["targets"]
    assert ExecutionTarget.INFERENCE_STRATEGY.value in action_plan["targets"]
    assert action_plan["scope_classification"] == "both"
    # Step 4: Verify both layers are modified in env_module.py
    updated_code = ws_env.read_text(encoding="utf-8")
    assert "corner_bonus" in updated_code, (
        "compute_reward() must be updated with the requested corner reward term"
    )
    assert "_good" in updated_code and "np.random.choice" in updated_code, (
        "interactive_step() must be patched with inference randomization logic"
    )
# -----------------------------------------------------------------------------
# Test 3: Honesty Guardrail Refusal for Unsupported Capabilities
# -----------------------------------------------------------------------------
def test_unsupported_capability_honest_refusal(client: TestClient):
    conv_id = f"test_planner_unsupported_{uuid.uuid4().hex[:6]}"
    # Step 1: Initialize conversation
    init_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train an agent for Tic-Tac-Toe"},
    )
    assert init_resp.status_code == 200
    # Step 2: Lock Arena and generate env_module.py
    lock_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Lock arena and start training for 1 round with 2 episodes",
            "training_params": {"max_auto_tune_rounds": 1, "episodes_per_round": 2},
            "run_sync": True,
        },
    )
    assert lock_resp.status_code == 200
    ws_env = Path("workspaces") / conv_id / "env_module.py"
    assert ws_env.exists()
    before_bytes = ws_env.read_bytes()
    # Step 3: Send impossible/unsupported capability request
    unsupported_prompt = "Connect this agent to a live Discord voice channel and stream 3D WebGL animations"
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": unsupported_prompt},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Assert honest refusal contract
    assert data["status"] == "unsupported"
    action_plan = data.get("action_plan")
    assert action_plan is not None
    assert action_plan["supported"] is False
    assert action_plan["unsupported_reason"] is not None
    assert len(action_plan["unsupported_reason"]) > 0
    assert ExecutionTarget.UNSUPPORTED_CAPABILITY.value in action_plan["targets"]
    assert action_plan["scope_classification"] == "unsupported"
    # Assert user-facing message explicitly refuses honestly
    msg = data.get("assistant_message") or data.get("message") or ""
    assert "I do not know how to do that yet" in msg or "not supported" in msg.lower(), (
        f"Message '{msg}' must state inability to fulfill unsupported capability"
    )
    # Assert 100% byte-for-byte identical code (zero mutations applied)
    after_bytes = ws_env.read_bytes()
    assert before_bytes == after_bytes, "env_module.py must remain 100% byte-for-byte unchanged!"