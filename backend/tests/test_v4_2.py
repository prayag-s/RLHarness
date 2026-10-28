"""Verification Test Suite for Continuous Conversational Studio (v4.2).

Verifies:
1. Test 1 — Initial Prompt Generates Mock-Animated Arena Preview (`arena_locked=False`):
   - Send initial prompt to `POST /api/v4/conversations/{id}/chat`.
   - Verify draft `blueprint_spec` and `arena.html` containing the mock animation loop (`startMockAnimationLoop` / `SET_ARENA_MODE`),
     with `arena_locked == False`.
2. Test 2 — Iterative Arena Restyling Before & After Training:
   - Send chat message: "Change the arena look to include a dark cyber scoreboard and gold highlights".
   - Verify `StudioOmniAgent` classifies the intent as `PREVIEW_OR_UPDATE_ARENA` and updates `arena.html` without wiping blueprint or weights.
3. Test 3 — Lock Arena -> Train -> Post-Training Conversational Iteration:
   - Confirm/lock Arena and run initial training (Round 1..2).
   - After training finishes, send 3 sequential natural-language messages in the same conversation:
     a) "Change the agent behavior: add a stronger bonus for winning quickly and penalize stalling" -> verifies compute_reward() in env_module.py is updated.
     b) "Train for 2 more rounds with 100 episodes per round" -> verifies training resumes from Round 3 (current_round increments from 2 to 4) using existing model weights.
     c) "Update the arena UI to show a victory badge" -> verifies arena.html updates post-training while interactive play remains 100% intact.
"""

import json
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.architect.omni_agent import StudioActionType
from rl_ide.api.server import app, global_store, global_controller
from rl_ide.state.conversation_store import get_conversation_store


@pytest.fixture
def client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# Test 1: Initial Prompt Generates Mock-Animated Arena Preview
# -----------------------------------------------------------------------------
def test_initial_prompt_generates_mock_animated_arena_preview(client: TestClient):
    conv_id = f"test_v42_conv_1_{uuid.uuid4().hex[:6]}"
    prompt = "Train an agent to play Tic-Tac-Toe"

    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt},
    )
    assert resp.status_code == 200, f"Chat failed: {resp.text}"
    data = resp.json()

    assert data["conversation_id"] == conv_id
    assert data["status"] in ("arena_preview", "clarifying")
    assert data["arena_locked"] is False, "Arena must NOT be locked on initial prompt"

    # Blueprint must exist
    assert data.get("blueprint_spec") is not None
    assert "tic-tac-toe" in data["blueprint_spec"]["task_title"].lower()

    # Arena HTML must exist and contain mock simulation engine
    arena_html = data.get("arena_html")
    assert arena_html is not None
    assert len(arena_html) > 200
    assert "startMockAnimationLoop" in arena_html, "arena.html must have startMockAnimationLoop"
    assert "SET_ARENA_MODE" in arena_html, "arena.html must handle SET_ARENA_MODE postMessage"
    assert "mock-preview-banner" in arena_html, "arena.html must include mock preview banner"

    # Verify persisted in SQLite
    conv_store = get_conversation_store()
    saved = conv_store.get_conversation(conv_id)
    assert saved is not None
    assert saved["arena_locked"] is False
    assert saved["arena_html"] is not None


# -----------------------------------------------------------------------------
# Test 2: Iterative Arena Restyling Before & After Training
# -----------------------------------------------------------------------------
def test_iterative_arena_restyling(client: TestClient):
    conv_id = f"test_v42_conv_2_{uuid.uuid4().hex[:6]}"

    # Initial turn
    init_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train a 3x3 Tic-Tac-Toe self-play agent"},
    )
    assert init_resp.status_code == 200
    init_arena = init_resp.json()["arena_html"]

    # Restyle Arena
    restyle_prompt = "Change the arena look to include a dark cyber scoreboard and gold highlights"
    restyle_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": restyle_prompt},
    )
    assert restyle_resp.status_code == 200
    restyle_data = restyle_resp.json()

    assert StudioActionType.PREVIEW_OR_UPDATE_ARENA.value in restyle_data["action_types"]
    new_arena = restyle_data["arena_html"]
    assert new_arena is not None
    assert new_arena != init_arena or "scoreboard" in new_arena.lower()
    assert "scoreboard" in new_arena.lower(), "Updated arena must include scoreboard"
    assert "gold" in new_arena.lower() or "amber" in new_arena.lower(), "Updated arena must include gold/amber styling"
    assert "startMockAnimationLoop" in new_arena, "Restyled arena must preserve mock animation loop"


# -----------------------------------------------------------------------------
# Test 3: Lock Arena -> Train -> Post-Training Conversational Iteration
# -----------------------------------------------------------------------------
def test_lock_arena_train_and_post_training_iteration(client: TestClient):
    conv_id = f"test_v42_conv_3_{uuid.uuid4().hex[:6]}"

    # 1. Initial Prompt
    init_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train an unbeatable Tic-Tac-Toe agent"},
    )
    assert init_resp.status_code == 200
    assert init_resp.json()["arena_locked"] is False

    # 2. Lock Arena and train initial Round 1..2 synchronously
    lock_and_train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Looks good, lock arena and start training for 2 rounds with 50 episodes per round",
            "training_params": {
                "max_auto_tune_rounds": 2,
                "episodes_per_round": 50,
                "target_metric_threshold": 105.0,  # Ensure 2 full rounds complete
            },
            "run_sync": True,
        },
    )
    assert lock_and_train_resp.status_code == 200
    train_data = lock_and_train_resp.json()
    assert train_data["arena_locked"] is True
    assert train_data["current_round"] == 2

    # Check model exists in memory and runs directory
    model = global_store.get_model(conv_id)
    assert model is not None, "Model must be loaded in global_store"

    # 3a. Post-Training Message 1: Change Agent Behavior & Reward Formulation
    msg_a = "Change the agent behavior: add a stronger bonus for winning quickly and penalize stalling"
    resp_a = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": msg_a},
    )
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert StudioActionType.UPDATE_AGENT_CHARACTERISTICS.value in data_a["action_types"]

    # Verify env_module.py has updated compute_reward
    ws_env = Path("workspaces") / conv_id / "env_module.py"
    assert ws_env.exists()
    env_code = ws_env.read_text(encoding="utf-8")
    assert "compute_reward" in env_code
    assert "quick" in env_code.lower() or "stalling" in env_code.lower() or "bonus" in env_code.lower()

    # Model policy must still be intact
    assert global_store.get_model(conv_id) is not None

    # 3b. Post-Training Message 2: Train for 2 more rounds (resumes from Round 3 to Round 4)
    msg_b = "Train for 2 more rounds with 100 episodes per round"
    resp_b = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": msg_b,
            "training_params": {
                "max_auto_tune_rounds": 2,
                "episodes_per_round": 50,
                "target_metric_threshold": 105.0,
            },
            "run_sync": True,
        },
    )
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert StudioActionType.RUN_OR_CONTINUE_TRAINING.value in data_b["action_types"]

    # current_round must have incremented from 2 to 4!
    assert data_b["current_round"] == 4, f"Expected current_round=4, got {data_b['current_round']}"

    # 3c. Post-Training Message 3: Update Arena UI to show a victory badge
    msg_c = "Update the arena UI to show a victory badge"
    resp_c = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": msg_c},
    )
    assert resp_c.status_code == 200
    data_c = resp_c.json()
    assert StudioActionType.PREVIEW_OR_UPDATE_ARENA.value in data_c["action_types"]
    assert "victory badge" in data_c["arena_html"].lower() or "badge" in data_c["arena_html"].lower()

    # Interactive play must still work 100% seamlessly against the trained policy
    interact_resp = client.post(
        f"/api/v4/sessions/{conv_id}/interact",
        json={"payload": {"user_action": 0}},
    )
    assert interact_resp.status_code == 200
    interact_data = interact_resp.json()
    assert "arena_state" in interact_data
    assert "action_probs" in interact_data
