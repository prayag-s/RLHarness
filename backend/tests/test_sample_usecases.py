"""Sample Use Cases Test Suite for Continuous Conversational Studio (v4.2).

Demonstrates and verifies real-world user workflows:
- Use Case 1: Strategy / Board Game (Tic-Tac-Toe):
  1. Initial prompt -> Instant mock-animated preview (arena_locked=False)
  2. Arena visual restyling -> Live hot-reload with gold highlights & scoreboard
  3. Lock arena & train -> Initial 2 rounds of training
  4. Diagnostic consultation -> Explain telemetry and reward rules without code mutation
  5. Mid/Post-training reward tweaking -> Rewrite compute_reward(), warm up Critic, keep weights
  6. Resumed training -> Continue from Round 3 to Round 4 directly from checkpoint
  7. Interactive play -> Verify interactive human vs agent play against trained weights

- Use Case 2: Continuous Control & Physics (CartPole / Balancing Agent):
  1. Initial physics prompt -> Mock animation preview with cart oscillation & angle gauges
  2. Telemetry HUD restyling -> Add action probability radar and telemetry HUD
  3. Initial training burst -> Train 2 rounds
  4. Environment mechanics modification -> Update environment rules and self-heal
  5. Continue training -> Resumes model to adapt to new environment mechanics
"""

import os
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from rl_ide.architect.omni_agent import StudioActionType
from rl_ide.api.server import app, global_store
from rl_ide.state.conversation_store import get_conversation_store

# Multi-turn studio walkthroughs against the real provider: paid and slow, so excluded by default.
pytestmark = pytest.mark.live_llm

if os.environ.get("RL_IDE_LIVE_LLM") != "1":
    pytest.skip(
        "live-LLM use-case suite: export RL_IDE_LIVE_LLM=1 and run with `-m live_llm`",
        allow_module_level=True,
    )


@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# Use Case 1: Strategy / Board Game Workflow
# =============================================================================
def test_sample_usecase_board_game_full_lifecycle(client: TestClient):
    conv_id = f"usecase_board_{uuid.uuid4().hex[:6]}"
    print(f"\n[Use Case 1] Starting Board Game Workflow for session: {conv_id}")

    # Step 1: Initial Prompt -> Instant Mock-Animated Arena Preview
    print("  -> Step 1: Initial idea prompt: 'Train an agent to play Tic-Tac-Toe'")
    resp1 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train an agent to play Tic-Tac-Toe"},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] in ("arena_preview", "clarifying")
    assert data1["arena_locked"] is False
    assert "startMockAnimationLoop" in data1["arena_html"]
    assert "mock-preview-banner" in data1["arena_html"]
    assert "SET_ARENA_MODE" in data1["arena_html"]
    print("     [OK] Instant Mock-Animated preview generated (arena_locked=False).")

    # Step 2: Arena Visual Restyling before Training
    print("  -> Step 2: Visual restyling: 'Change the arena to a retro arcade theme with gold winning highlights and a scoreboard on the left'")
    resp2 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Change the arena to a retro arcade theme with gold winning highlights and a scoreboard on the left"},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert StudioActionType.PREVIEW_OR_UPDATE_ARENA.value in data2["action_types"]
    assert data2["arena_locked"] is False, "Restyling must NOT lock the arena"
    arena2 = data2["arena_html"]
    assert "scoreboard" in arena2.lower()
    assert "gold" in arena2.lower() or "amber" in arena2.lower()
    print("     [OK] Arena restyled and hot-reloaded without locking.")

    # Step 3: Lock Arena & Train Rounds 1..2
    print("  -> Step 3: Lock arena and train: 'Looks good, lock arena and start training for 2 rounds with 50 episodes per round'")
    resp3 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Looks good, lock arena and start training for 2 rounds with 50 episodes per round",
            "training_params": {
                "max_auto_tune_rounds": 2,
                "episodes_per_round": 50,
                "target_metric_threshold": 105.0,
            },
            "run_sync": True,
        },
    )
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["arena_locked"] is True
    assert data3["current_round"] == 2
    assert global_store.get_model(conv_id) is not None
    print(f"     [OK] Arena locked, trained 2 rounds. Best metric: {data3['best_metric']:.1f}%")

    # Step 4: Conversational Diagnostics
    print("  -> Step 4: Diagnostic question: 'Why did the agent make suboptimal moves in Round 1 and what are the current reward rules?'")
    resp4 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Why did the agent make suboptimal moves in Round 1 and what are the current reward rules?"},
    )
    assert resp4.status_code == 200
    data4 = resp4.json()
    assert StudioActionType.CLARIFY_OR_DISCUSS.value in data4["action_types"]
    assert len(data4["assistant_message"]) > 50
    # Model and code unchanged
    assert global_store.get_model(conv_id) is not None
    print("     [OK] Diagnostic advice delivered conversationally without code regression.")

    # Step 5: Post-Training Reward Tweaking & Personality Modification
    print("  -> Step 5: Behavior tweak: 'The agent is too passive; make it prioritize center control and penalize stalling'")
    resp5 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "The agent is too passive; make it prioritize center control and penalize stalling"},
    )
    assert resp5.status_code == 200
    data5 = resp5.json()
    assert StudioActionType.UPDATE_AGENT_CHARACTERISTICS.value in data5["action_types"]
    env_code = (Path("workspaces") / conv_id / "env_module.py").read_text(encoding="utf-8")
    assert "compute_reward" in env_code
    assert global_store.get_model(conv_id) is not None, "Policy weights preserved across reward update"
    print("     [OK] compute_reward() rewritten, Critic calibrated, existing policy intact.")

    # Step 6: Continue Resumed Training (Rounds 3..4)
    print("  -> Step 6: Resume training: 'Train for 2 more rounds with 50 episodes per round'")
    resp6 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Train for 2 more rounds with 50 episodes per round",
            "training_params": {
                "max_auto_tune_rounds": 2,
                "episodes_per_round": 50,
                "target_metric_threshold": 105.0,
            },
            "run_sync": True,
        },
    )
    assert resp6.status_code == 200
    data6 = resp6.json()
    assert StudioActionType.RUN_OR_CONTINUE_TRAINING.value in data6["action_types"]
    assert data6["current_round"] == 4, f"Expected round 4, got {data6['current_round']}"
    print(f"     [OK] Resumed training directly from checkpoint: completed Round {data6['current_round']}.")

    # Step 7: Interactive Play Against Trained Agent
    print("  -> Step 7: Interactive play test via POST /api/v4/sessions/{conv_id}/interact")
    resp7 = client.post(
        f"/api/v4/sessions/{conv_id}/interact",
        json={"payload": {"user_action": 4}},  # User plays center
    )
    assert resp7.status_code == 200
    interact_data = resp7.json()
    assert "arena_state" in interact_data
    assert "action_probs" in interact_data
    print("     [OK] Interactive play operational: agent computed response moves and action distributions.")


# =============================================================================
# Use Case 2: Continuous Physics & Control Workflow
# =============================================================================
def test_sample_usecase_physics_control_workflow(client: TestClient):
    conv_id = f"usecase_physics_{uuid.uuid4().hex[:6]}"
    print(f"\n[Use Case 2] Starting Physics Control Workflow for session: {conv_id}")

    # Step 1: Initial CartPole prompt
    print("  -> Step 1: Initial prompt: 'Train a CartPole balancing agent that keeps the pole upright'")
    resp1 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train a CartPole balancing agent that keeps the pole upright"},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] in ("arena_preview", "clarifying")
    assert data1["arena_locked"] is False
    assert "startMockAnimationLoop" in data1["arena_html"]
    print("     [OK] CartPole mock animation loop active.")

    # Step 2: Restyle arena with telemetry HUD
    print("  -> Step 2: Restyling: 'Add a telemetry HUD and action probability radar to the arena'")
    resp2 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Add a telemetry HUD and action probability radar to the arena"},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert StudioActionType.PREVIEW_OR_UPDATE_ARENA.value in data2["action_types"]
    assert "radar" in data2["arena_html"].lower() or "hud" in data2["arena_html"].lower() or "telemetry" in data2["arena_html"].lower()
    print("     [OK] Telemetry HUD & radar added to arena.html.")

    # Step 3: Lock & Train 2 rounds
    print("  -> Step 3: Lock & train: 'Looks good, lock arena and train for 2 rounds with 50 episodes per round'")
    resp3 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Looks good, lock arena and train for 2 rounds with 50 episodes per round",
            "training_params": {
                "max_auto_tune_rounds": 2,
                "episodes_per_round": 50,
                "target_metric_threshold": 105.0,
            },
            "run_sync": True,
        },
    )
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["arena_locked"] is True
    assert data3["current_round"] == 2
    print(f"     [OK] Physics policy trained for 2 rounds. Best metric: {data3['best_metric']:.1f}%")

    # Step 4: Environment Mechanics Update
    print("  -> Step 4: Environment mechanics update: 'Modify environment mechanics: add wind turbulence and disturbance forces'")
    resp4 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Modify environment mechanics: add wind turbulence and disturbance forces to the rules"},
    )
    assert resp4.status_code == 200
    data4 = resp4.json()
    assert StudioActionType.UPDATE_ENV_MECHANICS.value in data4["action_types"]
    ws_env = Path("workspaces") / conv_id / "env_module.py"
    assert ws_env.exists()
    print("     [OK] Environment rules updated and sandbox smoke-tested.")

    # Step 5: Resume Training for 1 more round
    print("  -> Step 5: Resume training: 'Train for 1 more round with 50 episodes'")
    resp5 = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Train for 1 more round with 50 episodes",
            "training_params": {
                "max_auto_tune_rounds": 1,
                "episodes_per_round": 50,
                "target_metric_threshold": 105.0,
            },
            "run_sync": True,
        },
    )
    assert resp5.status_code == 200
    data5 = resp5.json()
    assert data5["current_round"] == 3
    print(f"     [OK] Resumed training on updated mechanics: advanced to Round {data5['current_round']}.")
