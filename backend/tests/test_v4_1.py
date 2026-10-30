"""Verification Test Suite for RL-Native Agent IDE v4.1.

Verifies:
1. Unique Episode Keys Across Multi-Round Loops:
   Assert every episode ID in session.episodes is 100% unique across rounds with r1_eval_* and r2_eval_* prefixes.
2. Conversation Persistence & Resumability:
   Full persistence in SQLite + workspace folder, list conversations, reload past conversation,
   re-instantiate SynthesizedGymEnv + SB3 model into memory, and execute live interact step seamlessly.
3. Multi-Question Intake & Custom Training Params:
   Intake interview returns distinct question IDs (q_objective, q_arena_ui, q_interaction),
   accepts combined answers + explicit custom training params (rounds=8, episodes=150, target=92.0),
   and locks those exact numbers into AgentBlueprintSpec.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.architect.intake import IntakeArchitectAgent
from rl_ide.autopilot.loop_controller import AutonomousLoopController, SessionStateStore
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer
from rl_ide.api.server import app, global_store, global_controller
from rl_ide.state.conversation_store import get_conversation_store


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def tictactoe_blueprint_v41() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Tic-Tac-Toe v4.1 Master Agent",
        domain_summary="3x3 zero-sum grid game for v4.1 loop verification",
        data_source_type="pure_simulation",
        observation_features=[
            "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "turn"
        ],
        action_labels=[
            "Cell 0", "Cell 1", "Cell 2", "Cell 3", "Cell 4", "Cell 5", "Cell 6", "Cell 7", "Cell 8"
        ],
        arena_visual_description="Modern 3x3 dark-mode grid with neon glow and winning lines",
        interactive_mode_description="Click-to-place human vs agent interactive board with reset button",
        success_metric_name="Non-Loss Rate (%)",
        target_metric_threshold=105.0,  # Ensure at least 2 rounds are trained
        max_auto_tune_rounds=2,
        episodes_per_round=100,
        initial_reward_components=[
            {"name": "win_bonus", "weight": 10.0},
            {"name": "loss_penalty", "weight": -10.0},
            {"name": "draw_bonus", "weight": 3.0},
            {"name": "illegal_move_penalty", "weight": -5.0},
        ],
    )


# -----------------------------------------------------------------------------
# 1. Unique Episode Keys Across Multi-Round Loops
# -----------------------------------------------------------------------------

def test_unique_episode_keys_across_multi_round_loops(tictactoe_blueprint_v41: AgentBlueprintSpec, tmp_path: Path):
    """Running a multi-round loop must produce 100% unique episode IDs with round prefixes."""
    store = SessionStateStore(base_dir=tmp_path / "workspaces")
    controller = AutonomousLoopController(store=store)

    session_id = "test_unique_keys_sess"
    ws_dir = tmp_path / "workspaces" / session_id
    env_builder = JITEnvironmentSynthesizer()
    _, _, env_cls = env_builder.synthesize(
        blueprint=tictactoe_blueprint_v41,
        workspace_dir=ws_dir,
    )

    events_captured: list[dict] = []

    def event_collector(ev):
        events_captured.append(ev)

    # Run 2-round loop
    session = controller.run_loop(
        session_id=session_id,
        blueprint=tictactoe_blueprint_v41,
        env_cls=env_cls,
        on_event_callback=event_collector,
        max_rounds=2,
        episodes_per_burst=10,
    )

    assert session.current_round >= 2
    assert len(session.rounds_history) >= 2
    assert len(session.episodes) > 0, "session.episodes must accumulate evaluation episodes"

    # Verify every episode ID in session.episodes is 100% unique
    episode_ids = [ep["episode_id"] for ep in session.episodes]
    assert len(episode_ids) == len(set(episode_ids)), (
        f"Duplicate episode IDs detected! Total: {len(episode_ids)}, Unique: {len(set(episode_ids))}"
    )

    # Verify r1_eval_* and r2_eval_* prefixes exist
    r1_evals = [eid for eid in episode_ids if str(eid).startswith("r1_eval_")]
    r2_evals = [eid for eid in episode_ids if str(eid).startswith("r2_eval_")]
    assert len(r1_evals) > 0, "Must have r1_eval_* episodes"
    assert len(r2_evals) > 0, "Must have r2_eval_* episodes"

    # Verify every episode dictionary contains "round"
    for ep in session.episodes:
        assert "round" in ep
        assert ep["round"] in (1, 2)


# -----------------------------------------------------------------------------
# 2. Conversation Persistence & Resumability
# -----------------------------------------------------------------------------

def test_conversation_persistence_and_resumability(test_client: TestClient):
    """Verifies SQLite persistence, listing, reloading state, and interactive play on reload."""
    conv_store = get_conversation_store()

    # Step 1: Create Conversation A via intake chat
    prompt_a = "Train an unbeatable Tic-Tac-Toe self-play agent"
    chat_resp_a = test_client.post(
        "/api/v4/architect/chat",
        json={"user_prompt": prompt_a},
    )
    assert chat_resp_a.status_code == 200
    data_a = chat_resp_a.json()
    id_a = data_a.get("conversation_id")
    assert id_a is not None

    # Step 2: Answer questions with custom choices and get locked blueprint
    answers_a = {
        "q_objective": "Unbeatable minimax play: Win whenever possible and force a draw otherwise",
        "q_arena_ui": "Modern 3x3 dark-mode grid with neon glow and winning lines",
        "q_interaction": "Click-to-place human vs agent interactive board with reset button",
    }
    chat_resp_locked = test_client.post(
        "/api/v4/architect/chat",
        json={
            "user_prompt": prompt_a,
            "conversation_id": id_a,
            "user_answers": answers_a,
            "training_params": {
                "max_auto_tune_rounds": 2,
                "episodes_per_round": 50,
                "target_metric_threshold": 80.0,
            },
        },
    )
    assert chat_resp_locked.status_code == 200
    bp_data = chat_resp_locked.json().get("blueprint_spec")
    assert bp_data is not None

    # Step 3: Run synchronous session start & loop for Conversation A
    start_resp = test_client.post(
        "/api/v4/sessions/start",
        json={
            "session_id": id_a,
            "blueprint": bp_data,
            "run_sync": True,
        },
    )
    assert start_resp.status_code == 200

    # Step 4: Create Conversation B
    create_resp_b = test_client.post(
        "/api/v4/conversations",
        json={"title": "Grid Navigator Agent B"},
    )
    assert create_resp_b.status_code == 200
    id_b = create_resp_b.json()["conversation_id"]

    # Step 5: List conversations via GET /api/v4/conversations
    list_resp = test_client.get("/api/v4/conversations")
    assert list_resp.status_code == 200
    conv_list = list_resp.json()
    conv_ids = [c["conversation_id"] for c in conv_list]
    assert id_a in conv_ids
    assert id_b in conv_ids

    # Find Conversation A metadata
    summary_a = next(c for c in conv_list if c["conversation_id"] == id_a)
    assert summary_a["best_metric"] >= 0.0
    assert summary_a["current_round"] >= 1

    # Clear in-memory session and model to simulate fresh server restart / agent switch
    global_store._sessions.clear()
    global_store._models.clear()
    global_store._envs.clear()

    # Step 6: Reload Conversation A via GET /api/v4/conversations/{id_a}
    reload_resp = test_client.get(f"/api/v4/conversations/{id_a}")
    assert reload_resp.status_code == 200
    reloaded_a = reload_resp.json()

    assert reloaded_a["conversation_id"] == id_a
    assert len(reloaded_a["chat_history"]) >= 2
    assert reloaded_a["arena_html"] is not None and len(reloaded_a["arena_html"]) > 100
    assert len(reloaded_a["episodes"]) > 0

    # Generated code must NOT be instantiated in the API process any more (P0-B.2); the
    # environment is loaded by the sandboxed worker on first interactive use instead.
    assert global_store.get_env(id_a) is None, "the API must not import generated code"

    # Step 7: Live interactive play against reloaded agent
    interact_payload = {"action": "human_move", "cell": 4}
    interact_resp = test_client.post(
        f"/api/v4/sessions/{id_a}/interact",
        json={"payload": interact_payload},
    )
    assert interact_resp.status_code == 200
    interact_data = interact_resp.json()
    assert "arena_state" in interact_data
    assert "action_probs" in interact_data

    # Step 8: Update training parameters on conversation
    patch_resp = test_client.patch(
        f"/api/v4/conversations/{id_a}/params",
        json={
            "max_auto_tune_rounds": 7,
            "episodes_per_round": 300,
            "target_metric_threshold": 95.0,
        },
    )
    assert patch_resp.status_code == 200
    updated_conv = patch_resp.json()
    assert updated_conv["training_params"]["max_auto_tune_rounds"] == 7
    assert updated_conv["training_params"]["episodes_per_round"] == 300
    assert updated_conv["training_params"]["target_metric_threshold"] == 95.0

    # Step 9: Delete Conversation B
    del_resp = test_client.delete(f"/api/v4/conversations/{id_b}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Ensure Conversation B is gone from list
    list_after = test_client.get("/api/v4/conversations").json()
    assert id_b not in [c["conversation_id"] for c in list_after]


# -----------------------------------------------------------------------------
# 3. Multi-Question Intake & Custom Training Params
# -----------------------------------------------------------------------------

def test_multi_question_intake_and_custom_training_params(test_client: TestClient):
    """Verifies multi-question IDs (q_objective, q_arena_ui, q_interaction) and explicit param locking."""
    # Step 1: Initial underspecified prompt returns distinct question IDs
    resp = test_client.post(
        "/api/v4/architect/chat",
        json={"user_prompt": "Synthesize a competitive Tic-Tac-Toe agent"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "needs_clarification"
    questions = data["questions"]
    assert len(questions) >= 3

    question_ids = {q.get("id") or q.get("question_id") for q in questions}
    assert "q_objective" in question_ids
    assert "q_arena_ui" in question_ids
    assert "q_interaction" in question_ids

    # Verify recommended training parameters included in IntakeResponse
    assert "recommended_training_params" in data
    rec = data["recommended_training_params"]
    assert rec["max_auto_tune_rounds"] == 5
    assert rec["episodes_per_round"] == 200
    assert rec["target_metric_threshold"] == 90.0

    # Step 2: Submit multi-question answers + custom training params
    answers = {
        "q_objective": "Defensive counter-puncher: Prioritize immediate threat blocking",
        "q_arena_ui": "Modern 3x3 dark-mode grid with neon glow and winning lines",
        "q_interaction": "Click-to-place human vs agent interactive board with reset button",
    }
    custom_params = {
        "max_auto_tune_rounds": 8,
        "episodes_per_round": 150,
        "target_metric_threshold": 92.0,
    }

    resp_build = test_client.post(
        "/api/v4/architect/chat",
        json={
            "user_prompt": "Synthesize a competitive Tic-Tac-Toe agent",
            "conversation_id": "conv_custom_params_test",
            "user_answers": answers,
            "training_params": custom_params,
        },
    )
    assert resp_build.status_code == 200
    build_data = resp_build.json()

    assert build_data["status"] == "ready_to_build"
    blueprint = build_data["blueprint_spec"]
    assert blueprint is not None

    # Step 3: Assert custom parameters are locked into AgentBlueprintSpec
    assert blueprint["max_auto_tune_rounds"] == 8
    assert blueprint["episodes_per_round"] == 150
    assert blueprint["target_metric_threshold"] == 92.0
    assert "Defensive" in blueprint["domain_summary"] or "counter-puncher" in blueprint["domain_summary"]
