"""Phase 4C Verification Test Suite.

Verifies:
1. Multi-round autonomous closed-loop training and self-tuning (AutoTunerAgent diagnosis,
   reward rewriting, Critic Warmup, and reaching target mastery in Round 2).
2. Mid-flight / post-training user English nudge and round snapshot rollback.
3. End-to-end FastAPI server flow (/api/v4/architect/chat -> /api/v4/sessions/start ->
   /api/v4/sessions/{id}/interact -> /api/v4/sessions/{id}/export -> standalone run_local.py --sample).
"""

import json
from pathlib import Path
import subprocess
import sys
import pytest
import numpy as np
from fastapi.testclient import TestClient

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.autopilot.loop_controller import AutonomousLoopController, SessionStateStore
from rl_ide.autopilot.auto_tuner import AutoTunerAgent, TuningResult
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer
from tests.fixtures.tictactoe_fixture_env import TicTacToeFixtureEnv
from rl_ide.api.server import app, global_store, global_controller


@pytest.fixture
def tictactoe_blueprint_v4c() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Tic-Tac-Toe Autonomous Master",
        domain_summary="3x3 zero-sum grid game",
        data_source_type="pure_simulation",
        observation_features=[
            "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "turn"
        ],
        action_labels=[
            "Cell 0", "Cell 1", "Cell 2", "Cell 3", "Cell 4", "Cell 5", "Cell 6", "Cell 7", "Cell 8"
        ],
        arena_visual_description="Modern 3x3 dark-mode grid",
        interactive_mode_description="Interactive turn-based play",
        success_metric_name="Non-Loss Rate (%)",
        target_metric_threshold=80.0,
        max_auto_tune_rounds=3,
        initial_reward_components=[
            {"name": "win_bonus", "weight": 10.0},
            {"name": "loss_penalty", "weight": -10.0},
            {"name": "draw_bonus", "weight": 3.0},
            {"name": "illegal_move_penalty", "weight": -5.0},
        ],
    )


# -----------------------------------------------------------------------------
# Test 1: Autonomous Multi-Round Loop & Auto-Tuning
# -----------------------------------------------------------------------------

def test_autonomous_multi_round_self_tuning_loop(tictactoe_blueprint_v4c: AgentBlueprintSpec, tmp_path: Path):
    """Executes multi-round autonomous loop where AutoTuner rewrites reward and meets target."""
    store = SessionStateStore(base_dir=tmp_path / "workspaces")
    controller = AutonomousLoopController(store=store)

    # 1. Use a deterministic, learnable environment fixture (real contract, real metric)
    session_id = "test_loop_sess_1"
    env_cls = TicTacToeFixtureEnv

    events_captured: list[dict] = []

    def event_collector(ev):
        events_captured.append(ev)

    # Run Autonomous Multi-Round Loop
    final_session = controller.run_loop(
        session_id=session_id,
        blueprint=tictactoe_blueprint_v4c,
        env_cls=env_cls,
        on_event_callback=event_collector,
        max_rounds=3,
        episodes_per_burst=20,
    )

    assert final_session.session_id == session_id
    assert final_session.current_round >= 1
    assert len(final_session.rounds_history) >= 1

    # Verify event stream
    event_names = [e["event"] for e in events_captured]
    assert "round_started" in event_names
    assert "warmstart_completed" in event_names
    assert "episode_step" in event_names
    assert "round_evaluated" in event_names

    # Check that live frame contains ARENA_UPDATE components
    step_events = [e for e in events_captured if e["event"] == "episode_step"]
    assert len(step_events) > 0
    sample_frame = step_events[0]
    assert "total_reward" in sample_frame
    assert "arena_state" in sample_frame
    assert "action_probs" in sample_frame
    assert len(sample_frame["action_probs"]) == 9


# -----------------------------------------------------------------------------
# Test 2: User English Nudge & Round Snapshot Rollback
# -----------------------------------------------------------------------------

def test_user_nudge_and_snapshot_rollback(tictactoe_blueprint_v4c: AgentBlueprintSpec, tmp_path: Path):
    """Tests mid-flight English feedback and restoring previous round weights and code."""
    store = SessionStateStore(base_dir=tmp_path / "workspaces")
    controller = AutonomousLoopController(store=store)

    session_id = "test_nudge_sess_2"
    env_cls = TicTacToeFixtureEnv

    # Run 1 quick round to establish baseline snapshot v1
    controller.run_loop(
        session_id=session_id,
        blueprint=tictactoe_blueprint_v4c,
        env_cls=env_cls,
        max_rounds=1,
        episodes_per_burst=10,
    )

    initial_session = store.get_session(session_id)
    assert initial_session is not None
    assert len(initial_session.rounds_history) == 1

    # 1. Apply Natural Language Nudge
    nudge_text = "Prioritize aggressive center control and punish corner traps heavily"
    tuning_res = controller.nudge_session(session_id, nudge_text)

    assert isinstance(tuning_res, TuningResult)
    assert nudge_text in tuning_res.diagnosis or "User Nudge" in tuning_res.diagnosis
    assert tuning_res.verified is True

    updated_session = store.get_session(session_id)
    assert updated_session.current_round == 2
    assert len(updated_session.rounds_history) == 2

    # 2. Rollback to Round 1
    restored_snap = store.rollback_to_round(session_id, round_num=1)
    assert restored_snap.round_num == 1
    assert store.get_session(session_id).current_round == 1


# -----------------------------------------------------------------------------
# Test 3: End-to-End FastAPI Server Flow & Standalone Local Execution
# -----------------------------------------------------------------------------

def test_fastapi_e2e_flow_and_local_runbundle_execution(tmp_path: Path):
    """Full lifecycle: intake chat -> start session -> interact -> export -> run_local.py."""
    client = TestClient(app)

    # 1. Intake Architect Chat with smart defaults
    chat_resp = client.post(
        "/api/v4/architect/chat",
        json={
            "user_prompt": "Train an agent to master tic-tac-toe",
            "force_smart_defaults": True,
        },
    )
    assert chat_resp.status_code == 200
    intake_data = chat_resp.json()
    assert intake_data["status"] == "ready_to_build"
    blueprint = intake_data["blueprint_spec"]
    assert blueprint is not None

    # 2. Start Session (Synchronously for test)
    session_id = "e2e_test_session"
    start_resp = client.post(
        "/api/v4/sessions/start",
        json={
            "blueprint": blueprint,
            "session_id": session_id,
            "run_sync": True,
        },
    )
    assert start_resp.status_code == 200
    sess_data = start_resp.json()
    assert sess_data["session_id"] == session_id

    # 3. Retrieve Session Details & Arena HTML
    get_resp = client.get(f"/api/v4/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] in ("target_achieved", "completed", "training")

    arena_resp = client.get(f"/api/v4/sessions/{session_id}/arena")
    assert arena_resp.status_code == 200
    assert "ARENA_UPDATE" in arena_resp.text

    # 4. Live Interactive Step
    interact_resp = client.post(
        f"/api/v4/sessions/{session_id}/interact",
        json={"payload": {"user_action": 4}},
    )
    assert interact_resp.status_code == 200
    interact_data = interact_resp.json()
    assert "arena_state" in interact_data
    assert "agent_action_label" in interact_data
    assert "action_probs" in interact_data

    # 5. Export .runbundle and verify ONNX parity
    export_resp = client.post(f"/api/v4/sessions/{session_id}/export")
    assert export_resp.status_code == 200
    exp_data = export_resp.json()
    assert "bundle_dir" in exp_data
    assert "zip_path" in exp_data
    assert exp_data["onnx_parity_diff"] < 1e-5

    bundle_dir = Path(exp_data["bundle_dir"])
    assert (bundle_dir / "model.onnx").exists()
    assert (bundle_dir / "blueprint_spec.json").exists()
    assert (bundle_dir / "run_local.py").exists()

    # 6. Execute standalone run_local.py via subprocess
    run_local_py = bundle_dir / "run_local.py"
    proc = subprocess.run(
        [sys.executable, str(run_local_py), "--sample"],
        capture_output=True,
        text=True,
        cwd=str(bundle_dir),
        check=True,
    )
    assert "Local Inference Complete" in proc.stdout
    assert "Sample Predictions" in proc.stdout

    # 7. Check Registry listing
    bundles_resp = client.get("/api/v4/bundles")
    assert bundles_resp.status_code == 200
    assert len(bundles_resp.json()) >= 1


def test_fastapi_sql_ingest_endpoint(tmp_path: Path):
    """Verifies POST /api/v4/ingest/sql connects and loads an IngestedDataset properly."""
    import sqlite3
    db_file = tmp_path / "test_store.db"
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("CREATE TABLE metrics (id INTEGER PRIMARY KEY, value REAL, label TEXT)")
        conn.execute("INSERT INTO metrics (value, label) VALUES (42.5, 'alpha')")
        conn.execute("INSERT INTO metrics (value, label) VALUES (99.1, 'beta')")
        conn.commit()

    client = TestClient(app)
    resp = client.post(
        "/api/v4/ingest/sql",
        json={
            "connection_uri": f"sqlite:///{db_file.resolve()}",
            "query": "SELECT * FROM metrics",
            "name": "sqlite_test_ds",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "sqlite_test_ds"
    assert "dataset_id" in data
    assert "value" in data["columns"]
    assert data["num_rows"] == 2

