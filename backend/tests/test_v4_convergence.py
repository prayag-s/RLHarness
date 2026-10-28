"""Phase 4 Convergence Verification Test.

Validates that an agent synthesized from an AgentBlueprintSpec rapidly and reliably converges:
- Achieves >= 90.0% deterministic win/draw (non-loss) rate within <= 4 rounds
- Incurs 0.0% illegal moves during deterministic evaluation
- Completes within < 15 seconds wall-clock time
"""

import time
from pathlib import Path
import pytest

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.autopilot.loop_controller import AutonomousLoopController, SessionStateStore
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer


@pytest.fixture
def tictactoe_blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Tic-Tac-Toe Self-Play Master",
        domain_summary="3x3 zero-sum grid game with heuristic self-play and tactical defense",
        data_source_type="pure_simulation",
        observation_features=[
            "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "turn"
        ],
        action_labels=[
            "place_0", "place_1", "place_2", "place_3", "place_4", "place_5", "place_6", "place_7", "place_8"
        ],
        arena_visual_description="A dark-themed 3x3 Tic-Tac-Toe grid",
        interactive_mode_description="Interactive play against trained policy",
        success_metric_name="Deterministic Win/Draw Rate",
        target_metric_threshold=90.0,
        max_auto_tune_rounds=4,
        initial_reward_components=[
            {"name": "win_bonus", "weight": 10.0},
            {"name": "loss_penalty", "weight": -10.0},
            {"name": "draw_bonus", "weight": 5.0},
            {"name": "illegal_move_penalty", "weight": -1.0},
        ],
    )


@pytest.fixture(autouse=True)
def warm_sb3_framework():
    """Pre-initializes PyTorch/SB3 JIT runtime to isolate execution time strictly to RL operations."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    try:
        dummy_env = gym.make("CartPole-v1")
        _ = PPO("MlpPolicy", dummy_env, n_steps=64, batch_size=32, device="cpu")
    except Exception:
        pass


def test_tictactoe_reaches_90_percent_non_loss_rate(tictactoe_blueprint: AgentBlueprintSpec, tmp_path: Path):
    """Asserts that Tic-Tac-Toe converges to >= 90% non-loss rate in <= 4 rounds (< 15s)."""
    start_time = time.time()

    store = SessionStateStore(base_dir=tmp_path / "workspaces")
    controller = AutonomousLoopController(store=store)

    # 1. Synthesize environment from blueprint
    env_builder = JITEnvironmentSynthesizer()
    session_id = "test_convergence_sess"
    ws_dir = tmp_path / "workspaces" / session_id

    code_str, env_path, env_cls = env_builder.synthesize(
        blueprint=tictactoe_blueprint,
        workspace_dir=ws_dir,
    )

    events: list[dict] = []
    def on_event(ev):
        events.append(ev)

    # 2. Run Autonomous Closed-Loop Training
    final_session = controller.run_loop(
        session_id=session_id,
        blueprint=tictactoe_blueprint,
        env_cls=env_cls,
        on_event_callback=on_event,
        max_rounds=4,
        episodes_per_burst=25,
    )

    elapsed_time = time.time() - start_time

    # 3. Print Convergence Metrics by Round
    print(f"\n--- Tic-Tac-Toe Convergence Test Results ({elapsed_time:.2f}s total) ---")
    for snap in final_session.rounds_history:
        print(
            f"Round {snap.round_num}: Metric = {snap.metric:.1f}% "
            f"(Target: {snap.target_threshold}%) | Diagnosis: {snap.diagnosis}"
        )

    # 4. Assert Convergence Criteria
    assert final_session.current_round <= 4, f"Failed: took {final_session.current_round} rounds (> 4)"
    assert final_session.current_metric >= 90.0, f"Failed: final metric {final_session.current_metric}% < 90.0%"
    assert elapsed_time < 15.0, f"Failed: took {elapsed_time:.2f}s (>= 15s limit)"

    # 5. Assert 0% illegal moves during deterministic evaluation
    round_eval_events = [e for e in events if e.get("event") == "round_evaluated"]
    assert len(round_eval_events) > 0
    final_eval_event = round_eval_events[-1]
    assert final_eval_event.get("illegal_moves_pct") == 0.0, (
        f"Expected 0.0% illegal moves in evaluation, got {final_eval_event.get('illegal_moves_pct')}%"
    )
