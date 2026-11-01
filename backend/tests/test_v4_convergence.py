"""Phase 4 Convergence Verification Test.

Validates that an agent synthesized from an AgentBlueprintSpec rapidly and reliably converges:
- Reaches a high deterministic win/draw (non-loss) rate within <= 4 rounds
- Does so with **no oracle masking**: the policy is scored exactly as it is exported
- Incurs 0.0% illegal moves during deterministic evaluation
- Completes within < 30 seconds wall-clock time

The previous 90 % floor was only reachable because evaluation masked the policy down to
``get_critical_actions()`` — an oracle that blocks immediate wins and threats. Unmasked, the same
policy scores in the 78-90 % band across runs, so the assertion is a best-of-rounds floor plus a
margin over the random baseline measured in the same run.
"""

import time
from pathlib import Path
import pytest

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.autopilot.loop_controller import AutonomousLoopController, SessionStateStore
from tests.fixtures.tictactoe_fixture_env import TicTacToeFixtureEnv


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


@pytest.mark.slow  # trains a PPO policy (~30s) and scores in the 68-90% band run to run
def test_tictactoe_converges_without_oracle_masking(tictactoe_blueprint: AgentBlueprintSpec, tmp_path: Path):
    """Asserts Tic-Tac-Toe converges on its own merits, unmasked, within 4 rounds."""
    start_time = time.time()

    store = SessionStateStore(base_dir=tmp_path / "workspaces")
    controller = AutonomousLoopController(store=store)

    # 1. Deterministic, learnable contract-compliant environment (test fixture, not runtime code)
    session_id = "test_convergence_sess"
    env_cls = TicTacToeFixtureEnv

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

    # 4. Assert Convergence Criteria (honest, unmasked measurement)
    best_round_metric = max((snap.metric for snap in final_session.rounds_history), default=0.0)
    assert final_session.current_round <= 4, f"Failed: took {final_session.current_round} rounds (> 4)"
    assert best_round_metric >= 80.0, f"Failed: best round metric {best_round_metric}% < 80.0%"
    assert final_session.current_metric >= 70.0, (
        f"Failed: final metric {final_session.current_metric}% < 70.0%"
    )
    assert elapsed_time < 30.0, f"Failed: took {elapsed_time:.2f}s (>= 30s limit)"

    # 5. The reported number must come with a baseline it beats, and must be unmasked.
    scorecard = final_session.scorecard
    assert scorecard is not None, "A trained session must produce a scorecard"
    random_baseline = next(
        (b["value_pct"] for b in scorecard["baselines"] if b["name"] == "random_policy"), None
    )
    assert random_baseline is not None, "Control tasks must report a random-policy baseline"
    assert scorecard["policy"]["value_pct"] >= random_baseline + 40.0, (
        f"Policy {scorecard['policy']['value_pct']}% does not clearly beat "
        f"random {random_baseline}%"
    )
    assert scorecard["policy"]["illegal_actions_pct"] == 0.0

    # The metric is computed from terminal outcomes only (never "any successful step").
    eval_episodes = [ep for ep in final_session.episodes if ep.get("episode_id", "").startswith("r")]
    assert eval_episodes
    for episode in eval_episodes:
        assert episode["is_success"] == bool(episode["steps"][-1]["is_success"])
        assert episode["metric_value"] in (0.0, 100.0)

    # 6. Assert 0% illegal moves during deterministic evaluation
    round_eval_events = [e for e in events if e.get("event") == "round_evaluated"]
    assert len(round_eval_events) > 0
    final_eval_event = round_eval_events[-1]
    assert final_eval_event.get("illegal_moves_pct") == 0.0, (
        f"Expected 0.0% illegal moves in evaluation, got {final_eval_event.get('illegal_moves_pct')}%"
    )
