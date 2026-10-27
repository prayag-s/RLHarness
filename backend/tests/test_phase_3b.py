"""Verification Suite for Phase 3B: DynamicSimAdapter (Tic-Tac-Toe, 2D Robot Kinematics, Smart Inventory)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from rl_ide.contracts.adapter import DomainAdapter
from rl_ide.adapters.dynamic_sim_adapter import (
    DynamicSimAdapter,
    TicTacToeSimEngine,
    RobotKinematicsSimEngine,
    SmartInventorySimEngine,
)
from rl_ide.compiler.nl_to_config import NLToConfigCompiler
from rl_ide.compiler.critique_engine import CritiqueEngine
from rl_ide.trainer.runner import TrainingRunner
from rl_ide.export.exporter import PolicyExporter


def test_1_tictactoe_training_and_interactive_play(tmp_path: Path):
    """Test 1: Tic-Tac-Toe synthesis, compilation, warmstart + burst, and live interactive sparring."""
    adapter = DynamicSimAdapter(mode="tictactoe", opponent_type="tactical")
    assert isinstance(adapter, DomainAdapter)
    assert adapter.ui_mode == "board_grid"

    obs_spec = adapter.observations()
    action_spec = adapter.actions()
    schema = adapter.event_schema()

    assert obs_spec.vector_dim == 15
    assert len(action_spec.labels) == 9
    assert "obs.center_open" in schema
    assert "obs.is_legal_move" in schema
    assert "obs.took_winning_move" in schema
    assert "obs.blocked_opponent_win" in schema
    assert "state.winner" in schema

    # 1. Compile natural language prompt into TaskConfig
    compiler = NLToConfigCompiler()
    task_config = compiler.compile(
        user_prompt="Train an agent to play tic-tac-toe, take center when open, block opponent wins, and win.",
        adapter_name="DynamicSimAdapter",
        adapter=adapter,
        task_id="tictactoe_task",
    )
    assert task_config.adapter_name == "DynamicSimAdapter"
    terms = task_config.reward_dsl.reward.terms
    assert any("center" in t.name for t in terms)
    assert any("block" in t.name or "win" in t.name for t in terms)

    # 2. Warm-start behavioral cloning
    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=task_config.reward_dsl,
        task_id="tictactoe_task",
    )
    runner.warmstart_policy(epochs=10, n_samples=100)

    # 3. Micro-burst training
    burst_logs = runner.run_burst(n_episodes=8)
    assert len(burst_logs) >= 8

    # Check that agent never plays illegal moves (0% illegal moves)
    illegal_moves = 0
    non_losses = 0
    for ep in burst_logs:
        for step in ep.steps:
            if not step.events["obs"]["is_legal_move"]:
                illegal_moves += 1
        last_state = ep.steps[-1].events["state"]
        if last_state.get("winner") in ("agent", "draw"):
            non_losses += 1

    assert illegal_moves == 0, f"Expected 0 illegal moves, found {illegal_moves}"
    non_loss_rate = non_losses / float(len(burst_logs))
    assert non_loss_rate >= 0.85, f"Expected non-loss rate >= 85%, got {non_loss_rate:.2f}"

    # 4. Test render_frame
    step_res = adapter.step(4)
    frame = adapter.render_frame(
        episode_id=1,
        step_result=step_res,
        total_reward=15.0,
        breakdown={"take_center_bonus": 5.0, "win_game_bonus": 10.0},
    )
    assert frame["ui_mode"] == "board_grid"
    assert "board" in frame
    assert "winner" in frame

    # 5. Test Interactive Live Testing API (Play Against Your Agent)
    # Human plays top-left corner (cell 0)
    interactive_res = adapter.play_interactive_step(
        policy=runner.model.policy,
        board_state=[0] * 9,
        human_cell_idx=0,
    )
    assert interactive_res["ui_mode"] == "board_grid"
    board = interactive_res["board"]
    assert board[0] == -1  # Human move
    assert interactive_res["agent_cell"] != -1  # Agent took a move
    assert board[interactive_res["agent_cell"]] == +1  # Agent placed piece
    assert len(interactive_res["action_probabilities"]) == 9
    assert sum(interactive_res["action_probabilities"]) > 0.95


def test_2_robot_rover_navigation_critique_and_onnx_export(tmp_path: Path):
    """Test 2: 2D Robot Rover obstacle avoidance, critique diff proposal, and ONNX parity."""
    adapter = DynamicSimAdapter(mode="robot_rover")
    assert isinstance(adapter, DomainAdapter)
    assert adapter.ui_mode == "robot_canvas"

    obs_spec = adapter.observations()
    action_spec = adapter.actions()
    schema = adapter.event_schema()

    assert obs_spec.vector_dim == 18
    assert len(action_spec.labels) == 9
    assert "obs.dist_to_target" in schema
    assert "obs.min_obstacle_dist" in schema
    assert "obs.collided" in schema
    assert "obs.reached_target" in schema
    assert "obs.jerk_magnitude" in schema

    # 1. Compile prompt
    compiler = NLToConfigCompiler()
    task_config = compiler.compile(
        user_prompt="Train a robot rover to reach the target quickly while avoiding obstacles and minimizing jerk.",
        adapter_name="DynamicSimAdapter",
        adapter=adapter,
        task_id="robot_rover_task",
    )
    assert task_config.adapter_name == "DynamicSimAdapter"
    terms = task_config.reward_dsl.reward.terms
    assert any("reach" in t.name or "target" in t.name for t in terms)
    assert any("obstacle" in t.name or "collision" in t.name for t in terms)

    # 2. Train with TrainingRunner
    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=task_config.reward_dsl,
        task_id="robot_rover_task",
    )
    runner.warmstart_policy(epochs=5, n_samples=50)
    burst1 = runner.run_burst(n_episodes=5)
    assert len(burst1) >= 5

    # 3. Two-Stage Critique Diff proposal
    critique_engine = CritiqueEngine(schema, action_spec.labels)
    proposal = critique_engine.propose_diff(
        current_dsl=task_config.reward_dsl,
        target_episode=burst1[0],
        recent_episodes=burst1,
        critique_text="Stay further away from obstacles and avoid collisions",
        policy=runner.model.policy,
    )
    assert proposal.diff_yaml is not None
    assert len(proposal.term_diffs) >= 1
    term_diff_names = [td.term_name for td in proposal.term_diffs]
    assert any("obstacle" in name or "collision" in name for name in term_diff_names)

    # Apply diff
    runner.apply_reward_diff(proposal.new_dsl, critique_text="Stay further away from obstacles")
    burst2 = runner.run_burst(n_episodes=5)
    assert len(burst2) >= 5

    # 4. Export to ONNX and verify numerical parity (< 1e-5)
    onnx_file = tmp_path / "robot_rover.onnx"
    PolicyExporter.export_onnx(
        model=runner.model,
        output_path=onnx_file,
        obs_dim=obs_spec.vector_dim,
    )
    assert onnx_file.exists()

    max_diff = PolicyExporter.verify_parity(
        model=runner.model,
        onnx_path=onnx_file,
        obs_dim=obs_spec.vector_dim,
        n_samples=32,
    )
    assert max_diff < 1e-5

    # 5. Interactive live testing API (Click to Move Target / Place Obstacle)
    interactive_res = adapter.play_interactive_step(
        policy=runner.model.policy,
        robot_state={"x": 12.0, "y": 12.0, "vx": 0.0, "vy": 0.0, "battery": 100.0},
        target_xy=(40.0, 50.0),  # User moves target
        obstacles_list=[
            {"x": 25.0, "y": 25.0, "radius": 6.0},
            {"x": 55.0, "y": 65.0, "radius": 8.0},
        ],
        n_frames=4,
    )
    assert interactive_res["ui_mode"] == "robot_canvas"
    assert len(interactive_res["trajectory"]) == 5  # start + 4 frames
    assert "robot" in interactive_res
    assert interactive_res["target"]["x"] == 40.0
    assert interactive_res["target"]["y"] == 50.0
    assert "action_taken" in interactive_res


def test_3_inventory_sim_stepping_and_dashboard():
    """Test 3: SmartInventorySimEngine replenishment stepping, financial reward, and dashboard telemetry."""
    adapter = DynamicSimAdapter(mode="inventory_sim")
    assert isinstance(adapter, DomainAdapter)
    assert adapter.ui_mode == "sim_dashboard"

    obs_spec = adapter.observations()
    action_spec = adapter.actions()
    schema = adapter.event_schema()

    assert obs_spec.vector_dim == 6
    assert len(action_spec.labels) == 4
    assert "obs.stock_level" in schema
    assert "obs.profit_margin" in schema
    assert "obs.is_stockout" in schema

    compiler = NLToConfigCompiler()
    task_config = compiler.compile(
        user_prompt="Optimize warehouse inventory restocking, avoid stockouts and minimize holding costs.",
        adapter_name="DynamicSimAdapter",
        adapter=adapter,
        task_id="inventory_task",
    )
    assert task_config.adapter_name == "DynamicSimAdapter"

    # Reset and step 10 days
    obs, info = adapter.reset(seed=123)
    assert len(obs) == 6

    step_results = []
    for day in range(10):
        # Order tier 1 (small restock)
        res = adapter.step(action_idx=1)
        step_results.append(res)
        assert res.events["obs"]["stock_level"] >= 0.0
        assert isinstance(res.events["obs"]["profit_margin"], float)

    # Telemetry frame
    frame = adapter.render_frame(
        episode_id=1,
        step_result=step_results[-1],
        total_reward=45.0,
        breakdown={"profit_sales_reward": 45.0},
    )
    assert frame["ui_mode"] == "sim_dashboard"
    assert "profit_margin" in frame["obs"]
    assert "stock_level" in frame["obs"]
