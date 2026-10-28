"""Phase 4B Verification Test Suite.

Verifies:
1. JIT synthesis and smoke verification of pure simulation/game tasks (Tic-Tac-Toe / Grid),
   including reset, 25 steps, get_demonstrations, and interactive_step.
2. Self-Healing Verifier catching runtime bugs (ZeroDivisionError, shape mismatch) and
   repairing the environment code automatically.
3. JIT synthesis of data-driven environments from IngestedDataset (CSV/SQL), and
   generation of arena.html with ARENA_UPDATE / ARENA_INTERACT postMessage bridges.
4. Static AST security scan blocking dangerous imports and system calls.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from rl_ide.architect import AgentBlueprintSpec, IntakeArchitectAgent
from rl_ide.ingest import load_bytes, IngestedDataset
from rl_ide.synthesizer import (
    JITEnvironmentSynthesizer,
    SandboxSmokeTester,
    JITArenaSynthesizer,
    audit_ast_security,
    SecurityViolationError,
)


@pytest.fixture
def tictactoe_blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Tic-Tac-Toe Master Agent",
        domain_summary="Zero-sum 3x3 competitive board game",
        data_source_type="pure_simulation",
        observation_features=[
            "cell_0_0", "cell_0_1", "cell_0_2",
            "cell_1_0", "cell_1_1", "cell_1_2",
            "cell_2_0", "cell_2_1", "cell_2_2",
            "current_turn"
        ],
        action_labels=[
            "Place (0,0)", "Place (0,1)", "Place (0,2)",
            "Place (1,0)", "Place (1,1)", "Place (1,2)",
            "Place (2,0)", "Place (2,1)", "Place (2,2)"
        ],
        arena_visual_description="Modern 3x3 dark-mode grid with neon markers",
        interactive_mode_description="Interactive turn-based play against agent",
        success_metric_name="Non-Loss Rate (%)",
        target_metric_threshold=85.0,
        max_auto_tune_rounds=4,
        initial_reward_components=[
            {"name": "win_bonus", "weight": 10.0, "description": "Reward for winning"},
            {"name": "draw_bonus", "weight": 3.0, "description": "Reward for draw"},
            {"name": "loss_penalty", "weight": -10.0, "description": "Penalty for loss"},
            {"name": "illegal_move_penalty", "weight": -5.0, "description": "Penalty for occupied cell"},
        ],
    )


# -----------------------------------------------------------------------------
# Test 1: JIT Synthesis & Smoke Verification of Game/Simulation
# -----------------------------------------------------------------------------

def test_jit_simulation_synthesis_and_smoke_test(tictactoe_blueprint: AgentBlueprintSpec, tmp_path: Path):
    """Synthesizes Gymnasium environment, passes smoke test, and executes interactive_step."""
    env_builder = JITEnvironmentSynthesizer()
    code_str, file_path, env_cls = env_builder.synthesize(
        tictactoe_blueprint,
        workspace_dir=tmp_path / "sim_session",
    )

    assert file_path.is_file()
    assert "class SynthesizedGymEnv" in code_str

    # Run Sandbox Smoke Test
    verifier = SandboxSmokeTester()
    res = verifier.run_smoke_test(env_cls)
    assert res["verified"] is True
    assert res["obs_dim"] == 10
    assert res["n_actions"] == 9
    assert "board" in res["sample_arena_state"]

    # Test Live Interactive Play
    env = env_cls()
    env.reset()

    def dummy_policy(obs):
        # Predicts first available move with uniform probabilities
        probs = [1.0 / 9] * 9
        return 0, probs

    # Human moves in center (cell 4)
    turn_res = env.interactive_step({"user_action": 4}, dummy_policy)
    assert isinstance(turn_res, dict)
    assert "arena_state" in turn_res
    assert turn_res["arena_state"]["board"][4] == -1  # Human mark is -1
    assert "agent_action_label" in turn_res
    assert len(turn_res["action_probs"]) == 9
    assert len(turn_res["status_text"]) > 0

    # Reset via interactive step
    reset_res = env.interactive_step({"reset": True}, dummy_policy)
    assert all(val == 0 for val in reset_res["arena_state"]["board"])


# -----------------------------------------------------------------------------
# Test 2: Deliberate Runtime Bug Injection & Self-Healing Verification
# -----------------------------------------------------------------------------

def test_self_healing_catches_and_repairs_division_by_zero(tictactoe_blueprint: AgentBlueprintSpec, tmp_path: Path):
    """Injects ZeroDivisionError into compute_reward and verifies self-healing repair."""
    env_builder = JITEnvironmentSynthesizer()
    clean_code = env_builder.generate_env_code(tictactoe_blueprint)

    # Deliberately inject ZeroDivisionError in compute_reward
    buggy_code = clean_code.replace("total = 0.0", "total = 1.0 / 0")
    assert "1.0 / 0" in buggy_code

    verifier = SandboxSmokeTester()
    success, healed_code, logs = verifier.verify_and_heal(
        buggy_code,
        tictactoe_blueprint,
        workspace_dir=tmp_path / "heal_div_zero",
    )

    assert success is True
    # Verify the failure was logged
    assert any("division by zero" in log_entry.lower() for log_entry in logs)
    # Verify the self-healing repair was invoked
    assert any("self-healing repair" in log_entry.lower() for log_entry in logs)
    # Verify the final attempt passed 100%
    assert any("verification passed 100%" in log_entry.lower() for log_entry in logs)
    assert "1.0 / 0" not in healed_code


def test_self_healing_catches_and_repairs_shape_mismatch(tictactoe_blueprint: AgentBlueprintSpec, tmp_path: Path):
    """Injects observation shape mismatch and verifies self-healing repair."""
    env_builder = JITEnvironmentSynthesizer()
    clean_code = env_builder.generate_env_code(tictactoe_blueprint)

    # Deliberately inject shape mismatch in _get_obs
    buggy_code = clean_code.replace("obs = np.zeros(10, dtype=np.float32)", "obs = np.zeros(17, dtype=np.float32)")
    assert "np.zeros(17" in buggy_code

    verifier = SandboxSmokeTester()
    success, healed_code, logs = verifier.verify_and_heal(
        buggy_code,
        tictactoe_blueprint,
        workspace_dir=tmp_path / "heal_shape",
    )

    assert success is True
    assert any("does not match observation_space.shape" in log_entry for log_entry in logs)
    assert any("self-healing repair" in log_entry.lower() for log_entry in logs)
    assert any("verification passed 100%" in log_entry.lower() for log_entry in logs)


# -----------------------------------------------------------------------------
# Test 3: Ingested Dataset Environment Synthesis & Arena postMessage Bridges
# -----------------------------------------------------------------------------

def test_data_driven_environment_and_arena_bridges(tmp_path: Path):
    """Synthesizes data-driven Gym environment and arena.html with postMessage protocol."""
    # 1. Ingest sample dataset
    csv_data = b"""tx_id,amount,risk_score,international,fraud_flag
tx_101,150.00,0.12,0,0
tx_102,8900.50,0.85,1,1
tx_103,45.20,0.05,0,0
tx_104,12400.00,0.92,1,1
"""
    dataset = load_bytes(filename="transactions.csv", content=csv_data)
    assert isinstance(dataset, IngestedDataset)

    # 2. Architect locks blueprint
    architect = IntakeArchitectAgent()
    intake_resp = architect.interview(
        user_prompt="Train an agent to detect fraud transactions from this table",
        dataset=dataset,
        force_smart_defaults=True,
    )
    assert intake_resp.status == "ready_to_build"
    blueprint = intake_resp.blueprint_spec
    assert blueprint is not None

    # 3. Synthesize Environment
    env_builder = JITEnvironmentSynthesizer()
    code_str, env_path, env_cls = env_builder.synthesize(
        blueprint,
        dataset=dataset,
        workspace_dir=tmp_path / "data_session",
    )
    assert env_path.is_file()

    # 4. Smoke Test Data Environment
    verifier = SandboxSmokeTester()
    smoke_res = verifier.run_smoke_test(
        env_cls,
        dataset_records=dataset.df.to_dict(orient="records"),
    )
    assert smoke_res["verified"] is True
    assert smoke_res["obs_dim"] == len(blueprint.observation_features)

    # 5. Synthesize Arena HTML
    arena_builder = JITArenaSynthesizer()
    html_str, arena_path = arena_builder.synthesize(
        blueprint,
        sample_arena_state=smoke_res["sample_arena_state"],
        workspace_dir=tmp_path / "data_session",
    )
    assert arena_path.is_file()

    # Verify standard postMessage protocol in generated arena.html
    assert 'type === "ARENA_UPDATE"' in html_str
    assert 'type: "ARENA_INTERACT"' in html_str
    assert "window.addEventListener" in html_str
    assert "window.parent.postMessage" in html_str

    # Verify telemetry elements exist in HTML
    assert 'id="telemetry-episode"' in html_str
    assert 'id="telemetry-step"' in html_str
    assert 'id="telemetry-reward"' in html_str
    assert 'id="reward-breakdown-bar"' in html_str
    assert 'id="action-probs-bar"' in html_str


# -----------------------------------------------------------------------------
# Test 4: Static AST Security Scan Blocks Prohibited Calls
# -----------------------------------------------------------------------------

def test_ast_security_blocks_malicious_code():
    """Verifies that static AST scan rejects dangerous imports and built-in calls."""
    # 1. Prohibited module imports
    with pytest.raises(SecurityViolationError, match="Prohibited import: 'os'"):
        audit_ast_security("import os\nprint(os.system('id'))")

    with pytest.raises(SecurityViolationError, match="Prohibited import: 'subprocess'"):
        audit_ast_security("import subprocess\nsubprocess.run(['ls'])")

    with pytest.raises(SecurityViolationError, match="Prohibited from-import: 'socket'"):
        audit_ast_security("from socket import socket\ns = socket()")

    # 2. Prohibited dangerous calls
    with pytest.raises(SecurityViolationError, match=r"Prohibited function call: 'eval\(\)'"):
        audit_ast_security("x = eval('2 + 2')")

    with pytest.raises(SecurityViolationError, match=r"Prohibited function call: 'exec\(\)'"):
        audit_ast_security("exec('import sys')")

    with pytest.raises(SecurityViolationError, match=r"Prohibited function call: '__import__\(\)'"):
        audit_ast_security("m = __import__('os')")

    with pytest.raises(SecurityViolationError, match=r"Prohibited function call: 'open\(\)'"):
        audit_ast_security("f = open('/etc/passwd')")

    # 3. Legitimate RL code should pass audit with 0 errors
    clean_rl_code = """
import math
import random
import numpy as np
import gymnasium as gym

class SafeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
"""
    audit_ast_security(clean_rl_code)  # Should not raise any exception
