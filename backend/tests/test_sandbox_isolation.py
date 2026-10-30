"""Regression tests for Phase P0-B.2: model-generated code runs behind an OS boundary.

The AST denylist in the verifier is defence in depth, not a boundary: a module can reach a file or a
socket through `numpy.fromfile`, `pandas.read_csv` or object-graph traversal. These tests bypass the
denylist entirely and prove the *sandbox* stops the code at the OS level.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from rl_ide.autopilot.sandbox_runner import run_training_in_sandbox
from rl_ide.security import SandboxError, isolation_available, isolation_tool
from rl_ide.security.sandbox import sandbox_env
from rl_ide.autopilot.sandbox_runner import run_smoke_in_sandbox

pytestmark = pytest.mark.skipif(
    not isolation_available(), reason="no isolation mechanism (bwrap/unshare) on this host"
)

TENANT_DB = (Path(__file__).resolve().parents[1] / "runs" / "conversations.db").as_posix()

_ENV_TEMPLATE = """
from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces

{probe}

class SynthesizedGymEnv(gym.Env):
    metadata = {{"render_modes": ["json"]}}
    is_placeholder = False
    task_class = "classification"
    episode_horizon = 5
    action_labels = ["A", "B"]

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.step_count = 0

    def _obs(self):
        return np.array([float(self.step_count % 2), 1.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self._obs(), {{"arena_state": self.get_render_state()}}

    def step(self, action):
        self.step_count += 1
        return (
            self._obs(),
            1.0,
            self.step_count >= 5,
            False,
            {{"is_success": True, "is_correct": True, "reward_breakdown": {{}},
              "arena_state": self.get_render_state()}},
        )

    def get_render_state(self):
        return {{"step": self.step_count}}

    def get_demonstrations(self, num_samples=50, split="train"):
        return np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 0, 1], dtype=np.int64)

    def interactive_step(self, payload, policy_predict_fn=None, mode="agent_vs_env"):
        return {{
            "arena_state": self.get_render_state(),
            "agent_action_label": "A",
            "action_probs": [0.5, 0.5],
            "status_text": "ok",
        }}
"""


def _write_env(tmp_path: Path, probe: str) -> Path:
    path = tmp_path / "env_module.py"
    path.write_text(_ENV_TEMPLATE.format(probe=probe), encoding="utf-8")
    return path


def _run(tmp_path: Path, probe: str):
    return run_smoke_in_sandbox(_write_env(tmp_path, probe))


def test_worker_cannot_read_another_tenants_data(tmp_path):
    probe = (
        "def _probe():\n"
        f"    open({TENANT_DB!r}, 'rb').read(16)\n"
        "_probe()\n"
    )
    with pytest.raises(SandboxError) as err:
        _run(tmp_path, probe)
    assert "FileNotFoundError" in str(err.value), str(err.value)
    assert "Permission" not in str(err.value)


def test_worker_has_no_network(tmp_path):
    probe = (
        "import socket\n"
        "def _probe():\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=2)\n"
        "_probe()\n"
    )
    with pytest.raises(SandboxError) as err:
        _run(tmp_path, probe)
    assert "Network is unreachable" in str(err.value) or "NetworkError" in str(err.value), str(err.value)


def test_benign_environment_still_passes(tmp_path):
    outcome = _run(tmp_path, "pass")
    assert outcome["smoke"]["verified"] is True
    assert outcome["smoke"]["obs_dim"] == 2


def test_provider_keys_are_not_forwarded_into_the_sandbox(tmp_path):
    os.environ["DEEPSEEK_API_KEY"] = "sk-should-not-leak"
    try:
        env = sandbox_env(tmp_path)
    finally:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    assert "DEEPSEEK_API_KEY" not in env
    assert not any("KEY" in name or "TOKEN" in name for name in env)


def test_sandbox_reports_which_mechanism_is_used():
    assert Path(isolation_tool() or "").name == "bwrap"


def test_training_worker_returns_session_and_copies_artefacts(tmp_path):
    from rl_ide.architect.models import AgentBlueprintSpec

    blueprint = AgentBlueprintSpec(
        task_title="Sandbox Train",
        domain_summary="two actions",
        data_source_type="pure_simulation",
        observation_features=["a", "b"],
        action_labels=["A", "B"],
        arena_visual_description="desk",
        interactive_mode_description="choose",
        success_metric_name="Success",
        target_metric_threshold=90.0,
    )
    env_path = _write_env(tmp_path, "pass")
    records = pd.DataFrame({"x": [1, 2, 3, 4]}).to_dict(orient="records")

    events: list[dict] = []
    summary = run_training_in_sandbox(
        session_id="sandbox_train",
        blueprint=blueprint.model_dump(),
        env_module=env_path,
        records=records,
        base_dir=tmp_path / "workspaces",
        max_rounds=1,
        episodes_per_burst=2,
        additional_rounds=None,
        on_event=events.append,
        timeout=600,
    )

    assert summary["status"] in {"completed", "target_achieved"}
    assert summary["rounds_history"], "the worker must report its rounds"
    assert (tmp_path / "workspaces" / "sandbox_train" / "scorecard.json").exists()
    assert any(event.get("event") == "scorecard_ready" for event in events)
    assert not list((tmp_path / "workspaces").glob("temp_verify_*"))


def test_hosted_interact_is_refused_until_sandboxed():
    """A tenant must never trigger in-process execution of generated code."""
    from fastapi.testclient import TestClient
    from rl_ide.api.server import app

    client = TestClient(app)
    hosted = client.post(
        "/api/v4/sessions/does_not_exist/interact",
        json={"payload": {"user_action": 0}},
        headers={"x-owner-id": "acme"},
    )
    assert hosted.status_code == 501, hosted.text

    local = client.post(
        "/api/v4/sessions/does_not_exist/interact",
        json={"payload": {"user_action": 0}},
    )
    assert local.status_code == 404, "the local studio keeps its existing behaviour"
