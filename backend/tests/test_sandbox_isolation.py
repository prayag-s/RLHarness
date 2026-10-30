"""Regression tests for Phase P0-B.2: model-generated code runs behind an OS boundary.

The AST denylist in the verifier is defence in depth, not a boundary: a module can reach a file or a
socket through `numpy.fromfile`, `pandas.read_csv` or object-graph traversal. These tests bypass the
denylist entirely and prove the *sandbox* stops the code at the OS level.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

from rl_ide.autopilot.sandbox_runner import run_training_in_sandbox
from rl_ide.security import SandboxError, isolation_available, isolation_tool
from rl_ide.security.sandbox import sandbox_env
from rl_ide.autopilot.sandbox_runner import run_smoke_in_sandbox

pytestmark = [
    pytest.mark.slow,  # spawns bwrap workers and imports torch
    pytest.mark.skipif(
        not isolation_available(), reason="no isolation mechanism (bwrap/unshare) on this host"
    ),
]

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


def test_hosted_tenant_can_interact_without_in_process_execution(tmp_path):
    """A non-local owner is served by the sandbox, never by the API process."""
    import shutil
    import uuid

    from fastapi.testclient import TestClient

    from rl_ide.api.server import app, close_interact_session, global_store

    session_id = f"sandbox_hosted_{uuid.uuid4().hex[:8]}"
    workspace = global_store.base_dir / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "env_module.py").write_text(_SERVE_ENV, encoding="utf-8")

    client = TestClient(app)
    try:
        resp = client.post(
            f"/api/v4/sessions/{session_id}/interact",
            json={"payload": {"user_action": 0}},
            headers={"x-owner-id": "acme"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status_text"] == "call 1"
        assert global_store.get_env(session_id) is None
    finally:
        close_interact_session(session_id)
        shutil.rmtree(workspace, ignore_errors=True)


_SERVE_ENV = """
from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SynthesizedGymEnv(gym.Env):
    is_placeholder = False
    task_class = "classification"
    episode_horizon = 5
    action_labels = ["A", "B"]

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.calls = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {"arena_state": self.get_render_state()}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 1.0, True, False, {"reward_breakdown": {}}

    def get_render_state(self):
        return {"calls": self.calls}

    def interactive_step(self, payload, policy_predict_fn=None, mode="agent_vs_env"):
        self.calls += 1
        label = "A"
        probs = [1.0, 0.0]
        if policy_predict_fn is not None:
            action, probs = policy_predict_fn(np.zeros(2, dtype=np.float32))
            label = self.action_labels[int(action)]
        return {
            "arena_state": self.get_render_state(),
            "agent_action_label": label,
            "action_probs": [float(p) for p in probs],
            "status_text": f"call {self.calls}",
        }
"""


def test_serve_session_keeps_state_between_requests(tmp_path):
    from rl_ide.security.serve_client import SandboxedEnvSession

    env_module = tmp_path / "env_module.py"
    env_module.write_text(_SERVE_ENV, encoding="utf-8")

    session = SandboxedEnvSession()
    try:
        session.start(env_module=env_module, records=[{"a": 1}])
        first = session.interact({"user_action": 0})
        second = session.interact({"user_action": 1})
    finally:
        session.close()

    assert first["status_text"] == "call 1"
    assert second["status_text"] == "call 2", "the worker must keep environment state between requests"
    assert second["agent_action_label"] in {"A", "B"}


def test_interact_endpoint_runs_in_the_sandbox(tmp_path, monkeypatch):
    """The API must answer interactive play without instantiating the module in-process."""
    import uuid

    from fastapi.testclient import TestClient

    from rl_ide.api.server import app, close_interact_session, global_store

    session_id = f"sandbox_serve_{uuid.uuid4().hex[:8]}"
    workspace = global_store.base_dir / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "env_module.py").write_text(_SERVE_ENV, encoding="utf-8")

    client = TestClient(app)
    try:
        first = client.post(f"/api/v4/sessions/{session_id}/interact", json={"payload": {"user_action": 0}})
        second = client.post(f"/api/v4/sessions/{session_id}/interact", json={"payload": {"user_action": 1}})
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert second.json()["status_text"] == "call 2"
        assert global_store.get_env(session_id) is None, "no in-process env may be created"
    finally:
        close_interact_session(session_id)
        import shutil

        shutil.rmtree(workspace, ignore_errors=True)


def test_orphan_sandbox_case_dirs_are_purged():
    """A crashed API must not leave staged customer rows in runs/sandbox/serve_*."""
    import uuid

    from rl_ide.api import server

    fresh = Path("runs") / "sandbox" / f"serve_{uuid.uuid4().hex[:8]}"
    orphan = Path("runs") / "sandbox" / f"serve_{uuid.uuid4().hex[:8]}"
    for case in (fresh, orphan):
        case.mkdir(parents=True, exist_ok=True)
        (case / "records.json").write_text('[{"tenant": "acme", "claim": "secret"}]', encoding="utf-8")
    old = time.time() - 7200
    os.utime(orphan, (old, old))  # abandoned by a process that died two hours ago

    server._purge_orphan_serve_dirs()
    try:
        assert not orphan.exists(), "staged rows must not survive the process that created them"
        assert fresh.exists(), "a concurrently running instance's case dir must never be deleted"
    finally:
        shutil.rmtree(fresh, ignore_errors=True)
