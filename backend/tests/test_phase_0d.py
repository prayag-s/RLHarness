"""Phase 0D Verification Suite.

Tests:
1. Unsupervised Drift Detector: Input PSI, Policy Entropy Z-score, Action Shift Z-score.
2. ONNX Policy Exporter: Numerical parity between PyTorch and ONNX Runtime (atol < 1e-5).
3. Standalone run_local.py execution via subprocess inside the exported bundle.
4. FastAPI REST endpoints and WebSocket telemetry stream via TestClient.
"""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from rl_ide.adapters.stream_adapter import CSVStreamAdapter
from rl_ide.compiler.models import TaskConfig
from rl_ide.reward.dsl import RewardDSL
from rl_ide.drift.monitor import DriftDetector, DriftBaseline
from rl_ide.trainer.runner import TrainingRunner
from rl_ide.export.exporter import PolicyExporter
from rl_ide.api.server import create_app

SAMPLE_DSL_YAML = """
version: "1.0"
horizon:
  gamma: 0.90
reward:
  terms:
    - name: accurate_match
      when: "truth.has_label == True and action.label == truth.label"
      value: 1.0
    - name: false_positive_low_vol
      when: "action.label == 'anomaly' and obs.volatility <= obs.rolling_vol_90p"
      value: -10.0
  aggregate: sum
  clamp: [-100.0, 100.0]
"""


def test_1_drift_detector_signals():
    """Verify DriftDetector computes Input PSI, Entropy Z-score, and Action Distribution shift."""
    adapter = CSVStreamAdapter(episode_length=50)
    dsl = RewardDSL.from_yaml(SAMPLE_DSL_YAML)

    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=dsl,
        hyperparameters={"learning_rate": 0.001, "gamma": 0.90, "n_steps": 32, "batch_size": 16},
    )
    runner.warmstart_policy(epochs=5, n_samples=50)
    episodes = runner.run_burst(n_episodes=4)

    # 1. Fit baseline
    feature_names = adapter.observations().feature_names
    baseline = DriftDetector.fit_baseline(episodes=episodes, feature_names=feature_names)

    assert len(baseline.quantiles_per_feature) == len(feature_names)
    assert baseline.entropy_mean > 0.0

    detector = DriftDetector(baseline)

    # 2. Healthy batch check: match baseline distribution
    healthy_obs = np.array([s.obs_vector for ep in episodes for s in ep.steps], dtype=np.float32)
    step_actions = [s.events.get("action", {}).get("action_idx", 0) for ep in episodes for s in ep.steps]
    healthy_probs = np.array(
        [[0.15, 0.85] if a == 1 else [0.85, 0.15] for a in step_actions],
        dtype=np.float32,
    )

    report_healthy = detector.check_drift(healthy_obs, healthy_probs)
    assert report_healthy.max_psi < 0.25
    assert not report_healthy.is_drifted

    # 3. Drifted batch check: shift input features by 50.0
    drifted_obs = healthy_obs + 50.0
    report_drifted = detector.check_drift(drifted_obs, healthy_probs)
    assert report_drifted.max_psi > 0.25
    assert report_drifted.is_drifted
    assert "Input PSI drift" in report_drifted.recommendation

    # 4. Drifted action distribution check
    drifted_action_probs = np.full((len(healthy_obs), 2), [0.05, 0.95], dtype=np.float32)
    report_act_drift = detector.check_drift(healthy_obs, drifted_action_probs)
    assert report_act_drift.action_shift_zscore > 3.0
    assert report_act_drift.is_drifted


def test_2_onnx_export_and_numerical_parity(tmp_path: Path):
    """Verify ONNX export achieves numerical parity with PyTorch policy across 32 test inputs."""
    adapter = CSVStreamAdapter(episode_length=30)
    dsl = RewardDSL.from_yaml(SAMPLE_DSL_YAML)

    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=dsl,
        hyperparameters={"learning_rate": 0.001, "gamma": 0.90, "n_steps": 32, "batch_size": 16},
    )
    runner.warmstart_policy(epochs=5, n_samples=50)

    onnx_file = tmp_path / "model.onnx"
    obs_dim = adapter.observations().vector_dim

    PolicyExporter.export_onnx(model=runner.model, output_path=onnx_file, obs_dim=obs_dim)
    assert onnx_file.exists()

    # Numerical parity gate (atol < 1e-5)
    max_diff = PolicyExporter.verify_parity(
        model=runner.model,
        onnx_path=onnx_file,
        obs_dim=obs_dim,
        n_samples=32,
    )
    print(f"ONNX Parity Max Diff: {max_diff:.3e}")
    assert max_diff < 1e-5


def test_3_runbundle_packaging_and_standalone_run_local(tmp_path: Path):
    """Verify .runbundle directory structure, ZIP archive, and standalone run_local.py execution."""
    adapter = CSVStreamAdapter(episode_length=30)
    dsl = RewardDSL.from_yaml(SAMPLE_DSL_YAML)
    config = TaskConfig(
        task_id="task_bundle_test",
        adapter_name="CSVStreamAdapter",
        reward_dsl=dsl,
    )

    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=dsl,
        hyperparameters={"learning_rate": 0.001, "gamma": 0.90, "n_steps": 32, "batch_size": 16},
    )
    runner.warmstart_policy(epochs=5, n_samples=50)
    episodes = runner.run_burst(n_episodes=3)

    bundle_dir, zip_path, max_diff = PolicyExporter.create_runbundle(
        task_config=config,
        reward_dsl=dsl,
        model=runner.model,
        episodes=episodes,
        output_dir=tmp_path / "exports",
        bundle_name="test_bundle",
    )

    assert bundle_dir.exists()
    assert (bundle_dir / "model.onnx").exists()
    assert (bundle_dir / "task_config.yaml").exists()
    assert (bundle_dir / "reward_dsl.yaml").exists()
    assert (bundle_dir / "drift_baseline.json").exists()
    assert (bundle_dir / "run_local.py").exists()
    assert zip_path.exists()
    assert max_diff < 1e-5

    # Execute standalone run_local.py via subprocess to prove zero-SB3 local inference
    res = subprocess.run(
        [sys.executable, "run_local.py", "--sample"],
        cwd=str(bundle_dir),
        capture_output=True,
        text=True,
    )

    print("run_local.py stdout:\n", res.stdout)
    if res.returncode != 0:
        print("run_local.py stderr:\n", res.stderr)

    assert res.returncode == 0
    assert "=== Local Policy Inference Results" in res.stdout
    assert "=== Unsupervised Drift Telemetry ===" in res.stdout
    assert "Drift Status:" in res.stdout


def test_4_fastapi_rest_endpoints_and_websocket(tmp_path: Path):
    """Verify all FastAPI REST endpoints and WebSocket stream with TestClient."""
    app = create_app()
    # Configure isolated runs directory
    app.state.app_state.state_store = type(app.state.app_state.state_store)(base_dir=tmp_path / "runs")

    with TestClient(app) as client:
        # 1. GET /api/adapters
        r_adapters = client.get("/api/adapters")
        assert r_adapters.status_code == 200
        adapters = r_adapters.json()["adapters"]
        assert any(a["name"] == "CSVStreamAdapter" for a in adapters)

        # 2. POST /api/tasks/compile
        r_compile = client.post(
            "/api/tasks/compile",
            json={
                "prompt": "Monitor price anomalies and volume spikes while ignoring normal volatility.",
                "adapter_name": "CSVStreamAdapter",
                "task_id": "test_api_task",
            },
        )
        assert r_compile.status_code == 200
        compile_data = r_compile.json()
        assert compile_data["status"] == "compiled"
        assert compile_data["version"] == 1

        # 3. POST /api/train/burst
        r_burst = client.post("/api/train/burst", json={"task_id": "test_api_task", "n_episodes": 3})
        assert r_burst.status_code == 200
        burst_data = r_burst.json()
        assert burst_data["status"] == "burst_completed"
        assert burst_data["episodes_count"] >= 3

        # 4. POST /api/episodes/{ep_id}/lock
        r_lock = client.post("/api/episodes/1/lock")
        assert r_lock.status_code == 200
        assert r_lock.json()["locked"] is True

        # Toggle unlock
        r_unlock = client.post("/api/episodes/1/lock")
        assert r_unlock.status_code == 200
        assert r_unlock.json()["locked"] is False

        # 5. POST /api/critique/preview
        r_preview = client.post(
            "/api/critique/preview",
            json={
                "task_id": "test_api_task",
                "target_episode_id": 1,
                "critique_text": "Penalize false alarms during volatility spikes",
            },
        )
        assert r_preview.status_code == 200
        preview_data = r_preview.json()
        assert "stage1_counterfactual_preview" in preview_data
        assert "stage2_forked_behavior_preview" in preview_data

        # 6. POST /api/critique/apply
        r_apply = client.post(
            "/api/critique/apply",
            json={
                "task_id": "test_api_task",
                "new_dsl": preview_data["new_dsl"],
                "critique_text": "Penalize false alarms during volatility spikes",
                "run_burst_after": False,
            },
        )
        assert r_apply.status_code == 200
        assert r_apply.json()["status"] == "applied"

        # 7. POST /api/tasks/rollback
        r_rollback = client.post("/api/tasks/rollback", json={"task_id": "test_api_task", "target_version": 1})
        assert r_rollback.status_code == 200
        assert r_rollback.json()["status"] == "rolled_back"
        assert r_rollback.json()["restored_version"] == 1

        # 8. POST /api/export
        r_export = client.post(
            "/api/export",
            json={"task_id": "test_api_task", "output_dir": str(tmp_path / "api_exports")},
        )
        assert r_export.status_code == 200
        export_data = r_export.json()
        assert export_data["status"] == "exported"
        assert export_data["parity_passed"] is True
        assert Path(export_data["zip_path"]).exists()

        # 9. WS /ws/stream
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text("ping")
            response = ws.receive_text()
            assert response == "pong"
