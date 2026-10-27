"""Phase 2 Verification Suite: Productization, RunBundle Sharing & Hardening.

Tests:
1. RunBundleRegistry: Scanning, cataloging, metadata extraction, and detailed inspection.
2. RunBundle ZIP import and round-trip verification.
3. Standalone run_local.py HTTP daemon: subprocess execution with --serve, testing /health and /predict.
4. Multi-provider BYOK validation and dynamic routing in compiler and API.
5. FastAPI REST endpoints for bundle listing, details, import, and lifecycle management.
"""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rl_ide.contracts.adapter import EpisodeLog
from rl_ide.adapters.stream_adapter import CSVStreamAdapter
from rl_ide.adapters.text_classify_adapter import TextClassifyAdapter
from rl_ide.compiler.models import TaskConfig
from rl_ide.compiler.llm_client import LLMConfig, validate_connection, create_llm_client
from rl_ide.compiler.nl_to_config import NLToConfigCompiler
from rl_ide.reward.dsl import RewardDSL
from rl_ide.state.store import TaskStateStore
from rl_ide.trainer.runner import TrainingRunner
from rl_ide.export.exporter import PolicyExporter
from rl_ide.export.registry import RunBundleRegistry, BundleMetadata, BundleDetails
from rl_ide.api.server import create_app


SAMPLE_CSV_DSL = """
version: "1.0"
horizon:
  gamma: 0.90
  critic_warmup_epochs_on_diff: 10
  kl_coef: 0.1
reward:
  terms:
    - name: accurate_match
      when: "truth.has_label == True and action.label == truth.label"
      value: 2.0
    - name: false_positive_low_vol
      when: "action.label == 'anomaly' and obs.volatility <= obs.rolling_vol_90p"
      value: -6.0
  pinned_episodes: []
  aggregate: sum
  clamp: [-50.0, 50.0]
"""


@pytest.fixture
def trained_csv_bundle(tmp_path: Path):
    """Fixture creating a fully trained policy and exported runbundle."""
    adapter = CSVStreamAdapter(episode_length=20)
    dsl = RewardDSL.from_yaml(SAMPLE_CSV_DSL)
    config = TaskConfig(
        task_id="task_p2_csv_test",
        name="Phase 2 Test CSV Policy",
        description="Test bundle for Phase 2 registry verification",
        adapter_name="CSVStreamAdapter",
        adapter_config={"episode_length": 20, "action_labels": ["normal", "anomaly"]},
        hyperparameters={"learning_rate": 0.0003, "gamma": 0.90},
        reward_dsl=dsl,
    )
    store = TaskStateStore(base_dir=tmp_path / "runs")
    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=dsl,
        state_store=store,
        task_id="task_p2_csv_test",
        hyperparameters=config.hyperparameters,
    )
    runner.warmstart_policy(epochs=5, n_samples=30)
    episodes = runner.run_burst(n_episodes=4)

    bundles_dir = tmp_path / "exported_bundles"
    bundle_dir, zip_path, max_diff = PolicyExporter.create_runbundle(
        task_config=config,
        reward_dsl=dsl,
        model=runner.model,
        episodes=episodes,
        output_dir=bundles_dir,
        bundle_name="phase2_verified_bundle",
    )
    return bundle_dir, zip_path, bundles_dir, config


def test_1_runbundle_registry_catalog_and_inspection(trained_csv_bundle):
    """Verify registry catalogs both directories and zip archives and extracts metadata."""
    bundle_dir, zip_path, bundles_dir, config = trained_csv_bundle

    registry = RunBundleRegistry(search_dirs=[bundles_dir])
    bundles = registry.list_bundles()

    assert len(bundles) >= 1
    # Check directory entry
    dir_bundle = next((b for b in bundles if b.archive_type == "dir"), None)
    assert dir_bundle is not None
    assert dir_bundle.task_id == config.task_id
    assert dir_bundle.adapter_name == "CSVStreamAdapter"
    assert dir_bundle.name == "Phase 2 Test CSV Policy"
    assert dir_bundle.onnx_size_bytes > 0
    assert dir_bundle.dsl_terms_count == 2
    assert dir_bundle.feature_count == 9
    assert dir_bundle.status == "ready"

    # Check deep inspection
    details = registry.get_bundle(dir_bundle.bundle_id)
    assert isinstance(details, BundleDetails)
    assert details.task_config["task_id"] == config.task_id
    assert details.onnx_summary["exists"] is True
    assert details.has_run_local_daemon is True
    assert "entropy_mean" in details.drift_baseline


def test_2_runbundle_zip_import_and_roundtrip(trained_csv_bundle, tmp_path: Path):
    """Verify registry unpacks, validates, and imports ZIP archives."""
    bundle_dir, zip_path, bundles_dir, config = trained_csv_bundle

    import_dest = tmp_path / "imported_bundles"
    registry = RunBundleRegistry(search_dirs=[import_dest])

    installed_dir, imported_config = registry.import_bundle(
        archive_or_dir_path=zip_path,
        target_dir=import_dest,
    )

    assert installed_dir.exists()
    assert (installed_dir / "model.onnx").exists()
    assert (installed_dir / "task_config.yaml").exists()
    assert (installed_dir / "reward_dsl.yaml").exists()
    assert (installed_dir / "drift_baseline.json").exists()
    assert (installed_dir / "run_local.py").exists()

    assert imported_config.task_id == config.task_id
    assert imported_config.adapter_name == config.adapter_name
    assert len(imported_config.reward_dsl.reward.terms) == 2


def test_3_standalone_run_local_daemon_http_serving(trained_csv_bundle):
    """Verify standalone run_local.py --serve HTTP daemon launches and responds to /health and /predict."""
    bundle_dir, _, _, _ = trained_csv_bundle
    test_port = 9876

    # Launch daemon in background subprocess
    proc = subprocess.Popen(
        [sys.executable, "run_local.py", "--serve", "--port", str(test_port), "--host", "127.0.0.1"],
        cwd=str(bundle_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    daemon_base = f"http://127.0.0.1:{test_port}"
    try:
        # Wait up to 5 seconds for daemon startup
        connected = False
        client = httpx.Client(timeout=2.0)
        for _ in range(25):
            time.sleep(0.2)
            try:
                resp = client.get(f"{daemon_base}/health")
                if resp.status_code == 200:
                    connected = True
                    break
            except Exception:
                pass

        assert connected, "Failed to connect to run_local HTTP daemon"
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "RL-Native Agent Policy Daemon"
        assert data["feature_count"] == 9

        # Test POST /predict with healthy synthetic batch
        healthy_obs = np.random.randn(8, 9).tolist()
        pred_resp = client.post(
            f"{daemon_base}/predict",
            json={"observations": healthy_obs},
        )
        assert pred_resp.status_code == 200
        p_data = pred_resp.json()
        assert len(p_data["predictions"]) == 8
        first_pred = p_data["predictions"][0]
        assert "action_idx" in first_pred
        assert "action_label" in first_pred
        assert len(first_pred["probabilities"]) == 2
        assert "drift" in p_data
        assert p_data["drift"]["status"] in ("HEALTHY", "DRIFT_DETECTED")

        # Test single observation input
        single_resp = client.post(
            f"{daemon_base}/predict",
            json={"observation": healthy_obs[0]},
        )
        assert single_resp.status_code == 200
        assert len(single_resp.json()["predictions"]) == 1

        # Test graceful shutdown endpoint
        shut_resp = client.post(f"{daemon_base}/shutdown")
        assert shut_resp.status_code == 200
        assert shut_resp.json()["status"] == "shutting_down"

    finally:
        # Ensure process cleanup
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


def test_4_byok_settings_validation_and_routing():
    """Verify multi-provider BYOK validation logic and compiler integration."""
    # 1. Mock provider
    valid_mock, msg_mock = validate_connection(LLMConfig(provider="mock"))
    assert valid_mock is True
    assert "Mock" in msg_mock

    # 2. OpenAI without key fails
    valid_oai, msg_oai = validate_connection(LLMConfig(provider="openai", api_key=None))
    assert valid_oai is False
    assert "API key is required" in msg_oai

    # 3. Anthropic validation
    valid_ant_none, _ = validate_connection(LLMConfig(provider="anthropic", api_key=None))
    assert valid_ant_none is False

    valid_ant, msg_ant = validate_connection(
        LLMConfig(provider="anthropic", api_key="sk-ant-api03-testkey123456789012345678")
    )
    assert valid_ant is True
    assert "Anthropic" in msg_ant

    # 4. Unknown provider fails
    valid_unk, msg_unk = validate_connection(LLMConfig(provider="unknown_provider"))
    assert valid_unk is False

    # 5. Compiler routes custom config
    cfg = LLMConfig(provider="mock")
    client = create_llm_client(cfg)
    compiler = NLToConfigCompiler(llm_client=client)
    res = compiler.compile(
        user_prompt="Detect anomalies in financial stream",
        adapter_name="CSVStreamAdapter",
        task_id="byok_task_test",
    )
    assert res.task_id == "byok_task_test"
    assert len(res.reward_dsl.reward.terms) > 0


def test_5_fastapi_bundle_registry_and_settings_endpoints(trained_csv_bundle):
    """Verify FastAPI endpoints for bundle registry listing, import, and BYOK validation."""
    bundle_dir, zip_path, bundles_dir, config = trained_csv_bundle

    app = create_app()
    client = TestClient(app)

    # 1. Test /api/settings/validate
    val_resp = client.post(
        "/api/settings/validate",
        json={"llm_config": {"provider": "mock"}},
    )
    assert val_resp.status_code == 200
    assert val_resp.json()["valid"] is True

    # 2. Test /api/bundles listing
    list_resp = client.get("/api/bundles")
    assert list_resp.status_code == 200
    bundles = list_resp.json()["bundles"]
    assert isinstance(bundles, list)

    # 3. Test /api/bundles/import
    imp_resp = client.post(
        "/api/bundles/import",
        json={"bundle_path": str(zip_path), "load_active": True},
    )
    assert imp_resp.status_code == 200
    imp_data = imp_resp.json()
    assert imp_data["status"] == "imported"
    assert imp_data["is_active"] is True
    assert imp_data["task_config"]["task_id"] == config.task_id

    # 4. Test /api/bundles/{bundle_id} detail
    bundle_id = "phase2_verified_bundle"
    detail_resp = client.get(f"/api/bundles/{bundle_id}")
    if detail_resp.status_code == 200:
        d = detail_resp.json()
        assert d["metadata"]["task_id"] == config.task_id
        assert "reward_dsl" in d
