"""Cross-Domain Conformance Suite for DomainAdapter Protocol.

Validates that both reference adapters (CSVStreamAdapter and TextClassifyAdapter)
execute the complete end-to-end training and authoring loop:
    NLToConfigCompiler
    -> warmstart_policy
    -> run_burst
    -> CritiqueEngine (Two-Stage Preview)
    -> apply_reward_diff (Critic Warmup)
    -> PolicyExporter (ONNX numerical parity < 1e-5)
    -> DriftDetector (Baseline & Report)
"""

from __future__ import annotations
import numpy as np
import pytest
import torch

from rl_ide.contracts.adapter import DomainAdapter
from rl_ide.adapters.stream_adapter import CSVStreamAdapter
from rl_ide.adapters.text_classify_adapter import TextClassifyAdapter
from rl_ide.compiler.nl_to_config import NLToConfigCompiler
from rl_ide.compiler.critique_engine import CritiqueEngine
from rl_ide.trainer.runner import TrainingRunner
from rl_ide.export.exporter import PolicyExporter
from rl_ide.drift.monitor import DriftDetector


@pytest.mark.parametrize("adapter_name,adapter_cls,test_prompt,critique_text", [
    (
        "CSVStreamAdapter",
        CSVStreamAdapter,
        "Detect sudden price jumps and volume anomalies in the financial stream while tolerating normal volatility chop.",
        "It is flagging normal volatility as anomalies; heavily penalize false positives when volatility is low.",
    ),
    (
        "TextClassifyAdapter",
        TextClassifyAdapter,
        "Classify support tickets into urgent, normal, spam, and escalate based on urgency, angry tone, and VIP status.",
        "Never route spam as urgent; heavily penalize routing spam to urgent.",
    ),
])
def test_full_cross_domain_conformance_pipeline(
    tmp_path,
    adapter_name: str,
    adapter_cls: type[DomainAdapter],
    test_prompt: str,
    critique_text: str,
):
    """Verifies that the entire RL-Native orchestrator executes seamlessly across domains."""
    # 1. Instantiate Adapter and verify Protocol conformance
    adapter = adapter_cls()
    assert isinstance(adapter, DomainAdapter), f"{adapter_name} must conform to DomainAdapter Protocol"

    obs_spec = adapter.observations()
    action_spec = adapter.actions()
    schema = adapter.event_schema()

    assert obs_spec.vector_dim > 0
    assert len(action_spec.labels) >= 2
    assert len(schema) > 0

    # 2. Compile Natural Language prose into TaskConfig
    compiler = NLToConfigCompiler()
    task_config = compiler.compile(
        user_prompt=test_prompt,
        adapter_name=adapter_name,
        adapter=adapter,
        task_id=f"test_{adapter_name.lower()}",
    )

    assert task_config.adapter_name == adapter_name
    assert len(task_config.reward_dsl.reward.terms) > 0

    # 3. Initialize Training Runner and Warm-Start Policy (Behavioral Cloning)
    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=task_config.reward_dsl,
        hyperparameters={
            "learning_rate": 0.002,
            "gamma": float(obs_spec.default_gamma),
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 4,
        },
    )

    ws_metrics = runner.warmstart_policy(epochs=6, n_samples=60)
    assert "final_loss" in ws_metrics
    assert "accuracy" in ws_metrics

    # 4. Run Training Micro-Burst (Streaming PPO)
    burst1_episodes = runner.run_burst(n_episodes=4)
    assert len(burst1_episodes) >= 4
    for ep in burst1_episodes:
        assert len(ep.steps) > 0
        assert len(ep.step_rewards) == len(ep.steps)
        # Verify step events conform to schema
        for s in ep.steps:
            assert "obs" in s.events
            assert "action" in s.events
            assert "truth" in s.events

    # 5. Two-Stage Critique Diff Proposal
    engine = CritiqueEngine(schema=schema, action_labels=action_spec.labels)
    target_ep = burst1_episodes[0]

    proposal = engine.propose_diff(
        current_dsl=task_config.reward_dsl,
        target_episode=target_ep,
        recent_episodes=burst1_episodes,
        critique_text=critique_text,
        policy=runner.model,
    )

    assert proposal.diff_yaml != "", "Diff YAML must be generated"
    assert len(proposal.term_diffs) > 0, "At least one term diff must be proposed"
    assert proposal.stage1_counterfactual_preview is not None
    assert proposal.stage2_forked_behavior_preview is not None

    # 6. Apply Diff and execute Critic Warmup
    runner.apply_reward_diff(proposal.new_dsl, critique_text=critique_text)
    assert runner.reward_dsl == proposal.new_dsl

    # 7. Post-critique burst
    burst2_episodes = runner.run_burst(n_episodes=4)
    assert len(burst2_episodes) >= 4

    # 8. ONNX Policy Export & Strict Numerical Parity Check (< 1e-5)
    onnx_path = tmp_path / f"{adapter_name}_model.onnx"
    PolicyExporter.export_onnx(
        model=runner.model,
        output_path=onnx_path,
        obs_dim=obs_spec.vector_dim,
    )
    max_diff = PolicyExporter.verify_parity(
        model=runner.model,
        onnx_path=onnx_path,
        obs_dim=obs_spec.vector_dim,
    )

    assert max_diff < 1e-5, f"Max diff {max_diff} exceeded 1e-5 gate"

    # 9. Standalone Runbundle creation
    bundle_dir, zip_path, b_diff = PolicyExporter.create_runbundle(
        task_config=task_config,
        reward_dsl=runner.reward_dsl,
        model=runner.model,
        episodes=runner.buffer.all(),
        output_dir=str(tmp_path / "bundles"),
        bundle_name=f"bundle_{adapter_name}",
    )
    assert bundle_dir.exists()
    assert (bundle_dir / "model.onnx").exists()
    assert (bundle_dir / "run_local.py").exists()
    assert (bundle_dir / "drift_baseline.json").exists()

    # 10. Unsupervised Drift Detector baseline fit & check
    baseline = DriftDetector.fit_baseline(
        episodes=runner.buffer.all(),
        feature_names=obs_spec.feature_names,
    )
    assert len(baseline.quantiles_per_feature) > 0

    detector = DriftDetector(baseline)
    sample_obs = np.array([s.obs_vector for s in target_ep.steps], dtype=np.float32)
    sample_probs = np.full((len(sample_obs), len(action_spec.labels)), 1.0 / len(action_spec.labels))
    report = detector.check_drift(sample_obs, sample_probs)
    assert isinstance(report.is_drifted, bool)
