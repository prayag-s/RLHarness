"""Phase 0C Verification Suite.

Tests:
1. AdapterGymEnv space conformance, stepping, and evaluator hot-swapping.
2. EpisodeBuffer capacity enforcement and batch re-scoring.
3. TrainingRunner warm-start behavioral cloning and critic warmup calibration.
4. End-to-end critique alignment loop:
   - Burst 1 produces high false-positive rate on volatility spikes.
   - CritiqueEngine generates valid diff with Stage 1 score drop and Stage 2 action flip.
   - Diff applied with critic warmup.
   - Burst 2 reduces false-positive rate by >40%.
"""

from __future__ import annotations
import pytest
import numpy as np

from rl_ide.adapters.stream_adapter import CSVStreamAdapter
from rl_ide.reward.dsl import RewardDSL, RewardTerm
from rl_ide.reward.evaluator import RewardEvaluator
from rl_ide.trainer.gym_env import AdapterGymEnv
from rl_ide.trainer.buffer import EpisodeBuffer
from rl_ide.trainer.runner import TrainingRunner
from rl_ide.compiler.critique_engine import CritiqueEngine

INITIAL_DSL_YAML = """
version: "1.0"
horizon:
  gamma: 0.90
  critic_warmup_epochs_on_diff: 15
  kl_coef: 0.1
reward:
  terms:
    - name: accurate_match
      when: "truth.has_label == True and action.label == truth.label"
      value: 1.0
    - name: volatility_spike_bonus
      when: "action.label == 'anomaly' and obs.volatility > 0.005"
      value: 2.5
  aggregate: sum
  clamp: [-100.0, 100.0]
"""


def test_1_adapter_gym_env_and_hotswap():
    """Verify AdapterGymEnv spaces, step lifecycle, and hot-swapping evaluator."""
    adapter = CSVStreamAdapter(episode_length=50)
    dsl1 = RewardDSL.from_yaml(INITIAL_DSL_YAML)
    evaluator1 = RewardEvaluator(dsl1, schema=adapter.event_schema())

    env = AdapterGymEnv(adapter, evaluator1)
    obs, info = env.reset(seed=42)

    assert obs.shape == (9,)
    assert obs.dtype == np.float32
    assert env.action_space.n == 2

    # Step in env
    next_obs, reward, terminated, truncated, step_info = env.step(1)
    assert next_obs.shape == (9,)
    assert "breakdown" in step_info
    assert "events" in step_info

    # Hot-swap evaluator to a new penalty DSL
    dsl2_yaml = INITIAL_DSL_YAML.replace("value: 2.5", "value: -10.0")
    dsl2 = RewardDSL.from_yaml(dsl2_yaml)
    evaluator2 = RewardEvaluator(dsl2, schema=adapter.event_schema())

    env.set_evaluator(evaluator2)
    # Next step with action 1 must evaluate under the new evaluator
    _, reward2, _, _, _ = env.step(1)
    assert reward2 < reward


def test_2_episode_buffer_operations():
    """Verify EpisodeBuffer stores episodes up to max capacity and rescores them."""
    buffer = EpisodeBuffer(max_episodes=5)
    adapter = CSVStreamAdapter(episode_length=10)
    dsl = RewardDSL.from_yaml(INITIAL_DSL_YAML)
    evaluator = RewardEvaluator(dsl, schema=adapter.event_schema())
    env = AdapterGymEnv(adapter, evaluator)

    # Generate 7 episodes
    for ep in range(7):
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, info = env.step(0)
            done = term or trunc
            if done:
                buffer.append(info["episode_log"])

    # Max capacity is 5
    assert len(buffer) == 5
    recent = buffer.get_recent(3)
    assert len(recent) == 3

    # Rescore buffer
    initial_total = recent[-1].total_reward
    new_dsl = RewardDSL.from_yaml(INITIAL_DSL_YAML.replace("value: 1.0", "value: 5.0"))
    new_evaluator = RewardEvaluator(new_dsl, schema=adapter.event_schema())
    buffer.rescore(new_evaluator)
    assert recent[-1].total_reward != initial_total


def test_3_warmstart_behavioral_cloning():
    """Verify warm-start pass trains policy actor to >60% accuracy on heuristic cases."""
    adapter = CSVStreamAdapter(episode_length=50)
    dsl = RewardDSL.from_yaml(INITIAL_DSL_YAML)

    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=dsl,
        hyperparameters={"learning_rate": 0.001, "gamma": 0.90, "n_steps": 32, "batch_size": 16},
    )

    metrics = runner.warmstart_policy(epochs=12, n_samples=100, heuristic_flags_volatility=False)
    assert metrics["accuracy"] >= 0.65
    assert metrics["final_loss"] < metrics["initial_loss"]


def test_4_critique_diff_engine_and_two_stage_preview():
    """Verify CritiqueEngine produces valid diffs with Stage 1 and Stage 2 previews."""
    adapter = CSVStreamAdapter(episode_length=50)
    dsl = RewardDSL.from_yaml(INITIAL_DSL_YAML)
    schema = adapter.event_schema()

    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=dsl,
        hyperparameters={"learning_rate": 0.001, "gamma": 0.90, "n_steps": 32, "batch_size": 16},
    )

    runner.warmstart_policy(epochs=5, n_samples=50)
    episodes = runner.run_burst(n_episodes=3)

    # Find an episode with an anomaly alert
    target_ep = episodes[0]

    engine = CritiqueEngine(schema=schema, action_labels=adapter.actions().labels)
    critique = "It is flagging normal volatility as anomalies; penalize false positives when volatility is below rolling_vol_90p"

    proposal = engine.propose_diff(
        current_dsl=dsl,
        target_episode=target_ep,
        recent_episodes=episodes,
        locked_episodes=[],
        critique_text=critique,
        policy=runner.model,
    )

    # Diff checks
    assert len(proposal.term_diffs) > 0
    assert "false_positive_low_vol" in proposal.new_dsl.to_yaml()

    # Stage 1 preview check: penalty lowers reward
    assert proposal.stage1_counterfactual_preview is not None
    assert proposal.stage1_counterfactual_preview.target_episode_id == target_ep.episode_id
    assert proposal.stage1_counterfactual_preview.score_delta <= 0.0

    # Stage 2 preview check: forked micro-burst
    assert proposal.stage2_forked_behavior_preview is not None
    assert proposal.stage2_forked_behavior_preview.target_episode_id == target_ep.episode_id
    assert proposal.stage2_forked_behavior_preview.locked_episodes_preserved == 1.0


def test_5_full_critique_alignment_loop():
    """Test 5 (Master Plan Gate):
    - Burst 1 on CSVStreamAdapter over-flags volatility spikes as anomalies.
    - Submit critique: 'It is flagging normal volatility as anomalies; penalize false positives when volatility is below rolling_vol_90p'.
    - CritiqueEngine produces valid diff, Stage 1 drops score, Stage 2 flips action.
    - Apply diff + critic warmup.
    - Burst 2 reduces false-positive rate by >40%.
    """
    adapter = CSVStreamAdapter(episode_length=50)
    initial_dsl = RewardDSL.from_yaml(INITIAL_DSL_YAML)

    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=initial_dsl,
        hyperparameters={
            "learning_rate": 0.003,
            "gamma": 0.90,
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 5,
        },
    )

    # Warm-start
    runner.warmstart_policy(epochs=10, n_samples=100)

    # Burst 1: with volatility_spike_bonus (+2.5), agent aggressively alerts on volatility
    burst1_episodes = runner.run_burst(n_episodes=8)

    # Compute False Positive Rate in Burst 1
    fp_count_1 = 0
    total_steps_1 = 0
    target_fp_ep = None

    for ep in burst1_episodes:
        for s in ep.steps:
            total_steps_1 += 1
            is_alert = (s.events.get("action", {}).get("label") == "anomaly")
            is_true_anomaly = bool(s.events.get("truth", {}).get("true_anomaly"))
            if is_alert and not is_true_anomaly:
                fp_count_1 += 1
                if target_fp_ep is None:
                    target_fp_ep = ep

    fp_rate_1 = fp_count_1 / max(1, total_steps_1)
    assert target_fp_ep is not None, "Burst 1 should produce at least one false positive episode"
    assert fp_rate_1 > 0.03, f"Expected noticeable false positive rate in Burst 1, got {fp_rate_1:.2f}"

    # Critique submission
    critique_text = (
        "It is flagging normal volatility as anomalies; penalize false positives "
        "when volatility is below rolling_vol_90p"
    )

    engine = CritiqueEngine(
        schema=adapter.event_schema(),
        action_labels=adapter.actions().labels,
    )

    proposal = engine.propose_diff(
        current_dsl=initial_dsl,
        target_episode=target_fp_ep,
        recent_episodes=burst1_episodes,
        critique_text=critique_text,
        policy=runner.model,
    )

    # Verify Stage 1 and Stage 2 previews
    assert proposal.stage1_counterfactual_preview.score_delta < 0, "Stage 1 must show score drop on FP episode"
    assert proposal.stage2_forked_behavior_preview.target_flipped is True, "Stage 2 must flip action on target episode"

    # Apply diff & run Critic Warmup
    runner.apply_reward_diff(proposal.new_dsl, critique_text=critique_text)

    # Burst 2: Agent now faces heavy penalty for alerting on low/normal volatility
    burst2_episodes = runner.run_burst(n_episodes=12)

    fp_count_2 = 0
    total_steps_2 = 0
    for ep in burst2_episodes:
        for s in ep.steps:
            total_steps_2 += 1
            is_alert = (s.events.get("action", {}).get("label") == "anomaly")
            is_true_anomaly = bool(s.events.get("truth", {}).get("true_anomaly"))
            if is_alert and not is_true_anomaly:
                fp_count_2 += 1

    fp_rate_2 = fp_count_2 / max(1, total_steps_2)
    relative_reduction = (fp_rate_1 - fp_rate_2) / fp_rate_1

    print(f"\nBurst 1 FP Rate: {fp_rate_1:.3f} ({fp_count_1}/{total_steps_1})")
    print(f"Burst 2 FP Rate: {fp_rate_2:.3f} ({fp_count_2}/{total_steps_2})")
    print(f"Relative Reduction: {relative_reduction:.1%}")

    # Verify gate: FP rate reduced by >40%
    assert relative_reduction > 0.40, (
        f"Expected >40% false positive reduction, got {relative_reduction:.1%} "
        f"(Burst 1: {fp_rate_1:.3f}, Burst 2: {fp_rate_2:.3f})"
    )
