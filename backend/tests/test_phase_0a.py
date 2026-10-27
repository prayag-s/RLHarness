"""Phase 0A Verification Suite.

Tests:
1. Valid YAML DSL evaluates accurately and returns per-term breakdown.
2. Undefined variable (e.g. obs.non_existent_field > 5) raises RewardValidationError during static check.
3. Malicious code injection (e.g. __import__('os').system('ls')) raises RewardSecurityError.
4. CSVStreamAdapter resets, steps through an episode, and rescore_episodes()
   re-evaluates 100 logged steps in <10ms when a term weight changes.
"""

from __future__ import annotations
import time
import pytest
from rl_ide.contracts.adapter import (
    ObservationSpec,
    ActionSpec,
    StepResult,
    EpisodeLog,
    DomainAdapter,
)
from rl_ide.reward.dsl import RewardDSL
from rl_ide.reward.evaluator import (
    RewardEvaluator,
    RewardValidator,
    RewardValidationError,
    RewardSecurityError,
)
from rl_ide.adapters.stream_adapter import CSVStreamAdapter, generate_synthetic_stream

VALID_DSL_YAML = """
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
    - name: false_positive_low_vol
      when: "action.label == 'anomaly' and obs.volatility <= obs.rolling_vol_90p"
      value: -8.0
    - name: high_vol_anomaly_bonus
      when: "action.label == 'anomaly' and obs.volatility > obs.rolling_vol_90p and state.cooldown == 0"
      value: 1.5
    - name: alert_spam_penalty
      when: "action.label == 'anomaly' and state.rolling_alert_rate > 0.15"
      value: -4.0
  pinned_episodes:
    - episode_id: 47
      forbid_action: "anomaly"
      penalty: -10.0
  aggregate: sum
  clamp: [-100.0, 100.0]
"""


def test_synthetic_stream_generation():
    """Verify synthetic stream produces required columns and invariant behaviors."""
    df = generate_synthetic_stream(n_rows=1000, seed=42)
    expected_cols = [
        "timestamp",
        "price",
        "return_1m",
        "volume",
        "volatility",
        "rolling_vol_90p",
        "z_score",
        "true_anomaly",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 1000

    # Invariant: high volatility spikes exist that are normal (not true anomalies)
    high_vol_normal = df[(df["volatility"] > df["rolling_vol_90p"]) & (~df["true_anomaly"])]
    assert len(high_vol_normal) > 0, "Should have normal rows during high volatility periods"

    # Invariant: true anomalies exist and have extreme z_score or volume
    true_anomalies = df[df["true_anomaly"]]
    assert len(true_anomalies) > 0, "Should have true anomalies"
    for _, row in true_anomalies.iterrows():
        assert abs(row["z_score"]) >= 2.5 or row["volume"] > 20000.0


def test_1_valid_yaml_dsl_evaluation():
    """Test 1: Valid YAML DSL evaluates accurately and returns per-term breakdown."""
    adapter = CSVStreamAdapter()
    schema = adapter.event_schema()

    dsl = RewardDSL.from_yaml(VALID_DSL_YAML)
    evaluator = RewardEvaluator(dsl, schema=schema)

    # Case A: Accurate match (normal action on normal data)
    events_normal = {
        "obs": {
            "price": 100.0,
            "return_1m": 0.0005,
            "volume": 9500.0,
            "volatility": 0.005,
            "rolling_vol_90p": 0.012,
            "z_score": 0.2,
        },
        "action": {"action_idx": 0, "label": "normal"},
        "state": {"cooldown": 0, "steps_since_last_alert": 10, "rolling_alert_rate": 0.0},
        "truth": {"has_label": True, "label": "normal", "true_anomaly": False},
    }

    reward, breakdown = evaluator.evaluate(episode_id=1, events=events_normal)
    assert reward == 1.0
    assert breakdown["accurate_match"] == 1.0
    assert breakdown["false_positive_low_vol"] == 0.0
    assert breakdown["high_vol_anomaly_bonus"] == 0.0
    assert breakdown["alert_spam_penalty"] == 0.0

    # Case B: False positive with low volatility
    events_fp = {
        "obs": {
            "price": 100.0,
            "return_1m": 0.0005,
            "volume": 9500.0,
            "volatility": 0.005,
            "rolling_vol_90p": 0.012,
            "z_score": 0.2,
        },
        "action": {"action_idx": 1, "label": "anomaly"},
        "state": {"cooldown": 0, "steps_since_last_alert": 10, "rolling_alert_rate": 0.0},
        "truth": {"has_label": True, "label": "normal", "true_anomaly": False},
    }

    reward_fp, breakdown_fp = evaluator.evaluate(episode_id=1, events=events_fp)
    assert reward_fp == -8.0
    assert breakdown_fp["accurate_match"] == 0.0
    assert breakdown_fp["false_positive_low_vol"] == -8.0

    # Case C: Pinned episode 47 penalty constraint
    reward_pinned, breakdown_pinned = evaluator.evaluate(episode_id=47, events=events_fp)
    # false_positive_low_vol (-8.0) + pinned_ep_47_penalty (-10.0) = -18.0
    assert reward_pinned == -18.0
    assert breakdown_pinned["pinned_ep_47_penalty"] == -10.0


def test_2_undefined_variable_raises_validation_error():
    """Test 2: Undefined variable raises RewardValidationError during static check."""
    adapter = CSVStreamAdapter()
    schema = adapter.event_schema()

    # Case A: Non-existent attribute on namespace
    invalid_dsl_yaml = VALID_DSL_YAML.replace(
        "obs.volatility <= obs.rolling_vol_90p",
        "obs.non_existent_field > 5",
    )
    dsl = RewardDSL.from_yaml(invalid_dsl_yaml)

    with pytest.raises(RewardValidationError) as exc_info:
        RewardValidator.validate(dsl, schema=schema)
    assert "obs.non_existent_field" in str(exc_info.value)

    # Case B: RewardEvaluator constructor also validates
    with pytest.raises(RewardValidationError):
        RewardEvaluator(dsl, schema=schema)

    # Case C: Standalone un-namespaced variable
    invalid_dsl_yaml_2 = VALID_DSL_YAML.replace(
        "obs.volatility <= obs.rolling_vol_90p",
        "unnamespaced_var > 5",
    )
    dsl2 = RewardDSL.from_yaml(invalid_dsl_yaml_2)
    with pytest.raises(RewardValidationError):
        RewardValidator.validate(dsl2, schema=schema)


def test_3_malicious_code_injection_raises_security_error():
    """Test 3: Malicious code injection raises RewardSecurityError."""
    adapter = CSVStreamAdapter()
    schema = adapter.event_schema()

    malicious_payloads = [
        "__import__('os').system('ls')",
        "__import__('subprocess').run(['echo', 'hacked'])",
        "open('/etc/passwd').read()",
        "obs.__class__.__base__.__subclasses__()",
        "exec('x = 1')",
        "(lambda: 42)()",
    ]

    for payload in malicious_payloads:
        # Test directly via validate_expression
        with pytest.raises(RewardSecurityError):
            RewardValidator.validate_expression(payload, schema=schema)

        # Test within DSL
        bad_dsl_yaml = f"""
version: "1.0"
reward:
  terms:
    - name: exploit
      when: "{payload}"
      value: 1.0
"""
        bad_dsl = RewardDSL.from_yaml(bad_dsl_yaml)
        with pytest.raises(RewardSecurityError):
            RewardValidator.validate(bad_dsl, schema=schema)

        with pytest.raises(RewardSecurityError):
            RewardEvaluator(bad_dsl, schema=schema)


def test_4_adapter_episode_stepping_and_fast_rescoring():
    """Test 4: CSVStreamAdapter resets, steps through an episode, and rescore_episodes()
    re-evaluates 100 logged steps in <10ms when a term weight changes.
    """
    adapter = CSVStreamAdapter(episode_length=100)
    schema = adapter.event_schema()

    # 1. Verify contract adherence
    assert isinstance(adapter, DomainAdapter)
    obs_spec = adapter.observations()
    assert isinstance(obs_spec, ObservationSpec)
    assert obs_spec.vector_dim == 9
    assert obs_spec.horizon_type == "stateful_mdp"
    assert obs_spec.default_gamma == 0.90

    action_spec = adapter.actions()
    assert isinstance(action_spec, ActionSpec)
    assert action_spec.labels == ["normal", "anomaly"]

    # 2. Reset and step through 100 steps
    init_obs, init_info = adapter.reset(seed=42)
    assert len(init_obs) == 9
    assert "episode_id" in init_info

    steps: list[StepResult] = []
    for step_i in range(100):
        # Trigger an anomaly alert periodically
        action_idx = 1 if (step_i % 15 == 0) else 0
        step_result = adapter.step(action_idx)
        steps.append(step_result)

    assert len(steps) == 100
    assert steps[-1].terminated is True

    # Test render_frame
    frame = adapter.render_frame(
        episode_id=1,
        step_result=steps[0],
        total_reward=1.0,
        breakdown={"accurate_match": 1.0},
    )
    assert frame["episode_id"] == 1
    assert "obs" in frame and "action" in frame and "state" in frame

    # 3. Create EpisodeLog
    ep_log = EpisodeLog(episode_id=1, steps=steps)

    # 4. Evaluate initial DSL
    dsl_v1 = RewardDSL.from_yaml(VALID_DSL_YAML)
    evaluator_v1 = RewardEvaluator(dsl_v1, schema=schema)
    evaluator_v1.rescore_episodes([ep_log])

    initial_total = ep_log.total_reward
    assert len(ep_log.step_rewards) == 100
    assert len(ep_log.step_breakdowns) == 100

    # 5. Modify term weight: false_positive_low_vol -8.0 -> -20.0
    dsl_v2_yaml = VALID_DSL_YAML.replace("value: -8.0", "value: -20.0")
    dsl_v2 = RewardDSL.from_yaml(dsl_v2_yaml)
    evaluator_v2 = RewardEvaluator(dsl_v2, schema=schema)

    # 6. Benchmark rescore_episodes() across the 100 steps (<10ms gate)
    t0 = time.perf_counter()
    evaluator_v2.rescore_episodes([ep_log])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Rescored 100 logged steps in: {elapsed_ms:.3f} ms")
    assert elapsed_ms < 10.0, f"Rescoring took {elapsed_ms:.3f}ms, must be <10ms"

    # Reward must have changed due to heavier false positive penalty
    assert ep_log.total_reward != initial_total
    assert ep_log.total_reward < initial_total
