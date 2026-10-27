"""Phase 0B Verification Suite.

Tests:
1. Compiling English task prose into a valid CSVStreamAdapter config.
2. Automatic validator retry when an invalid field name is proposed.
3. Saving 3 sequential DSL versions and rolling back from v3 to v1.
"""

from __future__ import annotations
from pathlib import Path
import pytest

from rl_ide.adapters.stream_adapter import CSVStreamAdapter
from rl_ide.compiler.models import TaskConfig
from rl_ide.compiler.llm_client import (
    LLMConfig,
    MockRulesLLMClient,
    create_llm_client,
)
from rl_ide.compiler.nl_to_config import NLToConfigCompiler
from rl_ide.reward.dsl import RewardDSL, RewardTerm
from rl_ide.reward.evaluator import (
    RewardEvaluator,
    RewardValidator,
    RewardValidationError,
)
from rl_ide.state.store import TaskStateStore


def test_1_compile_english_prose_to_valid_config():
    """Test 1: Test compiling English task prose into a valid CSVStreamAdapter config."""
    adapter = CSVStreamAdapter()
    schema = adapter.event_schema()

    client = MockRulesLLMClient()
    compiler = NLToConfigCompiler(llm_client=client)

    user_prompt = (
        "Monitor financial time-series stream. Flag true anomalies where returns and volume jump. "
        "Do not alert on normal market volatility. Heavily penalize false positives."
    )

    task_config = compiler.compile(
        user_prompt=user_prompt,
        adapter_name="CSVStreamAdapter",
        adapter=adapter,
    )

    assert isinstance(task_config, TaskConfig)
    assert task_config.adapter_name == "CSVStreamAdapter"
    assert len(task_config.reward_dsl.reward.terms) >= 3

    # Verify that the compiled RewardDSL passes static validation
    RewardValidator.validate(task_config.reward_dsl, schema=schema)

    # Verify evaluator evaluates events without error
    evaluator = RewardEvaluator(task_config.reward_dsl, schema=schema)
    sample_events = {
        "obs": {
            "price": 100.0,
            "return_1m": 0.001,
            "volume": 10000.0,
            "volatility": 0.005,
            "rolling_vol_90p": 0.012,
            "z_score": 0.3,
        },
        "action": {"action_idx": 0, "label": "normal"},
        "state": {"cooldown": 0, "steps_since_last_alert": 5, "rolling_alert_rate": 0.02},
        "truth": {"has_label": True, "label": "normal", "true_anomaly": False},
    }
    total_reward, breakdown = evaluator.evaluate(episode_id=1, events=sample_events)
    assert isinstance(total_reward, float)
    assert "accurate_match" in breakdown


def test_2_automatic_validator_retry_on_invalid_field():
    """Test 2: Test automatic validator retry when an invalid field name is proposed."""
    adapter = CSVStreamAdapter()
    schema = adapter.event_schema()

    # Attempt 1: proposes an invalid field 'obs.hallucinated_metric'
    bad_yaml = """
task_id: "test_task"
name: "Invalid Field Task"
adapter_name: "CSVStreamAdapter"
reward_dsl:
  version: "1.0"
  reward:
    terms:
      - name: bad_term
        when: "obs.hallucinated_metric > 10.0"
        value: 1.0
"""

    # Attempt 2: repaired YAML with valid field 'obs.volatility'
    good_yaml = """
task_id: "test_task"
name: "Repaired Field Task"
adapter_name: "CSVStreamAdapter"
reward_dsl:
  version: "1.0"
  reward:
    terms:
      - name: fixed_term
        when: "obs.volatility > 0.01"
        value: 1.0
"""

    # Client scripts bad_yaml then good_yaml
    scripted_client = MockRulesLLMClient(scripted_responses=[bad_yaml, good_yaml])
    compiler = NLToConfigCompiler(llm_client=scripted_client, max_retries=2)

    # Should succeed on the retry attempt
    task_config = compiler.compile(
        user_prompt="Flag volatility",
        adapter_name="CSVStreamAdapter",
        adapter=adapter,
    )

    assert task_config.name == "Repaired Field Task"
    assert task_config.reward_dsl.reward.terms[0].name == "fixed_term"
    assert "obs.volatility" in str(task_config.reward_dsl.reward.terms[0].when)

    # Also verify that if all retries persistently fail, RewardValidationError is raised
    persistently_bad_client = MockRulesLLMClient(
        scripted_responses=[bad_yaml, bad_yaml, bad_yaml]
    )
    strict_compiler = NLToConfigCompiler(llm_client=persistently_bad_client, max_retries=1)

    with pytest.raises(RewardValidationError) as exc_info:
        strict_compiler.compile(
            user_prompt="Flag volatility",
            adapter_name="CSVStreamAdapter",
            adapter=adapter,
        )
    assert "obs.hallucinated_metric" in str(exc_info.value)


def test_3_saving_sequential_versions_and_rollback(tmp_path: Path):
    """Test 3: Test saving 3 sequential DSL versions and rolling back from v3 to v1."""
    store = TaskStateStore(base_dir=tmp_path / "runs")
    task_id = "task_stock_anomalies"

    # Create dummy checkpoint files
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt1 = ckpt_dir / "ckpt_v1.pt"
    ckpt2 = ckpt_dir / "ckpt_v2.pt"
    ckpt3 = ckpt_dir / "ckpt_v3.pt"
    ckpt1.write_text("weights_v1")
    ckpt2.write_text("weights_v2")
    ckpt3.write_text("weights_v3")

    base_dsl_yaml = """
version: "1.0"
horizon:
  gamma: 0.90
reward:
  terms:
    - name: accurate_match
      when: "truth.has_label == True and action.label == truth.label"
      value: 1.0
  aggregate: sum
  clamp: [-100.0, 100.0]
"""

    dsl_v1 = RewardDSL.from_yaml(base_dsl_yaml)
    config_v1 = TaskConfig(
        task_id=task_id,
        name="Anomaly Monitor v1",
        adapter_name="CSVStreamAdapter",
        reward_dsl=dsl_v1,
    )

    # Save Version 1
    v1 = store.save_version(
        task_id=task_id,
        task_config=config_v1,
        reward_dsl=dsl_v1,
        policy_checkpoint_path=str(ckpt1),
        locked_episode_ids=[47],
    )
    assert v1.version == 1
    assert v1.version_tag == "v1"
    assert v1.locked_episode_ids == [47]
    assert Path(v1.policy_checkpoint_path).read_text() == "weights_v1"

    # Save Version 2 (critique: add false positive penalty)
    dsl_v2 = RewardDSL.from_yaml(base_dsl_yaml)
    dsl_v2.reward.terms.append(
        RewardTerm(
            name="false_positive_low_vol",
            when="action.label == 'anomaly' and obs.volatility <= obs.rolling_vol_90p",
            value=-10.0,
        )
    )
    config_v2 = TaskConfig(
        task_id=task_id,
        name="Anomaly Monitor v2",
        adapter_name="CSVStreamAdapter",
        reward_dsl=dsl_v2,
    )
    v2 = store.save_version(
        task_id=task_id,
        task_config=config_v2,
        reward_dsl=dsl_v2,
        applied_critique="Penalize false alarms during low volatility",
        policy_checkpoint_path=str(ckpt2),
        locked_episode_ids=[47, 52],
    )
    assert v2.version == 2
    assert v2.version_tag == "v2"
    assert len(v2.reward_dsl.reward.terms) == 2
    assert v2.locked_episode_ids == [47, 52]

    # Save Version 3 (critique: add alert spam penalty)
    dsl_v3 = RewardDSL.from_yaml(dsl_v2.to_yaml())
    dsl_v3.reward.terms.append(
        RewardTerm(
            name="alert_spam_penalty",
            when="action.label == 'anomaly' and state.rolling_alert_rate > 0.15",
            value=-8.0,
        )
    )
    config_v3 = TaskConfig(
        task_id=task_id,
        name="Anomaly Monitor v3",
        adapter_name="CSVStreamAdapter",
        reward_dsl=dsl_v3,
    )
    v3 = store.save_version(
        task_id=task_id,
        task_config=config_v3,
        reward_dsl=dsl_v3,
        applied_critique="Penalize alert spamming",
        policy_checkpoint_path=str(ckpt3),
        locked_episode_ids=[47, 52, 99],
    )
    assert v3.version == 3
    assert len(v3.reward_dsl.reward.terms) == 3

    # Verify listing all versions
    all_versions = store.list_versions(task_id)
    assert len(all_versions) == 3
    assert [v.version for v in all_versions] == [1, 2, 3]

    # Current active version before rollback is v3
    current_before = store.get_current_version(task_id)
    assert current_before.version == 3

    # Perform Rollback from v3 to v1
    restored = store.rollback(task_id=task_id, target_version=1)

    # Restored state must match v1
    assert restored.version == 4
    assert "Rollback to v1" in restored.applied_critique
    assert len(restored.reward_dsl.reward.terms) == len(v1.reward_dsl.reward.terms)
    assert restored.reward_dsl.reward.terms[0].name == v1.reward_dsl.reward.terms[0].name
    assert restored.reward_dsl.reward.terms[0].value == v1.reward_dsl.reward.terms[0].value
    assert restored.locked_episode_ids == v1.locked_episode_ids
    assert Path(restored.policy_checkpoint_path).read_text() == "weights_v1"

    # Current active version after rollback is the restored version
    current_after = store.get_current_version(task_id)
    assert current_after.version == 4
    assert Path(current_after.policy_checkpoint_path).read_text() == "weights_v1"

    # Verify database persistence survives a fresh TaskStateStore instance
    reopened_store = TaskStateStore(base_dir=tmp_path / "runs")
    persisted_current = reopened_store.get_current_version(task_id)
    assert persisted_current is not None
    assert persisted_current.version == 4
    assert persisted_current.reward_dsl.reward.terms[0].name == "accurate_match"


def test_byok_llm_factory_fallback():
    """Verify LLM client factory cleanly falls back to MockRulesLLMClient when no keys provided."""
    mock_client = create_llm_client(LLMConfig(provider="mock"))
    assert isinstance(mock_client, MockRulesLLMClient)

    # OpenAI without API key falls back to mock
    openai_no_key = create_llm_client(LLMConfig(provider="openai", api_key=None))
    assert isinstance(openai_no_key, MockRulesLLMClient)

    # Anthropic without API key falls back to mock
    anthropic_no_key = create_llm_client(LLMConfig(provider="anthropic", api_key=""))
    assert isinstance(anthropic_no_key, MockRulesLLMClient)
