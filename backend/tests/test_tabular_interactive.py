"""Regression tests for human interaction with tabular environments.

The tabular template used to treat ``payload["user_action"]`` as a record index and ignore
``action_index`` entirely. A human click therefore ran the policy instead of the person's choice,
and the response carried no decision or reward, which made the Play tab look dead.
"""

from __future__ import annotations

import pandas as pd

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.ingest import load_bytes
from rl_ide.synthesizer.env_builder import (
    JITEnvironmentSynthesizer,
    instantiate_env,
    load_synthesized_env_class,
)

ACTIONS = ["Auto-Approve", "Manager Review", "Finance Review", "Block"]


def _frame(rows: int = 40) -> pd.DataFrame:
    records = []
    for i in range(rows):
        risk = 10 + (i * 7) % 90
        records.append(
            {
                "amount": 500.0 + i * 125.0,
                "vendor_risk_score": risk,
                "department": ["IT", "Sales", "Legal"][i % 3],
                "decision": ACTIONS[3 if risk >= 85 else i % 3],
            }
        )
    return pd.DataFrame(records)


def _blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Invoice Approval",
        domain_summary="Review one invoice and choose a decision",
        data_source_type="uploaded_dataset",
        observation_features=["amount", "vendor_risk_score"],
        action_labels=ACTIONS,
        arena_visual_description="invoice desk",
        interactive_mode_description="review an invoice",
        success_metric_name="Agreement (%)",
        target_metric_threshold=90.0,
        label_column="decision",
        hard_constraints=[
            {
                "feature": "vendor_risk_score",
                "op": ">=",
                "value": 85,
                "forbidden_actions": ["Auto-Approve"],
                "required_actions": ["Block"],
                "reason": "High-risk invoices must be blocked.",
            }
        ],
    )


def _build_env(tmp_path):
    frame = _frame()
    dataset = load_bytes(
        filename="invoices.csv", content=frame.to_csv(index=False).encode("utf-8")
    )
    builder = JITEnvironmentSynthesizer.__new__(JITEnvironmentSynthesizer)
    builder._llm_client = None
    code, _ = builder._build_dataset_code(_blueprint(), dataset)
    env_file = tmp_path / "env_module.py"
    env_file.write_text(code, encoding="utf-8")
    env_cls = load_synthesized_env_class(env_file)
    return instantiate_env(env_cls, dataset_records=frame.to_dict(orient="records"))


def _policy(action: int):
    def predict(_obs):
        probs = [0.0] * len(ACTIONS)
        probs[action] = 1.0
        return action, probs

    return predict


def test_human_action_wins_over_policy(tmp_path):
    env = _build_env(tmp_path)
    truth = env._truth_for(env.records[0])
    human = (truth + 1) % len(ACTIONS)

    result = env.interactive_step(
        {"action_index": human, "record_index": 0},
        _policy(truth),  # the policy would choose the truth; the human must still win
    )

    assert result["agent_action_label"] == ACTIONS[human]
    assert result["arena_state"]["chosen_decision"] == ACTIONS[human]
    assert result["arena_state"]["reward"] is not None
    assert result["reward"] is not None
    assert result["arena_state"]["record_index"] == 0
    assert result["arena_state"]["step_count"] == 1
    assert result["is_success"] is False


def test_policy_acts_only_when_no_human_action_is_supplied(tmp_path):
    env = _build_env(tmp_path)
    truth = env._truth_for(env.records[2])

    result = env.interactive_step({"record_index": 2}, _policy(truth))

    assert result["agent_action_label"] == ACTIONS[truth]
    assert result["arena_state"]["chosen_decision"] == ACTIONS[truth]
    assert result["is_success"] is True


def test_hard_constraint_overrides_a_forbidden_human_action(tmp_path):
    env = _build_env(tmp_path)
    high_risk_index = next(
        i for i, r in enumerate(env.records) if float(r["vendor_risk_score"]) >= 85
    )

    result = env.interactive_step(
        {"action_index": 0, "record_index": high_risk_index},  # Auto-Approve
        _policy(0),
    )

    assert result["agent_action_label"] == "Block"
    assert result["arena_state"]["chosen_decision"] == "Block"
    assert result["arena_state"]["safety_override"] is True
    assert result["arena_state"]["action_mask"] == [False, False, False, True]


def test_hard_constraint_is_enforced_in_training_steps(tmp_path):
    env = _build_env(tmp_path)
    high_risk_index = next(
        i for i, r in enumerate(env.records) if float(r["vendor_risk_score"]) >= 85
    )
    env.current_idx = high_risk_index

    _, reward, _, _, info = env.step(0)  # Auto-Approve is forbidden here

    assert info["safety_override"] is True
    assert env.last_action == ACTIONS.index("Block")
    assert reward == 1.0


def test_reset_is_explicit(tmp_path):
    env = _build_env(tmp_path)
    result = env.interactive_step({"reset": True})
    assert result["agent_action_label"] == "Reset"
    assert result["arena_state"]["step_count"] == 0
