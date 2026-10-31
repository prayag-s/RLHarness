"""The exported bundle must apply the same hard constraints as the environment.

The scorecard is not a licence: if a bundle ships a policy that would auto-approve a high-risk
invoice, the supported runtime must either mask it or refuse to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_ide.export.exporter import STANDALONE_RUN_LOCAL_TEMPLATE

ACTIONS = ["Auto-Approve", "Manager Review", "Finance Review", "Block"]
RULES = [
    {
        "feature": "vendor_risk_score",
        "op": ">=",
        "value": 85,
        "forbidden_actions": ["Auto-Approve"],
        "required_actions": ["Block"],
        "reason": "High-risk invoices must be blocked.",
    }
]


def _runtime_namespace() -> dict:
    namespace: dict = {"__name__": "bundle_runtime"}
    exec(compile(STANDALONE_RUN_LOCAL_TEMPLATE, "run_local.py", "exec"), namespace)
    return namespace


class _FakeSession:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32)

    def get_inputs(self):
        return [type("Input", (), {"shape": [None, self._probs.shape[-1]]})()]

    def run(self, _outputs, _feeds):
        return [self._probs]


def test_mask_forces_the_required_action_on_a_high_risk_record():
    ns = _runtime_namespace()
    probs = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)

    masked, mask = ns["apply_constraint_mask"](probs, {"vendor_risk_score": 88}, RULES, ACTIONS)

    assert mask == [False, False, False, True]
    assert int(np.argmax(masked)) == ACTIONS.index("Block")


def test_mask_leaves_a_low_risk_record_untouched():
    ns = _runtime_namespace()
    probs = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)

    masked, mask = ns["apply_constraint_mask"](probs, {"vendor_risk_score": 20}, RULES, ACTIONS)

    assert mask == [True, True, True, True]
    assert int(np.argmax(masked)) == 0


def test_constrained_bundle_refuses_to_run_without_records():
    ns = _runtime_namespace()
    session = _FakeSession([[0.80, 0.10, 0.05, 0.05]])

    try:
        ns["execute_inference"](
            session, {}, ACTIONS, np.zeros((1, 4), dtype=np.float32), safety_rules=RULES
        )
    except ValueError as exc:
        assert "record" in str(exc).lower()
    else:
        raise AssertionError("a constrained bundle must not run blind")


def test_constrained_bundle_masks_when_records_are_supplied():
    ns = _runtime_namespace()
    session = _FakeSession([[0.80, 0.10, 0.05, 0.05]])

    result = ns["execute_inference"](
        session,
        {},
        ACTIONS,
        np.zeros((1, 4), dtype=np.float32),
        records=[{"vendor_risk_score": 88}],
        safety_rules=RULES,
    )

    prediction = result["predictions"][0]
    assert prediction["action_label"] == "Block"
    assert prediction["safety_override"] is True
    assert prediction["raw_action_label"] == "Auto-Approve"


def test_safety_rules_are_read_from_the_bundle(tmp_path: Path):
    (tmp_path / "safety_rules.json").write_text(json.dumps(RULES), encoding="utf-8")
    ns = _runtime_namespace()
    assert ns["load_safety_rules"](tmp_path) == RULES


def test_objective_phrasing_becomes_a_constraint():
    from rl_ide.security.constraints import extract_hard_constraints

    prompt = (
        "It must never Auto-Approve an invoice whose vendor_risk_score is 85 or higher "
        "-- those must be Block."
    )
    rules = extract_hard_constraints(prompt, ACTIONS, ["amount", "vendor_risk_score"])

    assert len(rules) == 1
    rule = rules[0]
    assert rule.feature == "vendor_risk_score"
    assert rule.op == ">="
    assert rule.value == 85.0
    assert rule.forbidden_actions == ["Auto-Approve"]
    assert rule.required_actions == ["Block"]
