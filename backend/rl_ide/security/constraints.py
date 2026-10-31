"""Declarative hard constraints for tabular decision agents.

A user can say "never auto-approve an invoice whose vendor_risk_score is 85 or higher". That is
not a preference to be learned statistically; it is a rule that must hold for every decision.
This module turns the common phrasings into a small, auditable structure that the environment,
the scorecard and the exported runtime can all execute identically.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

_COMPARATORS = (">=", "<=", "==", "!=", ">", "<")

_CUE = re.compile(
    r"\b(?:never|must\s+not|mustn't|should\s+never|do\s+not|don't)\b",
    re.IGNORECASE,
)
_REQUIRE = re.compile(
    r"\b(?:must\s+be|must\s+route\s+to|must\s+go\s+to|should\s+be)\s+[\"']?"
    r"(?P<action>[A-Za-z][\w /&-]{1,60})",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?P<num>-?\d+(?:\.\d+)?)")
_FEATURE_TAIL = re.compile(
    r"(?P<feature>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?:is|>=|>|above|over|at\s+least|exceeds|greater\s+than|more\s+than)\s+"
    r"(?P<num>-?\d+(?:\.\d+)?)\s*(?:or\s+(?:higher|more|above))?",
    re.IGNORECASE,
)
_LESS_THAN = re.compile(
    r"(?P<feature>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?:is|<=|<|below|under|at\s+most|less\s+than)\s+"
    r"(?P<num>-?\d+(?:\.\d+)?)\s*(?:or\s+(?:lower|less|below))?",
    re.IGNORECASE,
)


class HardConstraint(BaseModel):
    """A rule that forbids (or requires) an action when a record crosses a threshold."""

    feature: str
    op: str = ">="
    value: float | str
    forbidden_actions: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    reason: str = ""

    def matches(self, record: dict[str, Any]) -> bool:
        raw = record.get(self.feature)
        if raw is None:
            return False
        try:
            left = float(raw)
            right = float(self.value)
        except (TypeError, ValueError):
            return str(raw).strip().lower() == str(self.value).strip().lower()
        if self.op == ">=":
            return left >= right
        if self.op == ">":
            return left > right
        if self.op == "<=":
            return left <= right
        if self.op == "<":
            return left < right
        if self.op == "==":
            return left == right
        if self.op == "!=":
            return left != right
        return False


def _normalise_action(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _match_action(text: str, action_labels: Iterable[str]) -> str | None:
    """Returns the action label that appears in *text*, longest label first."""
    lowered = _normalise_action(text)
    for label in sorted(action_labels, key=len, reverse=True):
        if _normalise_action(label) and _normalise_action(label) in lowered:
            return label
    return None


def _feature_in(text: str, features: Iterable[str]) -> str | None:
    for feature in sorted(features, key=len, reverse=True):
        if re.search(rf"\b{re.escape(str(feature))}\b", text, re.IGNORECASE):
            return str(feature)
    return None


def extract_hard_constraints(
    prompt: str,
    action_labels: Iterable[str],
    feature_names: Iterable[str] = (),
) -> list[HardConstraint]:
    """Extracts simple, checkable rules from an English objective.

    Supports the phrasings operators actually use:
      * "must never Auto-Approve an invoice whose vendor_risk_score is 85 or higher"
      * "never block when amount is below 10"
      * "those must be Block" / "route them to Block"
    Anything more elaborate is ignored rather than guessed; the blueprint can still carry an
    explicit constraint supplied by another layer.
    """
    text = (prompt or "").strip()
    if not text:
        return []

    labels = [str(a) for a in action_labels if str(a).strip()]
    features = [str(f) for f in feature_names if str(f).strip()]
    results: list[HardConstraint] = []

    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
        cue = _CUE.search(sentence)
        if not cue:
            continue

        action = _match_action(sentence[cue.end():], labels)
        if action is None:
            continue

        window = sentence[max(0, cue.start() - 40): cue.end() + 160]
        feature = _feature_in(window, features) if features else None
        if feature is None:
            match = _FEATURE_TAIL.search(window) or _LESS_THAN.search(window)
            feature = match.group("feature") if match else None
        if feature is None:
            continue

        greater = _FEATURE_TAIL.search(window)
        lesser = _LESS_THAN.search(window)
        match = greater or lesser
        if match is None:
            continue
        op = ">=" if greater is not None else "<="
        value = float(match.group("num"))

        required: list[str] = []
        require = _REQUIRE.search(sentence)
        if require:
            candidate = _match_action(require.group("action"), labels)
            if candidate and candidate != action:
                required.append(candidate)

        results.append(
            HardConstraint(
                feature=feature,
                op=op,
                value=value,
                forbidden_actions=[action],
                required_actions=required,
                reason=sentence.strip()[:240],
            )
        )

    # De-duplicate identical rules while preserving order.
    unique: list[HardConstraint] = []
    for item in results:
        key = (item.feature, item.op, item.value, tuple(item.forbidden_actions))
        if all((u.feature, u.op, u.value, tuple(u.forbidden_actions)) != key for u in unique):
            unique.append(item)
    return unique


def action_mask(
    record: dict[str, Any],
    action_labels: list[str],
    constraints: Iterable[HardConstraint | dict[str, Any]],
) -> list[bool]:
    """Returns a boolean mask aligned with *action_labels* for one record."""
    mask = [True] * len(action_labels)
    label_to_index = {label: i for i, label in enumerate(action_labels)}
    for constraint in constraints:
        rule = constraint if isinstance(constraint, HardConstraint) else HardConstraint(**constraint)
        if not rule.matches(record):
            continue
        for label in rule.forbidden_actions:
            idx = label_to_index.get(label)
            if idx is not None:
                mask[idx] = False
        if rule.required_actions:
            allowed = {label_to_index[l] for l in rule.required_actions if l in label_to_index}
            for i in range(len(mask)):
                if i not in allowed:
                    mask[i] = False

    if not any(mask):
        # A contradictory rule set must never make every action illegal; keep the safest option
        # (the last action is conventionally the most restrictive) instead of crashing the run.
        mask[-1] = True
    return mask


def first_allowed_action(mask: list[bool]) -> int:
    for index, allowed in enumerate(mask):
        if allowed:
            return index
    return 0
