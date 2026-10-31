"""Passive harvesting of frontier-model interaction logs.

The expensive part of adopting a local policy is producing labelled examples. Teams that already
run a frontier model in production have them: every accept/reject/edit is a human judgement about
one decision. This module recognises that shape, turns it into supervised rows, and registers the
rejections as behavioural assertions instead of discarding them.

Conservative by design: a rejection only becomes a hard corpus assertion when the human also
supplied the action they wanted. A bare "rejected" is a soft negative - it tells us the frontier
action was wrong without claiming every other action is right - and is reported, not enforced.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from pydantic import BaseModel, Field

PROPOSED_COLUMNS = (
    "llm_action",
    "model_action",
    "predicted_action",
    "proposed_action",
    "agent_action",
    "candidate_action",
)
FEEDBACK_COLUMNS = ("accepted", "is_accepted", "feedback", "verdict", "human_verdict", "status")
CORRECTED_COLUMNS = ("corrected_action", "human_action", "edited_action", "final_action")
RATIONALE_COLUMNS = ("critique_text", "critique", "comment", "review_note", "notes")

ACCEPT_VALUES = {"accepted", "approve", "approved", "true", "1", "yes", "y", "kept", "correct"}
REJECT_VALUES = {"rejected", "reject", "false", "0", "no", "n", "edited", "overridden", "wrong"}
LOG_ONLY_COLUMNS = PROPOSED_COLUMNS + FEEDBACK_COLUMNS + CORRECTED_COLUMNS + RATIONALE_COLUMNS

DEFAULT_MIN_USABLE_ROWS = 100


class InteractionLogDetection(BaseModel):
    """What the profiler found, and how confident the shape is."""

    detected: bool = False
    proposed_column: str | None = None
    feedback_column: str | None = None
    corrected_column: str | None = None
    rationale_column: str | None = None
    total_rows: int = 0
    usable_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    corrected_rows: int = 0
    feature_columns: list[str] = Field(default_factory=list)
    sample_size_warning: str | None = None
    notes: list[str] = Field(default_factory=list)


def _find(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lowered = {str(c).strip().lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _normalise(value: Any) -> str:
    return str(value).strip().lower()


def detect_interaction_log(
    df: pd.DataFrame, *, min_usable_rows: int = DEFAULT_MIN_USABLE_ROWS
) -> InteractionLogDetection:
    """Detects a frontier interaction log by an action column plus a human verdict column."""
    detection = InteractionLogDetection(total_rows=0 if df is None else len(df))
    if df is None or df.empty:
        return detection

    proposed = _find(df, PROPOSED_COLUMNS)
    feedback = _find(df, FEEDBACK_COLUMNS)
    if proposed is None or feedback is None or proposed == feedback:
        return detection

    detected = InteractionLogDetection(
        detected=True,
        proposed_column=proposed,
        feedback_column=feedback,
        corrected_column=_find(df, CORRECTED_COLUMNS),
        rationale_column=_find(df, RATIONALE_COLUMNS),
        total_rows=len(df),
    )

    accepted = rejected = corrected = 0
    for _, row in df.iterrows():
        verdict = _normalise(row.get(feedback))
        proposal = row.get(proposed)
        if pd.isna(proposal):
            continue
        if verdict in ACCEPT_VALUES:
            accepted += 1
        elif verdict in REJECT_VALUES:
            rejected += 1
            if detected.corrected_column and not pd.isna(row.get(detected.corrected_column)):
                corrected += 1

    detected.accepted_rows = accepted
    detected.rejected_rows = rejected
    detected.corrected_rows = corrected
    detected.usable_rows = accepted + corrected
    detected.feature_columns = [
        str(c) for c in df.columns if str(c) not in LOG_ONLY_COLUMNS
    ]

    if detected.usable_rows < min_usable_rows:
        detected.sample_size_warning = (
            f"Only {detected.usable_rows} labelled decisions found; at least {min_usable_rows} "
            "are recommended before the held-out comparison carries statistical weight."
        )
    if rejected and not corrected:
        detected.notes.append(
            f"{rejected} rejection(s) carry no corrected action, so they are recorded as soft "
            "negatives rather than hard forbidden-action assertions."
        )
    return detected


def supervised_rows(
    df: pd.DataFrame, detection: InteractionLogDetection
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows the policy can learn from: accepted proposals and human corrections.

    Returns ``(records, target_key)`` where each record carries the feature columns plus a
    ``target_action`` labelled with the human's choice.
    """
    if not detection.detected or df is None:
        return [], "target_action"
    records: list[dict[str, Any]] = []
    features = detection.feature_columns or [
        str(c) for c in df.columns if str(c) != (detection.feedback_column or "")
    ]
    for _, row in df.iterrows():
        verdict = _normalise(row.get(detection.feedback_column))
        target = None
        if verdict in ACCEPT_VALUES:
            target = row.get(detection.proposed_column)
        elif verdict in REJECT_VALUES and detection.corrected_column:
            target = row.get(detection.corrected_column)
        if target is None or pd.isna(target):
            continue
        record = {col: row.get(col) for col in features}
        record["target_action"] = target
        records.append(record)
    return records, "target_action"


def negative_rows(
    df: pd.DataFrame, detection: InteractionLogDetection
) -> list[dict[str, Any]]:
    """Rejected proposals, with the corrected action when the human supplied one."""
    if not detection.detected or df is None:
        return []
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        verdict = _normalise(row.get(detection.feedback_column))
        if verdict not in REJECT_VALUES:
            continue
        corrected = (
            row.get(detection.corrected_column) if detection.corrected_column else None
        )
        out.append(
            {
                "proposed": row.get(detection.proposed_column),
                "corrected": None if corrected is None or pd.isna(corrected) else corrected,
                "rationale": (
                    str(row.get(detection.rationale_column))
                    if detection.rationale_column and not pd.isna(row.get(detection.rationale_column))
                    else "frontier action rejected by a human reviewer"
                ),
                "features": {col: row.get(col) for col in detection.feature_columns},
            }
        )
    return out


def _record_key(values: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{k}={str(values.get(k)).strip().lower()}" for k in sorted(values))


def harvest_into_corpus(
    corpus: Any,
    session_id: str,
    df: pd.DataFrame,
    detection: InteractionLogDetection,
    *,
    env: Any,
    label_to_action: dict[str, int] | None = None,
    max_cases: int = 50,
) -> dict[str, Any]:
    """Registers rejected frontier actions as corpus assertions.

    A case is only created when the rejected row can be matched to a record the environment can
    actually encode, because an assertion about an observation the policy never sees would be a
    fabricated test. Rejections without a corrected action are recorded as soft negatives and
    reported separately.
    """
    result = {
        "registered": 0,
        "soft_negatives": 0,
        "unmatched_rows": 0,
        "skipped_no_env": env is None,
    }
    if not detection.detected or env is None or not hasattr(env, "records"):
        return result

    labels = {str(k).strip().lower(): int(v) for k, v in (label_to_action or {}).items()}
    records_index: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in getattr(env, "records", []) or []:
        records_index[_record_key({k: record.get(k) for k in detection.feature_columns})] = record

    for row in negative_rows(df, detection):
        if result["registered"] >= max_cases:
            break
        features = row.get("features") or {}
        record = records_index.get(_record_key(features))
        if record is None:
            result["unmatched_rows"] += 1
            continue
        try:
            obs = env._extract_obs(record)  # noqa: SLF001 - the env's own encoder
        except Exception:
            result["unmatched_rows"] += 1
            continue

        proposed = labels.get(str(row.get("proposed")).strip().lower())
        corrected = labels.get(str(row.get("corrected")).strip().lower()) if row.get("corrected") else None
        if proposed is None:
            result["unmatched_rows"] += 1
            continue

        if corrected is not None:
            case = corpus.add_case(
                session_id,
                [float(x) for x in obs],
                expected_actions=[corrected],
                forbidden_actions=[proposed],
                source="passive_log",
                rationale=f"Frontier proposed {row.get('proposed')}; a human chose {row.get('corrected')}.",
                action_labels=list(labels.keys()),
            )
        else:
            case = corpus.add_case(
                session_id,
                [float(x) for x in obs],
                forbidden_actions=[proposed],
                source="passive_log",
                rationale=f"Frontier proposed {row.get('proposed')}; a human rejected it without a correction.",
                action_labels=list(labels.keys()),
            )
            if case is not None:
                result["soft_negatives"] += 1
        if case is not None:
            result["registered"] += 1
    return result
