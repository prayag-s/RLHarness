"""Deterministic fact extraction for the Journey Coach.

Every number the coach prints is produced here, before any model is called. The judge receives
this dict and nothing else, which is what makes "the LLM never invents a number" enforceable
rather than aspirational.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

_METRIC_CUE = re.compile(
    r"(\d+\s*%|success|accuracy|precision|recall|f1|auc|agreement|win|loss|draw|"
    r"score|return|reward|latency|cost|within|error|false\s+positive|throughput|rate)",
    re.IGNORECASE,
)
_PROHIBITION_CUE = re.compile(
    r"\b(?:never|must\s+not|mustn't|should\s+never|do\s+not|don't)\b",
    re.IGNORECASE,
)
_MULTI_JOB_CUE = re.compile(
    r"\b(and\s+also|as\s+well\s+as|plus\s+also|then\s+also)\b", re.IGNORECASE
)
_SUPERVISED_METRIC = re.compile(
    r"(accuracy|agreement|precision|recall|f1|auc|classification|match|correct)",
    re.IGNORECASE,
)
_PENALTY_WORD = re.compile(
    r"(penal|forbid|never|avoid|cost|risk|violation|incorrect|false)", re.IGNORECASE
)


def _as_dict(blueprint: Any) -> dict[str, Any]:
    if blueprint is None:
        return {}
    if isinstance(blueprint, dict):
        return blueprint
    dump = getattr(blueprint, "model_dump", None)
    return dump() if callable(dump) else {}


def wilson_half_width_pct(episodes: int, p: float = 0.85, z: float = 1.96) -> float | None:
    """Half-width of the 95% interval for a proportion, in percentage points."""
    if episodes <= 0:
        return None
    return round(z * math.sqrt(max(0.0, p * (1.0 - p)) / episodes) * 100.0, 2)


def episodes_for_half_width(half_width_pct: float, p: float = 0.85, z: float = 1.96) -> int:
    """Episodes needed to shrink the interval to ``half_width_pct`` points."""
    if half_width_pct <= 0:
        return 1
    n = (z * z) * p * (1.0 - p) / ((half_width_pct / 100.0) ** 2)
    return max(1, int(math.ceil(n)))


@dataclass
class ObjectiveFacts:
    prompt: str
    word_count: int
    action_labels: list[str] = field(default_factory=list)
    observation_features: list[str] = field(default_factory=list)
    hard_constraint_count: int = 0
    has_prohibition_cue: bool = False
    has_metric_cue: bool = False
    multiple_jobs: bool = False
    blueprint_actions: list[str] = field(default_factory=list)
    reward_terms: list[str] = field(default_factory=list)
    reward_has_penalty: bool = False
    success_metric_name: str = ""
    supervised_metric: bool = False
    data_source_type: str = ""
    dataset_has_target: bool = False
    dataset_row_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetFacts:
    name: str = ""
    row_count: int = 0
    column_count: int = 0
    target_column: str | None = None
    target_distribution: dict[str, int] = field(default_factory=dict)
    minority_ratio: float | None = None
    missing_columns: list[dict[str, Any]] = field(default_factory=list)
    max_missing_ratio: float = 0.0
    duplicate_rows: int = 0
    duplicate_ratio: float = 0.0
    id_like_columns: list[str] = field(default_factory=list)
    separation_columns: list[str] = field(default_factory=list)
    numeric_feature_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParameterFacts:
    rounds: int = 0
    episodes_per_round: int = 0
    target: float = 0.0
    metric_name: str = ""
    action_count: int = 0
    baseline_metric: float | None = None
    uniform_chance: float | None = None
    projected_episodes: int = 0
    ci_half_width_pct: float | None = None
    reward_terms: list[str] = field(default_factory=list)
    reward_has_penalty: bool = False
    prompt_has_prohibition: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreviewFacts:
    html_length: int = 0
    has_live_update: bool = False
    has_metric_signal: bool = False
    has_controls: bool = False
    metric_name: str = ""
    arena_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResultFacts:
    has_scorecard: bool = False
    held_out: bool = False
    task_class: str = ""
    metric_name: str = ""
    policy_value: float | None = None
    episodes: int = 0
    ci: list[float] | None = None
    best_baseline: float | None = None
    uplift: float | None = None
    illegal_actions_pct: float = 0.0
    corpus_regressions: list[str] = field(default_factory=list)
    corpus_pass_rate_pct: float | None = None
    crossfire_verdict: str = ""
    crossfire_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    target: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportFacts:
    has_scorecard: bool = False
    crossfire_closed: bool = False
    deployment_agreement_pct: float | None = None
    onnx_parity_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def objective_facts(
    prompt: str,
    blueprint: Any = None,
    dataset: Any = None,
) -> ObjectiveFacts:
    # Imported lazily so the coach package never participates in an import cycle at load time.
    from rl_ide.architect.intake import (
        extract_action_labels_from_prompt,
        extract_observation_features_from_prompt,
    )
    from rl_ide.security.constraints import extract_hard_constraints

    bp = _as_dict(blueprint)
    explicit_actions = extract_action_labels_from_prompt(prompt) or []
    bp_actions = list(bp.get("action_labels") or [])
    if not bp_actions:
        bp_actions = [
            str(a.get("label", "")) for a in (bp.get("actions") or []) if isinstance(a, dict)
        ]
    bp_actions = [a for a in bp_actions if a]

    constraints = extract_hard_constraints(prompt, explicit_actions or bp_actions) or []

    reward_terms: list[str] = []
    reward_has_penalty = False
    for term in bp.get("initial_reward_components") or []:
        if not isinstance(term, dict):
            continue
        name = str(term.get("name") or "")
        description = str(term.get("description") or "")
        reward_terms.append(name or description)
        try:
            weight = float(term.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if weight < 0 or _PENALTY_WORD.search(f"{name} {description}"):
            reward_has_penalty = True

    profile = getattr(dataset, "profile", None)
    target = getattr(profile, "target_candidate", None) if profile is not None else None
    row_count = None
    df = getattr(dataset, "df", None)
    if df is not None:
        try:
            row_count = int(len(df))
        except Exception:
            row_count = None

    metric_name = str(bp.get("success_metric_name") or "")
    return ObjectiveFacts(
        prompt=prompt or "",
        word_count=len((prompt or "").split()),
        action_labels=explicit_actions,
        observation_features=extract_observation_features_from_prompt(prompt) or [],
        hard_constraint_count=len(constraints),
        has_prohibition_cue=bool(_PROHIBITION_CUE.search(prompt or "")),
        has_metric_cue=bool(_METRIC_CUE.search(prompt or "")),
        multiple_jobs=bool(_MULTI_JOB_CUE.search(prompt or "")),
        blueprint_actions=bp_actions,
        reward_terms=reward_terms,
        reward_has_penalty=reward_has_penalty,
        success_metric_name=metric_name,
        supervised_metric=bool(_SUPERVISED_METRIC.search(metric_name)),
        data_source_type=str(bp.get("data_source_type") or ""),
        dataset_has_target=bool(target),
        dataset_row_count=row_count,
    )


def dataset_facts(dataset: Any) -> DatasetFacts:
    df = getattr(dataset, "df", None)
    facts = DatasetFacts(name=str(getattr(dataset, "name", "")))
    if df is None:
        return facts

    profile = getattr(dataset, "profile", None)
    row_count = int(len(df))
    facts.row_count = row_count
    facts.column_count = int(len(df.columns))
    target = getattr(profile, "target_candidate", None) if profile is not None else None
    if target is None:
        metadata = getattr(dataset, "metadata", None) or {}
        target = (
            metadata.get("label_column")
            or metadata.get("target_column")
            or getattr(dataset, "label_column", None)
        )
    if target is not None and str(target) in df.columns:
        facts.target_column = str(target)

    if facts.target_column:
        try:
            counts = df[facts.target_column].value_counts(dropna=True)
            facts.target_distribution = {
                str(k): int(v) for k, v in counts.head(20).items()
            }
            total = int(counts.sum())
            if total and len(counts) >= 2:
                facts.minority_ratio = round(float(counts.min()) / total, 4)
        except Exception:
            pass

    try:
        missing = []
        for column in df.columns:
            ratio = float(df[column].isna().mean())
            if ratio > 0:
                missing.append({"column": str(column), "ratio": round(ratio, 4)})
        missing.sort(key=lambda item: item["ratio"], reverse=True)
        facts.missing_columns = missing[:8]
        facts.max_missing_ratio = missing[0]["ratio"] if missing else 0.0
    except Exception:
        pass

    try:
        duplicates = int(df.duplicated().sum())
        facts.duplicate_rows = duplicates
        facts.duplicate_ratio = round(duplicates / row_count, 4) if row_count else 0.0
    except Exception:
        pass

    # An ID-like feature is a row identifier, not a signal: including it lets the model memorise
    # the training split and look excellent in-sample.
    for column in df.columns:
        if facts.target_column and str(column) == facts.target_column:
            continue
        try:
            unique = int(df[column].nunique(dropna=True))
        except Exception:
            continue
        if row_count and unique == row_count and re.search(
            r"(^|_)(id|uuid|guid|key|code|number|no)($|_)", str(column).lower()
        ):
            facts.id_like_columns.append(str(column))

    numeric_columns = [
        c
        for c in df.columns
        if str(c) != facts.target_column and df[c].dtype.kind in "biuf"
    ]
    facts.numeric_feature_count = len(numeric_columns)

    # A feature that splits the two classes perfectly is almost always the label in disguise.
    if facts.target_column and len(facts.target_distribution) == 2:
        labels = list(facts.target_distribution.keys())
        for column in numeric_columns[:25]:
            try:
                grouped = df[[column, facts.target_column]].dropna()
                if grouped.empty:
                    continue
                left = grouped[grouped[facts.target_column].astype(str) == labels[0]][column]
                right = grouped[grouped[facts.target_column].astype(str) == labels[1]][column]
                if left.empty or right.empty:
                    continue
                if left.max() < right.min() or right.max() < left.min():
                    facts.separation_columns.append(str(column))
            except Exception:
                continue

    return facts


def _majority_baseline(dataset: Any, facts: DatasetFacts) -> float | None:
    if not facts.target_column or not facts.target_distribution:
        return None
    total = sum(facts.target_distribution.values())
    if not total:
        return None
    return round(max(facts.target_distribution.values()) / total * 100.0, 2)


def parameter_facts(
    blueprint: Any,
    params: dict[str, Any] | None = None,
    dataset: Any = None,
    prompt: str = "",
) -> ParameterFacts:
    bp = _as_dict(blueprint)
    params = params or {}

    def _pick(key: str, fallback: Any) -> Any:
        # An explicit zero is a budget of zero, not "unset". Using ``or`` here would silently
        # replace a zero the user asked for with the blueprint default.
        value = params.get(key)
        return fallback if value is None else value

    rounds = int(_pick("max_auto_tune_rounds", bp.get("max_auto_tune_rounds") or 0))
    episodes = int(_pick("episodes_per_round", bp.get("episodes_per_round") or 0))
    target = float(_pick("target_metric_threshold", bp.get("target_metric_threshold") or 0.0))
    blueprints_actions = list(bp.get("action_labels") or [])
    if not blueprints_actions:
        blueprints_actions = [
            str(a.get("label", "")) for a in (bp.get("actions") or []) if isinstance(a, dict)
        ]
    action_count = len([a for a in blueprints_actions if a])

    reward_terms: list[str] = []
    reward_has_penalty = False
    for term in bp.get("initial_reward_components") or []:
        if not isinstance(term, dict):
            continue
        name = str(term.get("name") or "")
        description = str(term.get("description") or "")
        reward_terms.append(name or description)
        try:
            weight = float(term.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if weight < 0 or _PENALTY_WORD.search(f"{name} {description}"):
            reward_has_penalty = True

    ds_facts = dataset_facts(dataset) if dataset is not None else None
    return ParameterFacts(
        rounds=rounds,
        episodes_per_round=episodes,
        target=round(target, 2),
        metric_name=str(bp.get("success_metric_name") or ""),
        action_count=action_count,
        baseline_metric=_majority_baseline(dataset, ds_facts) if ds_facts else None,
        uniform_chance=round(100.0 / action_count, 2) if action_count else None,
        projected_episodes=max(0, rounds) * max(0, episodes),
        ci_half_width_pct=wilson_half_width_pct(episodes),
        reward_terms=reward_terms,
        reward_has_penalty=reward_has_penalty,
        prompt_has_prohibition=bool(_PROHIBITION_CUE.search(prompt or "")),
    )


def preview_facts(blueprint: Any, arena_html: str | None, env_code: str | None = None) -> PreviewFacts:
    bp = _as_dict(blueprint)
    html = arena_html or ""
    metric_name = str(bp.get("success_metric_name") or "")
    metric_tokens = [
        token for token in re.split(r"[^a-zA-Z%]+", metric_name.lower()) if len(token) > 3
    ]
    lowered = html.lower()
    return PreviewFacts(
        html_length=len(html),
        has_live_update=any(
            marker in lowered
            for marker in ("requestanimationframe", "setinterval", "postmessage", "addeventlistener")
        ),
        has_metric_signal=bool(
            "%" in html
            or any(token in lowered for token in metric_tokens)
            or any(word in lowered for word in ("success", "score", "metric", "win rate", "accuracy"))
        ),
        has_controls=any(marker in lowered for marker in ("<button", "onclick", "addeventlistener")),
        metric_name=metric_name,
        arena_description=str(bp.get("arena_visual_description") or bp.get("visual_metaphor") or "")[:400],
    )


def result_facts(
    scorecard: dict[str, Any] | None,
    conversation: dict[str, Any] | None = None,
) -> ResultFacts:
    facts = ResultFacts(has_scorecard=bool(scorecard))
    conv = conversation or {}
    rounds = list(conv.get("rounds_history") or [])
    params = conv.get("training_params") or {}
    if params.get("target_metric_threshold") is not None:
        facts.target = float(params["target_metric_threshold"])
    facts.rounds = [
        {
            "round": r.get("round_num", r.get("round", idx + 1)),
            "metric": r.get("metric"),
            "target": r.get("target_threshold", r.get("target")),
            "status": r.get("status", ""),
        }
        for idx, r in enumerate(rounds)
        if isinstance(r, dict)
    ]
    if facts.target is None and facts.rounds:
        last_target = facts.rounds[-1].get("target")
        if isinstance(last_target, (int, float)):
            facts.target = float(last_target)
    if not scorecard:
        return facts

    policy = scorecard.get("policy") or {}
    baselines = scorecard.get("baselines") or []
    facts.held_out = bool(scorecard.get("held_out"))
    facts.task_class = str(scorecard.get("task_class") or "")
    facts.metric_name = str(scorecard.get("metric_name") or "")
    facts.policy_value = policy.get("value_pct")
    facts.episodes = int(policy.get("episodes") or 0)
    ci = policy.get("ci95_pct")
    facts.ci = [float(ci[0]), float(ci[1])] if ci and len(ci) == 2 else None
    values = [
        float(b.get("value_pct"))
        for b in baselines
        if isinstance(b, dict) and b.get("value_pct") is not None
    ]
    facts.best_baseline = max(values) if values else None
    facts.uplift = scorecard.get("uplift_vs_best_baseline_pct")
    facts.illegal_actions_pct = float(policy.get("illegal_actions_pct") or 0.0)
    facts.notes = list(scorecard.get("notes") or [])

    corpus = scorecard.get("eval_corpus") or {}
    facts.corpus_regressions = list(corpus.get("regressed_case_ids") or [])
    facts.corpus_pass_rate_pct = corpus.get("pass_rate_pct")

    crossfire = scorecard.get("crossfire") or {}
    facts.crossfire_verdict = str(crossfire.get("verdict") or "")
    facts.crossfire_reasons = list(crossfire.get("reasons") or [])
    return facts


def export_facts(
    scorecard: dict[str, Any] | None,
    deployment: dict[str, Any] | None = None,
) -> ExportFacts:
    scorecard = scorecard or {}
    deployment = deployment or {}
    crossfire = scorecard.get("crossfire") or {}
    return ExportFacts(
        has_scorecard=bool(scorecard),
        crossfire_closed=bool(crossfire.get("closed")),
        deployment_agreement_pct=deployment.get("action_agreement_pct"),
        onnx_parity_diff=deployment.get("onnx_numeric_parity_max_diff"),
    )
