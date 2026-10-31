"""Held-out scorecard for a trained policy.

The reported number must describe the policy the customer will receive, measured on rows it never
trained on, and compared against a baseline a buyer can recognise. This module holds the pure
arithmetic (Wilson interval, baseline accuracy) plus the scorecard schema; the episode rollouts
themselves stay in :mod:`rl_ide.autopilot.loop_controller`.
"""

from __future__ import annotations

import math
from typing import Any

SCORECARD_VERSION = 1


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval for a proportion, in percent.

    Wilson rather than normal-approximation: policy success rates sit near 1.0 or 0.0, where the
    Wald interval collapses to a point and understates uncertainty.
    """
    if n <= 0:
        return None
    p = min(1.0, max(0.0, float(successes) / float(n)))
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denominator
    return (round(max(0.0, centre - margin) * 100.0, 2), round(min(1.0, centre + margin) * 100.0, 2))


def uniform_chance_pct(n_actions: int) -> float:
    """Accuracy a policy gets by guessing uniformly at random."""
    return round(100.0 / max(1, int(n_actions)), 2)


def majority_class_baseline(env: Any, eval_split: str, train_split: str = "train") -> dict[str, Any] | None:
    """Accuracy of "always predict the most frequent training label" on the evaluation split.

    This is the bar a routing policy must clear to be worth anything: if the policy cannot beat
    always guessing the biggest queue, the model has learned nothing.
    """
    getter = getattr(env, "get_demonstrations", None)
    if getter is None:
        return None
    try:
        _, train_targets = getter(num_samples=5000, split=train_split)
        _, eval_targets = getter(num_samples=5000, split=eval_split)
    except TypeError:
        return None
    except Exception:
        return None
    if len(train_targets) == 0 or len(eval_targets) == 0:
        return None

    counts: dict[int, int] = {}
    for target in train_targets.tolist():
        counts[int(target)] = counts.get(int(target), 0) + 1
    majority = max(counts, key=lambda k: counts[k])
    correct = sum(1 for t in eval_targets.tolist() if int(t) == majority)
    pct = round(100.0 * correct / len(eval_targets), 2)
    labels = list(getattr(env, "action_labels", []) or [])
    return {
        "name": "majority_class",
        "value_pct": pct,
        "detail": (
            labels[majority] if 0 <= majority < len(labels) else f"action_{majority}"
        ),
        "n": int(len(eval_targets)),
    }


def logreg_baseline(env: Any, eval_split: str, train_split: str = "train") -> dict[str, Any] | None:
    """Accuracy of a one-line logistic regression on the same derived features.

    Cheap and strong on tabular data, so it answers the only baseline question a buyer actually
    asks: "why do I need a reinforcement-learning policy instead of a classifier?"
    """
    getter = getattr(env, "get_demonstrations", None)
    if getter is None:
        return None
    try:
        import numpy as np

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return None

        train_x, train_y = getter(num_samples=5000, split=train_split)
        eval_x, eval_y = getter(num_samples=5000, split=eval_split)
        if len(train_x) == 0 or len(eval_x) == 0:
            return None
        classes = sorted(set(int(t) for t in train_y.tolist()))
        if len(classes) < 2:
            return None

        model = LogisticRegression(max_iter=1000)
        model.fit(np.asarray(train_x, dtype=np.float32), np.asarray(train_y, dtype=np.int64))
        predictions = model.predict(np.asarray(eval_x, dtype=np.float32))
        correct = int((predictions == np.asarray(eval_y, dtype=np.int64)).sum())
        return {
            "name": "logistic_regression",
            "value_pct": round(100.0 * correct / len(eval_y), 2),
            "detail": "same features, one linear model",
            "n": int(len(eval_y)),
        }
    except Exception:
        return None


def build_scorecard(
    *,
    task_title: str,
    task_class: str,
    metric_name: str,
    split_counts: dict[str, int],
    eval_split: str,
    split_fallback: bool,
    policy_metric_pct: float,
    episodes: int,
    illegal_rate_pct: float,
    baselines: list[dict[str, Any]],
    ci95_pct: tuple[float, float] | None = None,
    rows_total: int | None = None,
    embedded_rows: int | None = None,
    notes: list[str] | None = None,
    deployment: dict[str, Any] | None = None,
    economics: dict[str, Any] | None = None,
    eval_corpus: dict[str, Any] | None = None,
    crossfire: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assembles the persisted scorecard, including whether the number is genuinely held out."""
    held_out = (
        eval_split in ("test", "val")
        and bool(split_counts.get(eval_split))
        and not split_fallback
    )
    best_baseline = max((b["value_pct"] for b in baselines), default=0.0)

    scorecard: dict[str, Any] = {
        "version": SCORECARD_VERSION,
        "task_title": task_title,
        "task_class": task_class,
        "metric_name": metric_name,
        "held_out": held_out,
        "eval_split": eval_split,
        "split_counts": dict(split_counts),
        "policy": {
            "value_pct": round(float(policy_metric_pct), 2),
            "episodes": int(episodes),
            "ci95_pct": list(ci95_pct) if ci95_pct else None,
            "illegal_actions_pct": round(float(illegal_rate_pct), 2),
        },
        "baselines": baselines,
        "uplift_vs_best_baseline_pct": round(float(policy_metric_pct) - best_baseline, 2),
        "beats_all_baselines": bool(baselines) and float(policy_metric_pct) > best_baseline,
        "beats_best_baseline": bool(baselines) and float(policy_metric_pct) > best_baseline,
        "rows": {"total": rows_total, "embedded_in_module": embedded_rows},
        "notes": list(notes or []),
    }
    if deployment:
        scorecard["deployment"] = deployment
    if selection:
        scorecard["selection"] = selection
    if eval_corpus:
        scorecard["eval_corpus"] = eval_corpus
    if crossfire:
        scorecard["crossfire"] = crossfire
    if economics:
        scorecard["economics"] = economics
    if not held_out:
        if task_class == "classification":
            scorecard["notes"].append(
                "Held-out evaluation unavailable (too few usable rows for a test split); this "
                "number is in-sample and must not be quoted as a generalisation result."
            )
        else:
            scorecard["notes"].append(
                "Control task: no dataset split applies; success is measured over full episodes "
                "against the environment's own dynamics."
            )
    return scorecard


# ---------------------------------------------------------------------------
# Economics ledger
#
# The Crossfire objection was not "cost is never measured" - it was that the ledger was
# one-sided: authoring cost was quoted as token spend while the recurring cost of *maintaining*
# the reward, the verifier and the drift monitor was omitted, and the frontier baseline ignored
# that a person reviews API output too. So this ledger (a) prices the measured token spend,
# (b) prices the human hours the studio already consumes, (c) amortises retraining and drift
# review over a stated cadence, and (d) reports the result as a range with its assumptions
# attached. It never prints a single flattering number.
# ---------------------------------------------------------------------------

DEFAULT_LOADED_RATE_USD_PER_HOUR = 120.0
DEFAULT_MINUTES_PER_AUTHORING_TURN = 6.0
DEFAULT_RETRAIN_COMPUTE_USD = 0.02
DEFAULT_RETRAIN_CADENCE_DAYS = 30
DEFAULT_DECISIONS_PER_DAY = 250.0
DEFAULT_DRIFT_REVIEW_HOURS_PER_CYCLE = 1.5
DEFAULT_FRONTIER_LATENCY_MS = 850.0
DEFAULT_FRONTIER_TIERS: dict[str, dict[str, float | str]] = {
    "budget_frontier": {"usd_per_decision": 0.0016, "label": "$1.00/$5.00 per 1M tokens"},
    "flagship_frontier": {"usd_per_decision": 0.0045, "label": "$5.63 per 1M blended tokens"},
}


def _ceil(value: float) -> int:
    return int(math.ceil(max(0.0, float(value))))


def token_cost_for_session(session_id: str) -> dict[str, Any]:
    """Measured provider spend for a conversation, priced by the local table."""
    try:
        from rl_ide.llm import metering

        report = metering.usage_report(conversation_id=session_id, limit=1)
        totals = report.get("totals", {}) if isinstance(report, dict) else {}
        return {
            "cost_usd": round(float(totals.get("cost_usd") or 0.0), 6),
            "calls": int(totals.get("calls") or totals.get("api_calls") or 0),
            "source": "metered",
        }
    except Exception as exc:  # metering must never break a scorecard
        return {"cost_usd": 0.0, "calls": 0, "source": f"unavailable: {type(exc).__name__}"}


def compute_economics_summary(
    session_id: str,
    *,
    onnx_latency_stats: dict[str, float] | None = None,
    feature_pipeline_ms: float = 0.0,
    authoring_turns: int | None = None,
    minutes_per_authoring_turn: float = DEFAULT_MINUTES_PER_AUTHORING_TURN,
    loaded_rate_usd_per_hour: float = DEFAULT_LOADED_RATE_USD_PER_HOUR,
    retrain_compute_usd: float = DEFAULT_RETRAIN_COMPUTE_USD,
    retrain_cadence_days: int = DEFAULT_RETRAIN_CADENCE_DAYS,
    decisions_per_day: float = DEFAULT_DECISIONS_PER_DAY,
    drift_review_hours_per_cycle: float = DEFAULT_DRIFT_REVIEW_HOURS_PER_CYCLE,
    frontier_latency_ms: float = DEFAULT_FRONTIER_LATENCY_MS,
    frontier_tiers: dict[str, dict[str, float | str]] | None = None,
    frontier_review_minutes_per_decision: float = 0.25,
    local_review_minutes_per_decision: float = 0.02,
    measured: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """First-year cost of owning the policy vs. per-decision frontier spend.

    Returns the ledger, the payback point per frontier tier (in decisions and in days), and the
    assumptions used. ``latency_speedup_factor`` compares like with like: the ONNX p99 plus the
    feature pipeline the caller measured, against a configurable front-end latency.
    """
    usage = measured or token_cost_for_session(session_id)
    tiers = frontier_tiers or DEFAULT_FRONTIER_TIERS

    calls = int(usage.get("calls") or 0)
    turns = int(authoring_turns) if authoring_turns is not None else max(1, _ceil(calls / 6.0))
    human_hours = round(turns * float(minutes_per_authoring_turn) / 60.0, 2)
    human_usd = round(human_hours * float(loaded_rate_usd_per_hour), 4)

    cycles = max(1.0, 365.0 / max(1, int(retrain_cadence_days)))
    retrain_usd = round(float(retrain_compute_usd) * cycles, 4)
    drift_usd = round(
        float(drift_review_hours_per_cycle) * cycles * float(loaded_rate_usd_per_hour), 4
    )
    authoring_tokens_usd = round(float(usage.get("cost_usd") or 0.0), 6)

    first_year_usd = round(authoring_tokens_usd + human_usd + retrain_usd + drift_usd, 4)

    # Local inference is not free: ONNX on a laptop is cheap, but it is not zero.
    local_compute_usd_per_decision = 0.00002

    latency: dict[str, Any] = {"status": "not_measured"}
    if onnx_latency_stats:
        p50 = float(onnx_latency_stats.get("p50_ms") or 0.0)
        p99 = float(onnx_latency_stats.get("p99_ms") or 0.0)
        pipeline = max(0.0, float(feature_pipeline_ms))
        end_to_end_p99 = round(p99 + pipeline, 3)
        latency = {
            "status": "measured",
            "onnx_p50_ms": round(p50, 3),
            "onnx_p99_ms": round(p99, 3),
            "feature_pipeline_ms": round(pipeline, 3),
            "end_to_end_p99_ms": end_to_end_p99,
            "frontier_reference_ms": float(frontier_latency_ms),
            "latency_speedup_factor": round(
                float(frontier_latency_ms) / max(0.01, end_to_end_p99), 1
            ),
            "scope": "single decision, excluding data arrival and human review on both paths",
        }

    # Both sides of the ledger carry human review, so the comparison is like with like. A
    # frontier call is spot-checked per decision; an owned policy is spot-checked far less often
    # and pays for retraining/drift review amortised across the year instead.
    review_usd_per_min = float(loaded_rate_usd_per_hour) / 60.0
    frontier_review_usd = float(frontier_review_minutes_per_decision) * review_usd_per_min
    local_review_usd = float(local_review_minutes_per_decision) * review_usd_per_min
    local_per_decision_with_review = local_compute_usd_per_decision + local_review_usd

    payback: dict[str, Any] = {}
    for name, tier in tiers.items():
        per_decision = float(tier["usd_per_decision"])
        token_only_saving = max(1e-9, per_decision - local_compute_usd_per_decision)
        reviewed_saving = max(
            1e-9, per_decision + frontier_review_usd - local_per_decision_with_review
        )
        decisions_token = _ceil(first_year_usd / token_only_saving)
        decisions_reviewed = _ceil(first_year_usd / reviewed_saving)
        payback[name] = {
            "label": str(tier.get("label", name)),
            "usd_per_decision": per_decision,
            "basis_token_only": {
                "decisions_to_payback": decisions_token,
                "days_to_payback": round(decisions_token / max(1.0, float(decisions_per_day)), 2),
            },
            "basis_with_human_review": {
                "frontier_usd_per_decision": round(per_decision + frontier_review_usd, 5),
                "local_usd_per_decision": round(local_per_decision_with_review, 5),
                "decisions_to_payback": decisions_reviewed,
                "days_to_payback": round(
                    decisions_reviewed / max(1.0, float(decisions_per_day)), 2
                ),
            },
        }

    sensitivity = {}
    for multiplier in (0.5, 1.0, 2.0):
        scenario_human = round(human_hours * multiplier * float(loaded_rate_usd_per_hour), 4)
        scenario_total = round(
            authoring_tokens_usd + scenario_human + retrain_usd + drift_usd, 4
        )
        flagship = tiers["flagship_frontier"]
        reviewed_saving = max(
            1e-9,
            float(flagship["usd_per_decision"]) + frontier_review_usd - local_per_decision_with_review,
        )
        sensitivity[f"human_hours_x{multiplier}"] = {
            "first_year_usd": scenario_total,
            "days_to_payback_flagship": round(
                _ceil(scenario_total / reviewed_saving) / max(1.0, float(decisions_per_day)),
                2,
            ),
        }

    return {
        "ledger": {
            "authoring_tokens_usd": authoring_tokens_usd,
            "authoring_human_hours": human_hours,
            "authoring_human_usd": human_usd,
            "retrain_compute_usd_per_year": retrain_usd,
            "drift_review_usd_per_year": drift_usd,
            "local_compute_usd_per_decision": local_compute_usd_per_decision,
            "first_year_usd": first_year_usd,
        },
        "payback": payback,
        "sensitivity": sensitivity,
        "latency": latency,
        "assumptions": {
            "authoring_turns": turns,
            "minutes_per_authoring_turn": float(minutes_per_authoring_turn),
            "loaded_rate_usd_per_hour": float(loaded_rate_usd_per_hour),
            "retrain_cadence_days": int(retrain_cadence_days),
            "decisions_per_day": float(decisions_per_day),
            "frontier_review_minutes_per_decision": float(frontier_review_minutes_per_decision),
            "local_review_minutes_per_decision": float(local_review_minutes_per_decision),
            "usage_source": usage.get("source", "unknown"),
        },
        "caveats": [
            "Authoring uses a BYOK provider: prompts leave the boundary during authoring unless a "
            "local model is configured. Only inference is boundary-local.",
            "Human hours are the dominant term and are estimated from the number of provider "
            "calls; pass measured turns to replace the estimate.",
            "Latency compares one local decision to one provider round-trip; it excludes data "
            "arrival and human review on both paths.",
        ],
    }
