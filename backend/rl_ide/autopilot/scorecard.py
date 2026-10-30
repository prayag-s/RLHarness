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
        "rows": {"total": rows_total, "embedded_in_module": embedded_rows},
        "notes": list(notes or []),
    }
    if deployment:
        scorecard["deployment"] = deployment
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
