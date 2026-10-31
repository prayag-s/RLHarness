"""Independent verifier gate for every post-training mutation.

The product's central loop is "critique in English -> change the reward -> retrain briefly". The
failure mode that destroys trust is keeping a change because the *shaped reward* went up, when
the behaviour the user already approved got worse. This module decides retention from evidence
the mutation did not produce:

* a held-out validation metric, compared **paired** on the same deterministic episodes, and
* the golden eval corpus, compared case-by-case against the ids that passed before.

Every decision is deterministic and every failure is classified by an ordered rule table, so a
non-expert user is told *why* a critique was rolled back instead of being left to guess between
a bad compile, clashing reward terms and RL noise.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, Field

# 1.5 points of validation movement is inside the sampling noise of a 50-episode evaluation;
# above that the drop is treated as real unless a paired test says otherwise.
DEFAULT_TOLERANCE_PCT = 1.5
# Below this many paired episodes a sign test has no power, so the point estimate is honoured.
MIN_PAIRED_EPISODES = 20
# Two-sided exact binomial p-value above which a drop is attributed to noise.
PAIRED_SIGNIFICANCE = 0.05
# Entropy rise (nats) that marks an unexplored / thrashing policy rather than a wrong reward.
ENTROPY_DRIFT_THRESHOLD = 0.15


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sign_test_p(improved: int, worsened: int) -> float | None:
    """Two-sided exact binomial p-value for a paired comparison.

    Uses the sign test rather than a t-test because per-episode outcomes are 0/1 or bounded
    percentages, and because it needs no SciPy: only ties and the direction of each pair matter.
    """
    n = int(improved) + int(worsened)
    if n <= 0:
        return None
    k = min(int(improved), int(worsened))
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return round(min(1.0, 2.0 * tail), 6)


class VerifierSnapshot(BaseModel):
    """What the policy could do before a mutation was applied."""

    mutation_kind: str = "reward"  # reward | environment | inference
    session_id: str = ""
    env_code: str = ""
    model_path: str | None = None
    val_metric: float | None = None
    val_outcomes: list[float] = Field(default_factory=list)
    corpus_passing_ids: list[str] = Field(default_factory=list)
    corpus_scored_cases: int = 0
    corpus_pass_rate_pct: float | None = None
    term_stats: dict[str, Any] = Field(default_factory=dict)
    entropy: float | None = None
    taken_at: str = Field(default_factory=_now)


class CritiqueTriageReport(BaseModel):
    category: str  # compiler_mismatch | cross_term_interference | rl_variance | unclassified
    advice: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class GateDecision(BaseModel):
    passed: bool
    status: str  # retained | rolled_back
    reason: str
    val_before: float | None = None
    val_after: float | None = None
    val_delta: float | None = None
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT
    improved_episodes: int = 0
    worsened_episodes: int = 0
    paired_p_value: float | None = None
    regressed_case_ids: list[str] = Field(default_factory=list)
    corpus_pass_rate_pct: float | None = None
    triage: CritiqueTriageReport | None = None


def triage_failure(
    *,
    mutation_kind: str,
    term_stats: dict[str, Any] | None,
    val_delta: float | None,
    regressed_case_ids: Sequence[str],
    entropy_delta: float | None,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> CritiqueTriageReport:
    """Deterministic rule table: the first matching row wins.

    1. the mutated term could never fire on any evaluated state        -> compiler_mismatch
    2. the term was eligible but never activated in the burst          -> compiler_mismatch
    3. a previously passing corpus case regressed                      -> cross_term_interference
    4. validation dropped beyond tolerance with the term active        -> cross_term_interference
    5. entropy rose or the burst budget ran out                        -> rl_variance
    6. otherwise                                                       -> unclassified
    """
    stats = term_stats or {}
    eligible = stats.get("eligible")
    activation = stats.get("activation_rate")
    evidence = {
        "mutation_kind": mutation_kind,
        "term_eligible": eligible,
        "term_activation_rate": activation,
        "mean_term_value": stats.get("mean_term_value"),
        "val_delta_pct": None if val_delta is None else round(float(val_delta), 2),
        "regressed_cases": list(regressed_case_ids),
        "entropy_delta": None if entropy_delta is None else round(float(entropy_delta), 4),
    }

    if eligible is False:
        return CritiqueTriageReport(
            category="compiler_mismatch",
            advice=(
                "The new condition never became true on any state the policy actually visited. "
                "Show the generated expression and the observed variable ranges, then widen the "
                "threshold or fix the field name."
            ),
            evidence=evidence,
        )
    if stats.get("new_terms") and not activation:
        return CritiqueTriageReport(
            category="compiler_mismatch",
            advice=(
                "The new term never fired on any evaluated step: the compiled condition does not "
                "match the states the policy visits. Show the generated expression and the observed "
                "variable ranges, then widen the threshold."
            ),
            evidence=evidence,
        )
    if eligible is True and not activation:
        return CritiqueTriageReport(
            category="compiler_mismatch",
            advice=(
                "The condition is satisfiable but the policy never triggered it, so the new signal "
                "was never learned from. Lower its threshold or check that the planner reaches "
                "those states at all."
            ),
            evidence=evidence,
        )
    if regressed_case_ids:
        return CritiqueTriageReport(
            category="cross_term_interference",
            advice=(
                "The change over-dominated an existing rule and broke previously approved cases. "
                f"Regressed: {', '.join(list(regressed_case_ids)[:5])}. Halve the new term's weight "
                "and re-apply."
            ),
            evidence=evidence,
        )
    if val_delta is not None and val_delta < -tolerance_pct:
        return CritiqueTriageReport(
            category="cross_term_interference",
            advice=(
                "Held-out accuracy fell beyond tolerance while the new term was active: the term is "
                "trading real correctness for its own signal. Reduce its weight."
            ),
            evidence=evidence,
        )
    if entropy_delta is not None and entropy_delta >= ENTROPY_DRIFT_THRESHOLD:
        return CritiqueTriageReport(
            category="rl_variance",
            advice=(
                "Policy entropy rose during the burst: it thrashed instead of converging. Retry with "
                "a longer burst and a lower learning rate before changing the reward again."
            ),
            evidence=evidence,
        )
    return CritiqueTriageReport(
        category="rl_variance",
        advice=(
            "No term-level or corpus-level fault was identified; the burst most likely needed more "
            "steps. Retry with double the episodes."
        ),
        evidence=evidence,
    )


class VerifierGate:
    """Snapshots the pre-mutation state and decides whether the mutation may be kept."""

    def __init__(
        self,
        *,
        tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
        min_paired_episodes: int = MIN_PAIRED_EPISODES,
        significance: float = PAIRED_SIGNIFICANCE,
    ):
        self.tolerance_pct = tolerance_pct
        self.min_paired_episodes = min_paired_episodes
        self.significance = significance

    def decide(
        self,
        snapshot: VerifierSnapshot,
        *,
        val_metric_after: float | None,
        val_outcomes_after: Iterable[float] | None = None,
        corpus_passing_ids_after: Iterable[str] | None = None,
        corpus_pass_rate_after: float | None = None,
        term_stats: dict[str, Any] | None = None,
        entropy_after: float | None = None,
    ) -> GateDecision:
        """Applies the retention rule and, on failure, the triage table."""
        after_outcomes = [float(x) for x in (val_outcomes_after or [])]
        before_outcomes = [float(x) for x in snapshot.val_outcomes]

        improved = worsened = paired = 0
        for before, after in zip(before_outcomes, after_outcomes):
            paired += 1
            if after > before + 1e-9:
                improved += 1
            elif after < before - 1e-9:
                worsened += 1
        paired_p = sign_test_p(improved, worsened) if paired else None

        regressions = sorted(
            set(snapshot.corpus_passing_ids) - set(corpus_passing_ids_after or [])
        )

        delta = (
            None
            if (val_metric_after is None or snapshot.val_metric is None)
            else round(float(val_metric_after) - float(snapshot.val_metric), 2)
        )
        entropy_delta = (
            None
            if (entropy_after is None or snapshot.entropy is None)
            else round(float(entropy_after) - float(snapshot.entropy), 4)
        )

        passed = True
        reason = "Held-out accuracy held and no approved case regressed."
        if regressions:
            passed = False
            reason = f"{len(regressions)} previously passing corpus case(s) regressed."
        elif delta is not None and delta < -self.tolerance_pct:
            noisy = (
                paired >= self.min_paired_episodes
                and paired_p is not None
                and paired_p > self.significance
            )
            if noisy:
                reason = (
                    f"Held-out accuracy moved {delta:+.2f} pts but the paired sign test "
                    f"(p={paired_p}) cannot distinguish that from noise over {paired} episodes."
                )
            else:
                passed = False
                reason = f"Held-out accuracy fell {abs(delta):.2f} pts, beyond tolerance."

        decision = GateDecision(
            passed=passed,
            status="retained" if passed else "rolled_back",
            reason=reason,
            val_before=snapshot.val_metric,
            val_after=val_metric_after,
            val_delta=delta,
            tolerance_pct=self.tolerance_pct,
            improved_episodes=improved,
            worsened_episodes=worsened,
            paired_p_value=paired_p,
            regressed_case_ids=regressions,
            corpus_pass_rate_pct=corpus_pass_rate_after,
        )
        if not passed:
            decision.triage = triage_failure(
                mutation_kind=snapshot.mutation_kind,
                term_stats=term_stats if term_stats is not None else snapshot.term_stats,
                val_delta=delta,
                regressed_case_ids=regressions,
                entropy_delta=entropy_delta,
                tolerance_pct=self.tolerance_pct,
            )
        return decision


def rollout_metric(
    env: Any,
    predict_fn: Callable[[Sequence[float]], int],
    *,
    split: str | None = None,
    n_episodes: int = 20,
) -> tuple[float, list[float]]:
    """Deterministic per-episode metric used for paired before/after comparison.

    Kept here (not in the trainer) so the chat-mutation path and the training loop judge a change
    with identical arithmetic.
    """
    from rl_ide.autopilot.loop_controller import episode_metric_value, episode_step_limit

    values: list[float] = []
    for ep_idx in range(1, max(1, n_episodes) + 1):
        reset_opts: dict[str, Any] = {}
        if split:
            reset_opts["split"] = split
        try:
            obs, _info = env.reset(options=reset_opts) if reset_opts else env.reset()
        except (TypeError, ValueError):
            obs, _info = env.reset()
        steps: list[dict[str, Any]] = []
        for _ in range(episode_step_limit(env)):
            action = int(predict_fn([float(x) for x in obs]))
            obs, _reward, terminated, truncated, info = env.step(action)
            steps.append(
                {
                    "is_success": bool(info.get("is_success", False)),
                    "is_correct": info.get("is_correct"),
                }
            )
            if terminated or truncated:
                break
        values.append(round(episode_metric_value(env, steps), 2))
    metric = round(sum(values) / max(1, len(values)), 2)
    return metric, values


def restore_file(path: Path | str, content: str) -> None:
    """Writes back a snapshot; used for rollback of generated environment modules."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def offline_metric(
    env: Any,
    predict_fn: Callable[[Sequence[float]], int],
    *,
    split: str = "val",
    n_samples: int = 200,
) -> tuple[float, list[float]] | None:
    """Held-out accuracy on labelled rows, without resetting the environment.

    Used for chat mutations, where comparing two rollouts would compare two different row
    sequences. Scoring the same labelled rows before and after makes the comparison paired and
    cheap; returns None when the environment has no labelled demonstrations (control tasks).
    """
    getter = getattr(env, "get_demonstrations", None)
    if getter is None:
        return None
    try:
        features, targets = getter(num_samples=n_samples, split=split)
    except Exception:
        return None
    if features is None or targets is None or len(features) == 0:
        return None
    outcomes = [
        100.0 if int(predict_fn([float(x) for x in obs])) == int(target) else 0.0
        for obs, target in zip(features, targets)
    ]
    return round(sum(outcomes) / max(1, len(outcomes)), 2), outcomes
