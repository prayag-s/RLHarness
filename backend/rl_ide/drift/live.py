"""Live drift scoring for the deployed policy.

`DriftDetector` was built in v4.0 but only ever imported by the exporter, so the header said
"unmonitored" for every session and a policy could silently degrade while the studio showed a green
scorecard. This module scores the observations a policy actually sees — one per `/interact` step —
against the baseline fitted from its own training episodes, and reports the moment the boundary is
crossed.

Two rules shape the design:

* **Never score a window that is too small to mean anything.** PSI over five samples is noise, so a
  monitor warms up and reports ``warming_up`` rather than inventing a verdict.
* **Never drift silently, and never drift-permanently.** A single crossing raises ``drifted``; it is
  cleared only when a later window comes back inside the boundary, so a transient spike does not
  alarm forever.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rl_ide.drift.monitor import DriftBaseline, DriftDetector, DriftReport

UNMONITORED = "unmonitored"
WARMING_UP = "warming_up"
HEALTHY = "healthy"
DRIFTED = "drifted"

# A PSI computed on a handful of observations is noise. Measured against a healthy policy this
# detector flags ~10 % of 50-sample windows by chance at the standard 0.25 threshold, so a window is
# scored at a coarser resolution and a single crossing is not treated as an incident.
DEFAULT_WINDOW = 50
DEFAULT_STRIDE = 5
# Consecutive drifting windows required before the status changes. Two independent windows agreeing
# turns a ~10 % per-window false-positive rate into ~1 %, while genuine drift flags every window.
DEFAULT_CONFIRMATIONS = 2
# Decile histograms need ~10 samples per bin; a live window holds tens of samples in total.
LIVE_MAX_BINS = 5
# The textbook PSI boundary (0.25) is calibrated for batches of thousands. At a 50-sample live
# window the estimator's own noise reaches 0.29 on unchanged traffic, so live scoring uses twice
# that boundary; measured below in `tests/test_drift_live.py`.
LIVE_PSI_THRESHOLD = 0.5


class DriftMonitor:
    """Scores live observations against one conversation's training baseline."""

    def __init__(
        self,
        baseline: DriftBaseline,
        *,
        window: int = DEFAULT_WINDOW,
        stride: int = DEFAULT_STRIDE,
        confirmations: int = DEFAULT_CONFIRMATIONS,
        max_bins: int = LIVE_MAX_BINS,
        psi_threshold: float = LIVE_PSI_THRESHOLD,
    ):
        self.detector = DriftDetector(baseline)
        self.window = max(2, int(window))
        self.stride = max(1, int(stride))
        self.confirmations = max(1, int(confirmations))
        self.max_bins = max(2, int(max_bins))
        self.psi_threshold = float(psi_threshold)
        self.consecutive_drifted = 0
        self._obs: list[list[float]] = []
        self._probs: list[list[float]] = []
        self._since_check = 0
        self.last_report: DriftReport | None = None
        self.status: str = WARMING_UP

    def observe(
        self, obs: list[float] | None, action_probs: list[float] | None, n_actions: int | None = None
    ) -> DriftReport | None:
        """Records one live step. Returns a report only when a window was actually evaluated."""
        if obs is None:
            return None
        try:
            vector = [float(v) for v in obs]
        except (TypeError, ValueError):
            return None

        probs = self._as_distribution(action_probs, n_actions)
        self._obs.append(vector)
        self._probs.append(probs)
        if len(self._obs) > self.window:
            self._obs.pop(0)
            self._probs.pop(0)

        self._since_check += 1
        if len(self._obs) < self.window or self._since_check < self.stride:
            # Between judgments the previous verdict stands: flipping back to "warming_up" after the
            # first window would erase a real answer on the next click.
            if self.last_report is None:
                self.status = WARMING_UP
            return None
        self._since_check = 0

        width = len(self.detector.baseline.feature_names)
        obs_batch = np.array([row[:width] for row in self._obs], dtype=np.float32)
        probs_batch = np.array(self._probs, dtype=np.float32)
        report = self.detector.check_drift(
            obs_batch, probs_batch, max_bins=self.max_bins, psi_threshold=self.psi_threshold
        )
        self.last_report = report
        self.consecutive_drifted = self.consecutive_drifted + 1 if report.is_drifted else 0
        # A single crossing is reported but not acted on: PSI at this sample size is noisy.
        self.status = DRIFTED if self.consecutive_drifted >= self.confirmations else HEALTHY
        return report

    @staticmethod
    def _as_distribution(action_probs: list[float] | None, n_actions: int | None) -> list[float]:
        """Normalises whatever the env returned into a probability vector."""
        try:
            values = [max(0.0, float(v)) for v in (action_probs or [])]
        except (TypeError, ValueError):
            values = []
        if not values:
            # Without the policy's own probabilities a uniform vector keeps the entropy signal
            # neutral instead of fabricating confidence the model never expressed.
            return [1.0 / max(1, n_actions or 1)] * max(1, n_actions or 1)
        total = sum(values)
        if total <= 0:
            return [1.0 / len(values)] * len(values)
        return [v / total for v in values]

    def snapshot(self) -> dict[str, Any]:
        """Public view for the API and the header."""
        report = self.last_report
        return {
            "status": self.status,
            "samples_seen": len(self._obs),
            "window": self.window,
            "confirmations": self.confirmations,
            "psi_threshold": self.psi_threshold,
            "consecutive_drifted": self.consecutive_drifted,
            # Deliberately the *confirmed* verdict: a lone noisy window must not raise the alarm.
            "is_drifted": self.status == DRIFTED,
            "unconfirmed_crossing": bool(report.is_drifted) if report else False,
            "max_psi": float(report.max_psi) if report else 0.0,
            "entropy_zscore": float(report.entropy_zscore) if report else 0.0,
            "action_shift_zscore": float(report.action_shift_zscore) if report else 0.0,
            "reasons": list(report.reasons) if report else [],
            "recommendation": report.recommendation if report else "Waiting for enough live samples.",
            "baseline_samples": int(self.detector.baseline.n_samples),
        }


def _episode_observations(raw: dict[str, Any]) -> tuple[list[list[float]], list[int]]:
    """Reads observations and chosen actions from either stored-episode shape.

    The conversations row keeps the API shape (`{"obs": [...], "action": 2}`) while the contracts
    module uses `EpisodeLog` (`{"obs_vector": [...], "events": {"action": {...}}}`). Baseline fitting
    must accept both: a training run stores the first, and it is the only record that survives.
    """
    observations: list[list[float]] = []
    actions: list[int] = []
    for step in raw.get("steps") or []:
        if not isinstance(step, dict):
            continue
        vector = step.get("obs_vector")
        if vector is None:
            vector = step.get("obs")
        try:
            observations.append([float(v) for v in vector])
        except (TypeError, ValueError):
            continue
        action = step.get("action")
        if action is None:
            events = step.get("events") or {}
            action = (events.get("action") or {}).get("action_idx") if isinstance(events, dict) else None
        try:
            actions.append(int(action))
        except (TypeError, ValueError):
            actions.append(0)
    return observations, actions


def fit_baseline_from_episodes(
    episodes: list[dict[str, Any]] | list[Any],
    feature_names: list[str],
) -> DriftBaseline | None:
    """Fits the live baseline from the episodes a training run actually stored.

    Only the input distribution is fitted: stored episodes keep the chosen action but not the policy's
    probabilities, so the entropy and action-rate checks would be comparing against a baseline nobody
    measured. `DriftBaseline.policy_signals` records that, and `check_drift` skips those two signals.
    """
    from rl_ide.contracts.adapter import EpisodeLog, StepResult

    steps: list[StepResult] = []
    for raw in episodes or []:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        if not isinstance(raw, dict):
            continue
        observations, actions = _episode_observations(raw)
        for vector, action in zip(observations, actions):
            steps.append(
                StepResult(
                    obs_vector=vector,
                    events={"action": {"action_idx": action}},
                    terminated=False,
                    truncated=False,
                )
            )
    if not steps:
        return None

    width = max(len(step.obs_vector) for step in steps)
    if width == 0:
        return None
    steps = [step for step in steps if len(step.obs_vector) == width]
    names = list(feature_names or [])
    if len(names) < width:
        names += [f"feature_{i}" for i in range(len(names), width)]
    names = names[:width]
    try:
        return DriftDetector.fit_baseline(
            episodes=[EpisodeLog(episode_id=0, steps=steps)],
            feature_names=names,
            action_probs=None,
        )
    except Exception:
        return None


def capture_observations(env: Any) -> list[float]:
    """Makes the last observation an environment produced available to the caller.

    A human click specifies the action, so `interactive_step` does not always invoke the policy and
    the observation never reaches `predict()`. Wrapping `step`/`reset` captures it regardless of how
    the generated module drives itself, without changing that module.
    """
    last: list[float] = []
    original_step = getattr(env, "step", None)
    original_reset = getattr(env, "reset", None)

    def _record(value: Any) -> None:
        try:
            last[:] = [float(x) for x in np.asarray(value, dtype="float32").reshape(-1)]
        except (TypeError, ValueError):
            pass

    if callable(original_step):
        def step(*args: Any, **kwargs: Any) -> Any:
            out = original_step(*args, **kwargs)
            if isinstance(out, tuple) and out:
                _record(out[0])
            return out

        try:
            env.step = step
        except Exception:
            pass

    if callable(original_reset):
        def reset(*args: Any, **kwargs: Any) -> Any:
            out = original_reset(*args, **kwargs)
            _record(out[0] if isinstance(out, tuple) and out else out)
            return out

        try:
            env.reset = reset
        except Exception:
            pass

    return last


class DriftStatusStore:
    """Persists the latest live status next to the conversation it belongs to.

    A process restart must not silently reset a drifting policy to "unmonitored": the status is a
    property of the conversation, not of the process that happened to observe it.
    """

    def status_path(self, workspace_dir: Path) -> Path:
        return Path(workspace_dir) / "drift" / "live_status.json"

    def save(self, workspace_dir: Path, snapshot: dict[str, Any]) -> None:
        path = self.status_path(workspace_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load(self, workspace_dir: Path) -> dict[str, Any]:
        path = self.status_path(workspace_dir)
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {"status": UNMONITORED}
