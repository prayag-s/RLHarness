"""Pre-registered falsification protocol for the "small policy beats frontier" claim.

An instrument that *can* measure a claim is not evidence that the claim is true. The only way to
close a premise objection without circularity is to freeze the rubric and the acceptance
threshold **before** the policy is trained, keep a hold-out set nobody tuned against, and then
report a verdict the run cannot influence.

This module is that guard. A closure verdict is refused (``closed: False``) whenever the rubric
was registered after training started, the rubric text no longer hashes to what was registered,
or fewer hold-out judgements exist than the protocol asked for. What remains is a scorecard that
says "win", "loss" or "inconclusive" instead of "v5.5 shipped".
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

PROTOCOL_FILENAME = "crossfire_protocol.json"
PROTOCOL_VERSION = 1

# The smallest held-out critique set with enough power to say anything at all; below this the
# honest answer is "inconclusive", not "closed".
DEFAULT_MIN_HOLDOUT = 100


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _epoch() -> float:
    return round(time.time(), 3)


def rubric_hash(rubric: Sequence[str]) -> str:
    """Stable hash of the acceptance rubric, so post-hoc edits are detectable."""
    canonical = json.dumps([str(line).strip() for line in rubric], separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProtocolViolation(RuntimeError):
    """Raised when code tries to score a claim against a rubric that was not pre-registered."""


class FalsificationProtocol(BaseModel):
    version: int = PROTOCOL_VERSION
    session_id: str
    task_title: str = ""
    task_class: str = ""
    rubric: list[str] = Field(default_factory=list)
    rubric_hash: str = ""
    min_holdout: int = DEFAULT_MIN_HOLDOUT
    target_pass_rate_pct: float = 60.0
    opened_at: str = Field(default_factory=_now)
    opened_at_epoch: float = Field(default_factory=_epoch)
    training_started_at: str | None = None
    training_started_at_epoch: float | None = None
    holdout_case_ids: list[str] = Field(default_factory=list)
    verdict: dict[str, Any] | None = None


class CrossfireProtocolStore:
    """Persists the protocol next to the session it governs."""

    def __init__(self, base_dir: Path | str = "workspaces"):
        self.base_dir = Path(base_dir)

    def path(self, session_id: str) -> Path:
        return self.base_dir / session_id / PROTOCOL_FILENAME

    def load(self, session_id: str) -> FalsificationProtocol | None:
        path = self.path(session_id)
        if not path.exists():
            return None
        try:
            return FalsificationProtocol.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception:
            return None

    def save(self, protocol: FalsificationProtocol) -> Path:
        path = self.path(protocol.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(protocol.model_dump_json(indent=2), encoding="utf-8")
        return path

    def open(
        self,
        session_id: str,
        *,
        rubric: Sequence[str],
        task_title: str = "",
        task_class: str = "",
        min_holdout: int = DEFAULT_MIN_HOLDOUT,
        target_pass_rate_pct: float = 60.0,
    ) -> FalsificationProtocol:
        """Registers the rubric; must be called before the first training step."""
        protocol = FalsificationProtocol(
            session_id=session_id,
            task_title=task_title,
            task_class=task_class,
            rubric=[str(line).strip() for line in rubric if str(line).strip()],
            rubric_hash=rubric_hash(rubric),
            min_holdout=int(min_holdout),
            target_pass_rate_pct=float(target_pass_rate_pct),
        )
        return protocol if self.save(protocol) else protocol

    def record_training_start(self, session_id: str) -> FalsificationProtocol | None:
        """Stamps the first training step; never overwrites an earlier stamp."""
        protocol = self.load(session_id)
        if protocol is None:
            return None
        if protocol.training_started_at is None:
            protocol.training_started_at = _now()
            protocol.training_started_at_epoch = _epoch()
            self.save(protocol)
        return protocol

    def register_holdout(self, session_id: str, case_ids: Iterable[str]) -> int:
        """Records which judgements were frozen before training began."""
        protocol = self.load(session_id)
        if protocol is None:
            return 0
        merged = sorted(set(protocol.holdout_case_ids) | {str(c) for c in case_ids})
        protocol.holdout_case_ids = merged
        self.save(protocol)
        return len(merged)

    # -- evaluation --------------------------------------------------------
    def preregistration_gaps(self, protocol: FalsificationProtocol) -> list[str]:
        """Everything that makes a verdict unsound; empty list means the protocol holds."""
        gaps: list[str] = []
        if not protocol.rubric:
            gaps.append("No acceptance rubric was registered.")
        if rubric_hash(protocol.rubric) != protocol.rubric_hash:
            gaps.append("The rubric text no longer matches the hash registered before training.")
        if protocol.training_started_at_epoch is None:
            gaps.append(
                "No training-start stamp exists, so pre-registration cannot be proven."
            )
        elif protocol.opened_at_epoch > protocol.training_started_at_epoch:
            gaps.append(
                "The rubric was registered after training started; the result is post-hoc."
            )
        if len(protocol.holdout_case_ids) < protocol.min_holdout:
            gaps.append(
                f"Only {len(protocol.holdout_case_ids)} frozen hold-out judgements exist; "
                f"the protocol requires {protocol.min_holdout}."
            )
        return gaps

    def evaluate(
        self,
        session_id: str,
        *,
        holdout_passed: int,
        holdout_total: int,
        policy_metric: float | None = None,
        baseline_metric: float | None = None,
    ) -> dict[str, Any]:
        """Scores the frozen claim. Refuses to call it closed when pre-registration is broken."""
        protocol = self.load(session_id)
        if protocol is None:
            return {
                "closed": False,
                "verdict": "not_preregistered",
                "reasons": ["No falsification protocol was registered for this session."],
            }

        gaps = self.preregistration_gaps(protocol)
        pass_rate = (
            round(100.0 * holdout_passed / holdout_total, 2) if holdout_total > 0 else 0.0
        )
        beats_baseline = (
            policy_metric is not None
            and baseline_metric is not None
            and float(policy_metric) > float(baseline_metric)
        )

        if gaps:
            verdict = "inconclusive"
            closed = False
        elif pass_rate >= protocol.target_pass_rate_pct and beats_baseline:
            verdict = "win"
            closed = True
        elif pass_rate >= protocol.target_pass_rate_pct and baseline_metric is None:
            verdict = "inconclusive"
            closed = False
            gaps = ["No baseline was measured, so the margin over a frontier-style baseline is unknown."]
        else:
            verdict = "loss"
            closed = True

        result = {
            "closed": closed,
            "verdict": verdict,
            "holdout_total": int(holdout_total),
            "holdout_passed": int(holdout_passed),
            "holdout_pass_rate_pct": pass_rate,
            "target_pass_rate_pct": protocol.target_pass_rate_pct,
            "min_holdout": protocol.min_holdout,
            "policy_metric": policy_metric,
            "baseline_metric": baseline_metric,
            "beats_baseline": beats_baseline,
            "rubric_hash": protocol.rubric_hash,
            "opened_at": protocol.opened_at,
            "training_started_at": protocol.training_started_at,
            "reasons": gaps,
        }
        protocol.verdict = result
        self.save(protocol)
        return result

    def closure_report(self, session_id: str) -> dict[str, Any]:
        """The block a scorecard may show; never claims closure on an unsound protocol."""
        protocol = self.load(session_id)
        if protocol is None:
            return {"closed": False, "verdict": "not_preregistered", "reasons": [
                "No falsification protocol was registered for this session."
            ]}
        if protocol.verdict:
            return dict(protocol.verdict)
        gaps = self.preregistration_gaps(protocol)
        return {
            "closed": False,
            "verdict": "inconclusive",
            "rubric_hash": protocol.rubric_hash,
            "opened_at": protocol.opened_at,
            "training_started_at": protocol.training_started_at,
            # Enough for a caller to show progress toward the requirement, not just the complaint.
            "holdout_registered": len(protocol.holdout_case_ids),
            "min_holdout": protocol.min_holdout,
            "target_pass_rate_pct": protocol.target_pass_rate_pct,
            "reasons": gaps or ["The protocol is registered but has not been scored yet."],
        }
