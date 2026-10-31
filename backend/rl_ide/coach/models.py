"""Journey Coach contract: one finding, one next action.

The coach is deliberately separate from the verifier (was the change good?) and Crossfire (can
the claim be made?). It reviews the artefacts a user supplies *before* they are spent on a run,
and interprets the evidence *after* one finishes.

A finding is only useful if it changes what the user does next, so every item carries an
executable action. The LLM may add prioritisation and prose, but it may never add a number: all
evidence is computed deterministically before the model is called.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CoachStage(str, Enum):
    OBJECTIVE = "objective"
    DATA = "data"
    PARAMETERS = "parameters"
    PREVIEW = "preview"
    RESULTS = "results"
    EXPORT = "export"


class Severity(str, Enum):
    BLOCKER = "blocker"
    WARNING = "warning"
    TIP = "tip"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.BLOCKER: 0,
    Severity.WARNING: 1,
    Severity.TIP: 2,
}

# The only actions the UI knows how to execute. A finding that cannot name one of these is a
# lecture, so the judge is told to return nothing rather than invent a kind.
ActionKind = Literal[
    "apply_params",
    "use_objective",
    "send_message",
    "open_guide",
    "run_episodes",
]


class GuidanceAction(BaseModel):
    kind: ActionKind
    label: str
    payload: dict[str, Any] = Field(default_factory=dict)


class GuidanceItem(BaseModel):
    rule_id: str
    stage: CoachStage
    severity: Severity
    title: str
    why: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    actions: list[GuidanceAction] = Field(default_factory=list)
    source: Literal["rule", "llm"] = "rule"
    status: Literal["open", "acknowledged", "resolved"] = "open"

    def fingerprint(self) -> str:
        """Stable id for "the same finding on the same evidence".

        Re-running an evaluation must update a row, not duplicate it, and an acknowledged
        warning must stay acknowledged until the artefact (the evidence) actually changes.
        """
        payload = json.dumps(
            {
                "rule_id": self.rule_id,
                "stage": self.stage.value,
                "evidence": self.evidence,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class CoachReport(BaseModel):
    stage: CoachStage | None = None  # None means a multi-stage preflight
    status: Literal["ok", "attention", "blocked"] = "ok"
    items: list[GuidanceItem] = Field(default_factory=list)
    conversation_id: str | None = None
    evaluated_at: str = Field(default_factory=utc_now_iso)
    llm_used: bool = False
    llm_skipped_reason: str | None = None

    def model_post_init(self, __context: Any) -> None:
        self.status = self.derive_status(self.items)

    @staticmethod
    def derive_status(items: list[GuidanceItem]) -> str:
        if any(i.severity == Severity.BLOCKER and i.status != "acknowledged" for i in items):
            return "blocked"
        if any(i.severity == Severity.WARNING and i.status != "acknowledged" for i in items):
            return "attention"
        return "ok"

    @property
    def blockers(self) -> list[GuidanceItem]:
        return [
            i
            for i in self.items
            if i.severity == Severity.BLOCKER and i.status != "acknowledged"
        ]

    @property
    def open_items(self) -> list[GuidanceItem]:
        return [i for i in self.items if i.status == "open"]

    def refresh(self) -> "CoachReport":
        self.status = self.derive_status(self.items)
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CoachBlocked(Exception):
    """Raised by an execution choke point when an unresolved blocker remains."""

    def __init__(self, report: CoachReport):
        self.report = report
        titles = "; ".join(i.title for i in report.blockers) or "unresolved blocker"
        super().__init__(f"Journey Coach blocked this action: {titles}")
