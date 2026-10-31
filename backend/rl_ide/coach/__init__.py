"""Journey Coach: evaluate inputs and interpretations, not just outcomes.

The verifier decides whether a change was good; Crossfire decides whether a claim is earned. The
coach answers the question that comes earlier and later: *is this the right thing to attempt, and
does the user understand what they got?*
"""

from rl_ide.coach.models import (
    CoachBlocked,
    CoachReport,
    CoachStage,
    GuidanceAction,
    GuidanceItem,
    Severity,
)
from rl_ide.coach.service import JourneyCoach, blockers_mode, get_coach
from rl_ide.coach.store import CoachStore, get_coach_store

__all__ = [
    "CoachBlocked",
    "CoachReport",
    "CoachStage",
    "CoachStore",
    "GuidanceAction",
    "GuidanceItem",
    "JourneyCoach",
    "Severity",
    "blockers_mode",
    "get_coach",
    "get_coach_store",
]
