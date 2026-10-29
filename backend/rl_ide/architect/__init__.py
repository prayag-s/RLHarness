"""Conversational Intake Architect Package for RL-Native Agent IDE v4.3."""

from rl_ide.architect.models import (
    ClarifyingQuestion,
    AgentBlueprintSpec,
    DynamicTaskBlueprint,
    ObservationFeatureSpec,
    ActionSpec,
    ExecutionTarget,
    ChatActionPlan,
    IntakeResponse,
    PILLAR_OBJECTIVE,
    PILLAR_DATA,
    PILLAR_ARENA_UI,
    PILLAR_EXPECTATIONS,
    PILLAR_INTERACTION,
)
from rl_ide.architect.intake import (
    IntakeArchitectAgent,
    ARCHITECT_SYSTEM_PROMPT,
    synthesize_blueprint_from_prompt,
)

__all__ = [
    "ClarifyingQuestion",
    "AgentBlueprintSpec",
    "DynamicTaskBlueprint",
    "ObservationFeatureSpec",
    "ActionSpec",
    "ExecutionTarget",
    "ChatActionPlan",
    "IntakeResponse",
    "IntakeArchitectAgent",
    "StudioOmniAgent",
    "StudioActionType",
    "StudioChatRequest",
    "StudioChatResponse",
    "ARCHITECT_SYSTEM_PROMPT",
    "synthesize_blueprint_from_prompt",
    "PILLAR_OBJECTIVE",
    "PILLAR_DATA",
    "PILLAR_ARENA_UI",
    "PILLAR_EXPECTATIONS",
    "PILLAR_INTERACTION",
]


def __getattr__(name: str):
    if name in ("StudioOmniAgent", "StudioActionType", "StudioChatRequest", "StudioChatResponse"):
        import rl_ide.architect.omni_agent as _oa
        return getattr(_oa, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
