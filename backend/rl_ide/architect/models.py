"""Data models for Conversational Intake Architect (v4.0)."""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field, model_validator

# 5 Pillars constants
PILLAR_OBJECTIVE = "objective"
PILLAR_DATA = "data"
PILLAR_ARENA_UI = "arena_ui"
PILLAR_EXPECTATIONS = "expectations"
PILLAR_INTERACTION = "interaction"


class ClarifyingQuestion(BaseModel):
    """A concise, high-impact question targeting an underspecified Pillar."""
    pillar: str = "objective"    # "objective" | "data" | "arena_ui" | "expectations" | "interaction"
    question: str                # Friendly 1-sentence question
    suggested_options: list[str] = Field(default_factory=list) # 3-4 clickable pills in the UI
    default_option: str = ""     # Used if user clicks "Use Smart Defaults"
    question_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_question(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "pillar" not in data and "question_id" in data:
                data["pillar"] = data["question_id"]
            if not data.get("suggested_options") and "options" in data:
                data["suggested_options"] = data["options"]
            if not data.get("default_option") and data.get("suggested_options"):
                data["default_option"] = data["suggested_options"][0]
        return data

    def model_post_init(self, __context: Any) -> None:
        if not self.question_id:
            self.question_id = self.pillar


class AgentBlueprintSpec(BaseModel):
    """The locked architectural blueprint required for JIT code & arena synthesis."""
    task_title: str
    domain_summary: str
    data_source_type: str        # "pure_simulation" | "uploaded_dataset" | "hybrid_doc_and_sim"
    dataset_id: str | None = None
    observation_features: list[str]
    action_labels: list[str]
    arena_visual_description: str
    interactive_mode_description: str
    success_metric_name: str     # e.g., "Non-Loss Rate (%)", "Target Docking Rate (%)", "Audit Accuracy (%)"
    target_metric_threshold: float # e.g., 85.0 ( meaning 85% )
    max_auto_tune_rounds: int = 4
    initial_reward_components: list[dict[str, Any]] = Field(default_factory=list)


class IntakeResponse(BaseModel):
    """Response returned by the Intake Architect Agent."""
    status: str                  # "needs_clarification" | "ready_to_build"
    assistant_message: str = ""
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    blueprint_spec: AgentBlueprintSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "assistant_message" not in data:
                data["assistant_message"] = (
                    data.get("message")
                    or data.get("assistant_response")
                    or data.get("response")
                    or data.get("summary")
                    or ""
                )
            if "questions" not in data and "clarifying_questions" in data:
                data["questions"] = data["clarifying_questions"]
            if "blueprint_spec" not in data and "blueprint" in data:
                data["blueprint_spec"] = data["blueprint"]
        return data
