"""Data models for Conversational Intake Architect (v4.1)."""

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
    id: str = "q_objective"
    question_id: str = "q_objective"
    pillar: str = "objective"    # "objective" | "data" | "arena_ui" | "expectations" | "interaction"
    question: str                # Friendly 1-sentence question
    suggested_options: list[str] = Field(default_factory=list) # 3-4 clickable pills in the UI
    default_option: str = ""     # Used if user clicks "Use Smart Defaults"

    @model_validator(mode="before")
    @classmethod
    def normalize_question(cls, data: Any) -> Any:
        if isinstance(data, dict):
            qid = data.get("id") or data.get("question_id") or data.get("pillar") or "q_objective"
            data["id"] = qid
            data["question_id"] = qid
            if "pillar" not in data:
                data["pillar"] = qid
            if not data.get("suggested_options") and "options" in data:
                data["suggested_options"] = data["options"]
            if not data.get("default_option") and data.get("suggested_options"):
                data["default_option"] = data["suggested_options"][0]
        return data

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = self.question_id or self.pillar or "q_objective"
        if not self.question_id:
            self.question_id = self.id


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
    target_metric_threshold: float = 85.0 # e.g., 85.0 ( meaning 85% )
    max_auto_tune_rounds: int = 4
    episodes_per_round: int = 200
    initial_reward_components: list[dict[str, Any]] = Field(default_factory=list)


class IntakeResponse(BaseModel):
    """Response returned by the Intake Architect Agent."""
    status: str = "ready_to_build"  # "needs_clarification" | "ready_to_build"
    assistant_message: str = ""
    conversation_id: str | None = None
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    blueprint_spec: AgentBlueprintSpec | None = None
    recommended_training_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_auto_tune_rounds": 5,
            "episodes_per_round": 200,
            "target_metric_threshold": 90.0,
        }
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            data = {"assistant_message": str(data) if data else "", "status": "needs_clarification"}
        else:
            data = dict(data)
            msg = (
                data.get("assistant_message")
                or data.get("message")
                or data.get("assistant_response")
                or data.get("response")
                or data.get("summary")
                or ""
            )
            data["assistant_message"] = str(msg) if msg is not None else ""
            if not data.get("status"):
                data["status"] = "ready_to_build" if (data.get("blueprint_spec") or data.get("blueprint")) else "needs_clarification"
            if "questions" not in data and "clarifying_questions" in data:
                data["questions"] = data["clarifying_questions"]
            if "blueprint_spec" not in data and "blueprint" in data:
                data["blueprint_spec"] = data["blueprint"]
            if "recommended_training_params" not in data or not data["recommended_training_params"]:
                data["recommended_training_params"] = {
                    "max_auto_tune_rounds": 5,
                    "episodes_per_round": 200,
                    "target_metric_threshold": 90.0,
                }
        return data


from enum import Enum
from typing import List, Optional


class ExecutionTarget(str, Enum):
    REWARD_SHAPING = "reward_shaping"                 # Modifies RewardDSL / compute_reward (Training incentive)
    INFERENCE_STRATEGY = "inference_strategy"         # Modifies action selection in interactive_step() / policy wrapper (Arena/Eval behavior)
    ENV_SIM_MECHANICS = "env_sim_mechanics"           # Modifies reset() / step() / opponent logic in env_module.py (Training + Arena physics/rules)
    TRAINING_HYPERPARAMS = "training_hyperparams"     # Modifies entropy coef, training burst length, or stochastic exploration
    UNSUPPORTED_CAPABILITY = "unsupported_capability" # Capability not built or impossible in current runtime


class ChatActionPlan(BaseModel):
    supported: bool = Field(
        description="True if the platform can execute this accurately; False if the capability is not built."
    )
    unsupported_reason: Optional[str] = Field(
        default=None,
        description="If supported=False, clear explanation of what capability is missing."
    )
    scope_classification: str = Field(
        description="'arena_only' | 'training_only' | 'both' | 'unsupported'"
    )
    targets: List[ExecutionTarget] = Field(
        description="One or more execution layers required to fulfill the user prompt completely."
    )
    requires_retrain: bool = Field(
        description="True if reward/env/training changes require a PPO training burst; False if inference-only."
    )
    surgical_plan: str = Field(
        description="Exact functions/lines in env_module.py, adapter, or trainer to modify."
    )
    user_facing_summary: str = Field(
        description="Clear, concise response explaining the exact changes applied (or why it is unsupported)."
    )

