"""Data models for Conversational Intake Architect and Agentic IDE (v4.3)."""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
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


class ObservationFeatureSpec(BaseModel):
    """Specification of an observation vector feature."""
    name: str = Field(description="snake_case feature name exposed in obs vector and render_state")
    description: str = ""
    low: float = -1.0
    high: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def normalize_spec(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data, "description": data, "low": -1.0, "high": 1.0}
        return data


class ActionSpec(BaseModel):
    """Specification of a discrete action."""
    index: int = 0
    label: str = Field(description="Human-readable action button label for the Arena UI")
    mechanic_description: str = Field(default="", description="Exact effect of this action in step() and interactive_step()")

    @model_validator(mode="before")
    @classmethod
    def normalize_spec(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"index": 0, "label": data, "mechanic_description": data}
        return data


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
    state_dict_schema: dict[str, str] = Field(default_factory=dict)
    visual_metaphor: str = ""
    actions: list[ActionSpec] = Field(default_factory=list)
    observation_feature_specs: list[ObservationFeatureSpec] = Field(default_factory=list)
    label_column: str | None = None  # Optional target column for dataset-backed tasks
    # "one_shot" = each step is an independent decision (contextual bandit).
    # "sequential" = an episode is one case that unfolds over `episode_length` decisions, the state
    # carries across steps and the reward arrives only at the end (G10).
    task_structuring: str = "one_shot"
    episode_length: int = 1
    # Declarative "never X when feature >= N" rules. Kept as plain dicts so the blueprint stays
    # serialisable; the environment and exporter validate them into HardConstraint.
    hard_constraints: list[dict[str, Any]] = Field(default_factory=list)
    pending_directive: str | None = None  # Latest mechanics/rule instruction for the synthesizer

    def to_dynamic_task_blueprint(self) -> DynamicTaskBlueprint:
        return DynamicTaskBlueprint.from_agent_blueprint_spec(self)


class DynamicTaskBlueprint(BaseModel):
    """Dynamic, domain-agnostic task blueprint matching the Universal Generative Pipeline specification."""
    domain_title: str
    domain_summary: str
    visual_metaphor: str = Field(
        description="Detailed description of what the bespoke Arena UI should look like (e.g., 'Connect 4 7x6 vertical drop grid', 'Elevator shaft building cross-section', 'Casino table', 'Orderbook depth ladder')"
    )
    observation_features: List[ObservationFeatureSpec]
    actions: List[ActionSpec]
    state_dict_schema: Dict[str, str] = Field(
        default_factory=dict,
        description="Exact JSON-serializable keys emitted by get_render_state() so env_builder and arena_builder share a strict contract"
    )
    reward_components: Dict[str, str] = Field(default_factory=dict)
    clarifying_questions: List[ClarifyingQuestion] = Field(default_factory=list)
    data_source_type: str = "pure_simulation"
    dataset_id: Optional[str] = None
    label_column: Optional[str] = None
    success_metric_name: str = "Task Success Rate (%)"
    target_metric_threshold: float = 85.0
    max_auto_tune_rounds: int = 4
    episodes_per_round: int = 200
    task_structuring: str = "one_shot"
    episode_length: int = 1
    hard_constraints: list[dict[str, Any]] = Field(default_factory=list)

    def to_agent_blueprint_spec(self) -> AgentBlueprintSpec:
        reward_list = []
        for k, v in self.reward_components.items():
            reward_list.append({"name": k, "weight": 1.0, "description": str(v)})
        return AgentBlueprintSpec(
            task_title=self.domain_title,
            domain_summary=self.domain_summary,
            data_source_type=self.data_source_type,
            dataset_id=self.dataset_id,
            observation_features=[f.name for f in self.observation_features],
            action_labels=[a.label for a in self.actions],
            arena_visual_description=self.visual_metaphor,
            interactive_mode_description=f"Interactive arena testing {self.domain_title}",
            success_metric_name=self.success_metric_name,
            target_metric_threshold=self.target_metric_threshold,
            max_auto_tune_rounds=self.max_auto_tune_rounds,
            episodes_per_round=self.episodes_per_round,
            initial_reward_components=reward_list,
            state_dict_schema=self.state_dict_schema,
            visual_metaphor=self.visual_metaphor,
            actions=self.actions,
            observation_feature_specs=self.observation_features,
            label_column=self.label_column,
            task_structuring=self.task_structuring,
            episode_length=self.episode_length,
            hard_constraints=list(self.hard_constraints or []),
        )

    @classmethod
    def from_agent_blueprint_spec(cls, spec: AgentBlueprintSpec) -> DynamicTaskBlueprint:
        obs_specs = spec.observation_feature_specs or [
            ObservationFeatureSpec(name=f, description=f, low=-1.0, high=1.0)
            for f in spec.observation_features
        ]
        action_specs = spec.actions or [
            ActionSpec(index=i, label=lbl, mechanic_description=f"Execute action {lbl}")
            for i, lbl in enumerate(spec.action_labels)
        ]
        reward_dict = {}
        for r in spec.initial_reward_components:
            reward_dict[r.get("name", "reward")] = str(r.get("description", r.get("weight", "1.0")))
        return cls(
            domain_title=spec.task_title,
            domain_summary=spec.domain_summary,
            visual_metaphor=spec.visual_metaphor or spec.arena_visual_description or "",
            observation_features=obs_specs,
            actions=action_specs,
            state_dict_schema=spec.state_dict_schema or {},
            reward_components=reward_dict,
            clarifying_questions=[],
            label_column=spec.label_column,
            success_metric_name=spec.success_metric_name,
            target_metric_threshold=spec.target_metric_threshold,
            max_auto_tune_rounds=spec.max_auto_tune_rounds,
            episodes_per_round=spec.episodes_per_round,
            task_structuring=spec.task_structuring,
            episode_length=spec.episode_length,
            hard_constraints=list(spec.hard_constraints or []),
        )


class IntakeResponse(BaseModel):
    """Response returned by the Intake Architect Agent."""
    status: str = "ready_to_build"  # "needs_clarification" | "ready_to_build"
    assistant_message: str = ""
    conversation_id: str | None = None
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    blueprint_spec: AgentBlueprintSpec | None = None
    blueprint: DynamicTaskBlueprint | None = None
    recommended_training_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_auto_tune_rounds": 5,
            "episodes_per_round": 200,
            "target_metric_threshold": 90.0,
        }
    )

    @property
    def clarifying_questions(self) -> list[ClarifyingQuestion]:
        return self.questions

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

            # Synchronize blueprint_spec and blueprint
            if "blueprint" in data and ("blueprint_spec" not in data or data["blueprint_spec"] is None):
                bp = data["blueprint"]
                if isinstance(bp, DynamicTaskBlueprint):
                    data["blueprint_spec"] = bp.to_agent_blueprint_spec()
                elif isinstance(bp, dict):
                    data["blueprint_spec"] = DynamicTaskBlueprint.model_validate(bp).to_agent_blueprint_spec()
            elif "blueprint_spec" in data and ("blueprint" not in data or data["blueprint"] is None) and data["blueprint_spec"] is not None:
                bs = data["blueprint_spec"]
                if isinstance(bs, AgentBlueprintSpec):
                    data["blueprint"] = bs.to_dynamic_task_blueprint()
                elif isinstance(bs, dict):
                    data["blueprint"] = AgentBlueprintSpec.model_validate(bs).to_dynamic_task_blueprint()

            if "recommended_training_params" not in data or not data["recommended_training_params"]:
                data["recommended_training_params"] = {
                    "max_auto_tune_rounds": 5,
                    "episodes_per_round": 200,
                    "target_metric_threshold": 90.0,
                }
        return data


class ExecutionTarget(str, Enum):
    REANCHOR_DOMAIN = "reanchor_domain"               # User pivoted to a completely new task/domain or current blueprint is wrong
    ENV_SIM_MECHANICS = "env_sim_mechanics"           # Patch reset(), step(), or opponent logic in env_module.py
    REWARD_SHAPING = "reward_shaping"                 # Patch compute_reward() / RewardDSL weights and rules
    INFERENCE_STRATEGY = "inference_strategy"         # Patch action selection inside interactive_step() (e.g. randomizing opening move among good options)
    ARENA_VISUAL_UI = "arena_visual_ui"               # Re-synthesize or patch the HTML/JS visual arena
    TRAINING_HYPERPARAMS = "training_hyperparams"     # Adjust PPO burst, entropy, or exploration settings
    UNSUPPORTED_CAPABILITY = "unsupported_capability" # Honest refusal when runtime genuinely lacks the capability


class ChatActionPlan(BaseModel):
    supported: bool = Field(
        description="True if executable within the platform; False if capability is not built."
    )
    unsupported_reason: Optional[str] = Field(
        default=None,
        description="If supported=False, clear explanation of what capability is missing."
    )
    targets: List[ExecutionTarget] = Field(
        description="Ordered list of architectural layers to modify."
    )
    requires_retrain: bool = Field(
        default=False,
        description="True if reward/env changes require a PPO training burst."
    )
    surgical_instructions: str = Field(
        default="",
        description="Exact code/UI modifications to perform across the selected targets."
    )
    surgical_plan: Optional[str] = Field(
        default=None,
        description="Alias for surgical_instructions."
    )
    scope_classification: Optional[str] = Field(
        default=None,
        description="'arena_only' | 'training_only' | 'both' | 'unsupported'"
    )
    user_facing_summary: str = Field(
        default="",
        description="Direct, helpful explanation of what was changed (or why it is unsupported)."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_plan(cls, data: Any) -> Any:
        if isinstance(data, dict):
            inst = data.get("surgical_instructions") or data.get("surgical_plan") or ""
            data["surgical_instructions"] = inst
            data["surgical_plan"] = inst
            if not data.get("scope_classification"):
                if not data.get("supported", True):
                    data["scope_classification"] = "unsupported"
                else:
                    targets = data.get("targets", [])
                    has_train = any(
                        t in (
                            ExecutionTarget.REWARD_SHAPING,
                            ExecutionTarget.ENV_SIM_MECHANICS,
                            ExecutionTarget.TRAINING_HYPERPARAMS,
                            "reward_shaping",
                            "env_sim_mechanics",
                            "training_hyperparams",
                        )
                        for t in targets
                    )
                    has_arena = any(
                        t in (
                            ExecutionTarget.INFERENCE_STRATEGY,
                            ExecutionTarget.ARENA_VISUAL_UI,
                            "inference_strategy",
                            "arena_visual_ui",
                        )
                        for t in targets
                    )
                    if has_train and has_arena:
                        data["scope_classification"] = "both"
                    elif has_train:
                        data["scope_classification"] = "training_only"
                    else:
                        data["scope_classification"] = "arena_only"
        return data
