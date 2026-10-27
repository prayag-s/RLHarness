"""Domain Adapter contracts and data models."""

from typing import Protocol, Any, runtime_checkable
from pydantic import BaseModel, Field


class ObservationSpec(BaseModel):
    vector_dim: int
    feature_names: list[str]
    horizon_type: str  # "stateful_mdp" | "contextual_bandit"
    default_gamma: float  # 0.90 for stateful stream, 0.0 for independent records


class ActionSpec(BaseModel):
    action_type: str  # "discrete"
    labels: list[str]  # e.g. ["normal", "anomaly"]


class StepResult(BaseModel):
    obs_vector: list[float]
    events: dict[str, Any]  # {"obs": {...}, "action": {...}, "state": {...}, "truth": {...}}
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)


class EpisodeLog(BaseModel):
    episode_id: int
    steps: list[StepResult] = Field(default_factory=list)
    total_reward: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    step_rewards: list[float] = Field(default_factory=list)
    step_breakdowns: list[dict[str, float]] = Field(default_factory=list)
    locked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class DomainAdapter(Protocol):
    def observations(self) -> ObservationSpec:
        ...

    def actions(self) -> ActionSpec:
        ...

    def event_schema(self) -> dict[str, str]:
        """Returns {field_path: type_str} of all variables legal in Reward DSL."""
        ...

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        ...

    def step(self, action_idx: int) -> StepResult:
        ...

    def render_frame(
        self,
        episode_id: int,
        step_result: StepResult,
        total_reward: float,
        breakdown: dict[str, float]
    ) -> dict[str, Any]:
        ...
