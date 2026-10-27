"""Data models for Compiler, Task Configurations, and Critique Diffs."""

from __future__ import annotations
from typing import Any
import yaml
from pydantic import BaseModel, Field
from rl_ide.reward.dsl import RewardDSL


class TaskConfig(BaseModel):
    """Configuration specification for a training task."""
    task_id: str
    name: str = "Untitled Task"
    description: str = ""
    adapter_name: str
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=lambda: {
        "learning_rate": 0.0003,
        "gamma": 0.90,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 10,
    })
    reward_dsl: RewardDSL

    @classmethod
    def from_yaml(cls, yaml_content: str) -> TaskConfig:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            raise ValueError("YAML content must parse to a dictionary")
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.dump(self.model_dump(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskConfig:
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TermDiff(BaseModel):
    """Represents a change to a single reward term."""
    action: str  # "added" | "modified" | "removed"
    term_name: str
    old_term: dict[str, Any] | None = None
    new_term: dict[str, Any] | None = None
    rationale: str = ""

    @property
    def new_weight(self) -> float:
        if self.new_term and "value" in self.new_term:
            try:
                return float(self.new_term["value"])
            except (ValueError, TypeError):
                pass
        return 0.0


class ConflictWarning(BaseModel):
    """Detected contradiction or regression conflict."""
    term_name: str
    message: str
    severity: str = "warning"  # "warning" | "error"


class Stage1Preview(BaseModel):
    """Counterfactual re-scoring preview without retraining."""
    target_episode_id: int
    before_score: float
    after_score: float
    score_delta: float
    recent_scores_before: list[float] = Field(default_factory=list)
    recent_scores_after: list[float] = Field(default_factory=list)
    summary: str = ""


class Stage2Preview(BaseModel):
    """Speculative forked policy behavioral preview."""
    target_episode_id: int
    target_action_before: str
    target_action_after: str
    target_flipped: bool
    total_episodes_evaluated: int = 0
    total_actions_flipped: int = 0
    flip_rate: float = 0.0
    locked_episodes_preserved: float = 1.0  # 1.0 = 100%
    summary: str = ""


class RewardDiffProposal(BaseModel):
    """Structured proposal generated from human critique."""
    new_dsl: RewardDSL
    diff_yaml: str = ""
    term_diffs: list[TermDiff] = Field(default_factory=list)
    rationale: dict[str, str] = Field(default_factory=dict)
    conflicts: list[ConflictWarning] = Field(default_factory=list)
    stage1_counterfactual_preview: Stage1Preview | None = None
    stage2_forked_behavior_preview: Stage2Preview | None = None
