"""Reward DSL schema and serialization."""

from __future__ import annotations
from typing import Any
import yaml
from pydantic import BaseModel, Field, model_validator


class HorizonConfig(BaseModel):
    gamma: float = 0.90
    critic_warmup_epochs_on_diff: int = 15
    kl_coef: float = 0.1


class RewardTerm(BaseModel):
    name: str
    when: str | bool = True
    value: float | int | str = 0.0

    def __init__(
        self,
        name: str,
        when: str | bool | None = None,
        value: float | int | str | None = None,
        expression: str | None = None,
        weight: float | None = None,
        **kwargs: Any,
    ):
        final_when = when if when is not None else (expression if expression is not None else True)
        final_val = value if value is not None else (weight if weight is not None else 0.0)
        super().__init__(name=name, when=final_when, value=final_val, **kwargs)

    @property
    def expression(self) -> str:
        return str(self.when)

    @property
    def weight(self) -> float:
        try:
            return float(self.value)
        except (ValueError, TypeError):
            return 1.0


class PinnedEpisode(BaseModel):
    episode_id: int
    forbid_action: str
    penalty: float = -10.0


class RewardConfig(BaseModel):
    terms: list[RewardTerm] = Field(default_factory=list)
    pinned_episodes: list[PinnedEpisode] = Field(default_factory=list)
    aggregate: str = "sum"  # "sum" | "mean"
    clamp: list[float] | tuple[float, float] = [-100.0, 100.0]


class RewardDSL(BaseModel):
    version: str = "1.0"
    horizon: HorizonConfig = Field(default_factory=HorizonConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        
        # If 'reward' is a dict, but 'aggregate' or 'clamp' were placed at top-level
        reward_dict = data.get("reward")
        if isinstance(reward_dict, dict):
            if "aggregate" in data and "aggregate" not in reward_dict:
                reward_dict["aggregate"] = data["aggregate"]
            if "clamp" in data and "clamp" not in reward_dict:
                reward_dict["clamp"] = data["clamp"]
            if "terms" in data and "terms" not in reward_dict:
                reward_dict["terms"] = data["terms"]
            if "pinned_episodes" in data and "pinned_episodes" not in reward_dict:
                reward_dict["pinned_episodes"] = data["pinned_episodes"]
        elif "terms" in data:
            # Flat format: 'terms' at top level
            data["reward"] = {
                "terms": data.get("terms", []),
                "pinned_episodes": data.get("pinned_episodes", []),
                "aggregate": data.get("aggregate", "sum"),
                "clamp": data.get("clamp", [-100.0, 100.0]),
            }

        return data

    @classmethod
    def from_yaml(cls, yaml_content: str) -> RewardDSL:
        parsed = yaml.safe_load(yaml_content)
        if not isinstance(parsed, dict):
            raise ValueError("YAML content must parse to a dictionary")
        return cls.model_validate(parsed)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardDSL:
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_yaml(self) -> str:
        return yaml.dump(self.model_dump(), sort_keys=False)
