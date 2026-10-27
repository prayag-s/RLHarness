"""Gymnasium Environment Wrapper bridging DomainAdapter and RewardEvaluator."""

from __future__ import annotations
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_ide.contracts.adapter import DomainAdapter, StepResult, EpisodeLog
from rl_ide.reward.evaluator import RewardEvaluator


class AdapterGymEnv(gym.Env):
    """Gymnasium environment wrapping a DomainAdapter and RewardEvaluator.
    
    Supports hot-swapping the RewardEvaluator between bursts without resetting environment state.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, adapter: DomainAdapter, evaluator: RewardEvaluator):
        super().__init__()
        self.adapter = adapter
        self.evaluator = evaluator

        obs_spec = self.adapter.observations()
        action_spec = self.adapter.actions()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_spec.vector_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(action_spec.labels))

        self.current_episode_id = 0
        self.current_steps: list[StepResult] = []
        self.current_step_rewards: list[float] = []
        self.current_step_breakdowns: list[dict[str, float]] = []
        self.completed_episodes: list[EpisodeLog] = []
        self.on_episode_callback = None

    def set_evaluator(self, evaluator: RewardEvaluator) -> None:
        """Hot-swaps RewardEvaluator without re-initializing or resetting adapter state."""
        self.evaluator = evaluator

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_episode_id += 1
        self.current_steps.clear()
        self.current_step_rewards.clear()
        self.current_step_breakdowns.clear()

        obs_vector, info = self.adapter.reset(seed=seed)
        return np.array(obs_vector, dtype=np.float32), info

    def step(self, action: int | np.integer) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_result = self.adapter.step(int(action))
        reward, breakdown = self.evaluator.evaluate(self.current_episode_id, step_result.events)

        self.current_steps.append(step_result)
        self.current_step_rewards.append(reward)
        self.current_step_breakdowns.append(breakdown)

        info: dict[str, Any] = {
            "breakdown": breakdown,
            "events": step_result.events,
            **step_result.info,
        }

        terminated = bool(step_result.terminated)
        truncated = bool(step_result.truncated)

        if terminated or truncated:
            ep_log = EpisodeLog(
                episode_id=self.current_episode_id,
                steps=list(self.current_steps),
                total_reward=float(sum(self.current_step_rewards)),
                breakdown=self._aggregate_breakdown(self.current_step_breakdowns),
                step_rewards=list(self.current_step_rewards),
                step_breakdowns=list(self.current_step_breakdowns),
            )
            self.completed_episodes.append(ep_log)
            info["episode_log"] = ep_log
            if self.on_episode_callback:
                self.on_episode_callback(ep_log)

        return (
            np.array(step_result.obs_vector, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _aggregate_breakdown(self, step_breakdowns: list[dict[str, float]]) -> dict[str, float]:
        agg: dict[str, float] = {}
        for b in step_breakdowns:
            for k, v in b.items():
                agg[k] = agg.get(k, 0.0) + v
        return agg
