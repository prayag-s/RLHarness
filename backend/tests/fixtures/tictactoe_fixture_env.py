"""Deterministic Tic-Tac-Toe environment used by tests (test-only fixture, not runtime code).

Implements the synthesized-environment contract so the training/evaluation loop can be verified
against a real, learnable task instead of a fabricated placeholder.
"""

from __future__ import annotations
import random
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


class TicTacToeFixtureEnv(gym.Env):
    metadata = {"render_modes": ["json"]}
    action_labels = [f"Cell {i}" for i in range(9)]
    task_class = "control"
    episode_horizon = 9
    is_placeholder = False
    success_criteria = "Win or draw against a random opponent."

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.board = [0] * 9  # 1 = agent, -1 = opponent
        self.turn = 1
        self.done = False
        self.winner = 0
        self._rng = random.Random(1234)

    # -- helpers ---------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return np.asarray(self.board + [self.turn], dtype=np.float32)

    def _winner(self) -> int:
        for a, b, c in WIN_LINES:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        if all(v != 0 for v in self.board):
            return 2  # draw
        return 0

    def _empty(self) -> list[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def get_valid_actions(self) -> list[int]:
        return self._empty()

    def get_critical_actions(self) -> list[int]:
        for player in (1, -1):
            for line in WIN_LINES:
                vals = [self.board[i] for i in line]
                if vals.count(player) == 2 and vals.count(0) == 1:
                    cell = line[vals.index(0)]
                    return [cell]
        return []

    def render_arena_state(self) -> dict[str, Any]:
        return {
            "board": list(self.board),
            "current_player": int(self.turn),
            "winner": int(self.winner),
            "step_count": int(self.step_count),
        }

    def get_render_state(self) -> dict[str, Any]:
        return self.render_arena_state()

    # -- gym API ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self.board = [0] * 9
        self.turn = 1
        self.done = False
        self.winner = 0
        self.step_count = 0
        return self._obs(), {
            "is_success": False,
            "reward_breakdown": {},
            "arena_state": self.render_arena_state(),
        }

    def _opponent_move(self) -> None:
        empties = self._empty()
        if empties:
            self.board[self._rng.choice(empties)] = -1
        self.turn = 1

    def step(self, action: int):
        if not hasattr(self, "step_count"):
            self.step_count = 0
        self.step_count += 1
        action = int(action)
        breakdown: dict[str, float] = {}
        reward = 0.0

        if action not in self._empty():
            breakdown["illegal_move_penalty"] = -1.0
            reward -= 1.0
            self.done = True
            self.winner = -1
        else:
            self.board[action] = 1
            self.winner = self._winner()
            if self.winner == 0:
                self._opponent_move()
                self.winner = self._winner()

        terminated = self.winner != 0 or not self._empty()
        if terminated:
            self.done = True
            if self.winner == 1:
                reward += 10.0
                breakdown["win_bonus"] = 10.0
            elif self.winner == 2:
                reward += 3.0
                breakdown["draw_bonus"] = 3.0
            else:
                reward -= 10.0
                breakdown["loss_penalty"] = -10.0

        info = {
            "is_success": bool(self.winner in (1, 2)),
            "reward_breakdown": breakdown,
            "arena_state": self.render_arena_state(),
        }
        return self._obs(), float(reward), bool(terminated), False, info

    def compute_reward(self, obs, action, next_obs, info):
        bd = dict(info.get("reward_breakdown", {}))
        return float(sum(bd.values())), bd

    def interactive_step(self, payload: dict[str, Any], policy_predict_fn: Callable | None = None, mode: str = "agent_vs_env"):
        if isinstance(payload, dict) and payload.get("reset"):
            self.reset()
            if payload.get("human_first") is False and policy_predict_fn is not None:
                action_idx, probs = policy_predict_fn(self._obs())
                action_idx = int(action_idx)
                self.board[action_idx] = 1
                self.last_move = action_idx
                return {
                    "arena_state": self.render_arena_state(),
                    "agent_action_label": self.action_labels[action_idx],
                    "action_probs": [float(p) for p in probs],
                    "status_text": f"Agent opened at {action_idx}",
                }
            return {
                "arena_state": self.render_arena_state(),
                "agent_action_label": "Reset",
                "action_probs": [1.0 / 9] * 9,
                "status_text": "Reset",
            }

        human_move = int(payload.get("user_action", payload.get("cell", 0)))
        if human_move in self._empty():
            self.board[human_move] = -1
        self.winner = self._winner()
        probs = [1.0 / 9] * 9
        action_idx = 0
        if policy_predict_fn is not None and self.winner == 0:
            action_idx, probs = policy_predict_fn(self._obs())
            action_idx = int(action_idx)
            if action_idx in self._empty():
                self.board[action_idx] = 1
            self.winner = self._winner()
        self.last_move = action_idx
        return {
            "arena_state": self.render_arena_state(),
            "agent_action_label": self.action_labels[min(action_idx, 8)],
            "action_probs": [float(p) for p in probs],
            "status_text": "Played",
        }

    def get_demonstrations(self, num_samples: int = 200):
        rng = random.Random(7)
        X = np.zeros((num_samples, 10), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.int64)
        for i in range(num_samples):
            board = [0] * 9
            for _ in range(rng.randint(0, 4)):
                empties = [c for c, v in enumerate(board) if v == 0]
                if not empties:
                    break
                board[rng.choice(empties)] = 1
                empties = [c for c, v in enumerate(board) if v == 0]
                if empties:
                    board[rng.choice(empties)] = -1
            self.board = board
            move = self._teacher_move()
            X[i] = np.asarray(board + [1], dtype=np.float32)
            y[i] = move
        self.board = [0] * 9
        return X, y

    def _teacher_move(self) -> int:
        critical = self.get_critical_actions()
        if critical:
            return critical[0]
        if self.board[4] == 0:
            return 4
        for corner in (0, 2, 6, 8):
            if self.board[corner] == 0:
                return corner
        empties = self._empty()
        return empties[0] if empties else 0
