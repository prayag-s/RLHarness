"""JIT Environment Synthesizer for RL-Native Agent IDE v4.0.

Synthesizes self-contained Gymnasium environment modules (env_module.py)
tailored to any AgentBlueprintSpec and IngestedDataset.
"""

from __future__ import annotations
import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Callable
import gymnasium as gym

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.architect.intake import is_tictactoe_spec
from rl_ide.ingest.loader import IngestedDataset
from rl_ide.llm.client import BaseLLMClient, create_llm_client, FallbackCodeSynthesizer


ENV_BUILDER_SYSTEM_PROMPT = """You are an Expert RL Environment Engineer. Given an `AgentBlueprintSpec` (and optional `DatasetProfile`), write a single, self-contained, bug-free Python module defining `SynthesizedGymEnv(gymnasium.Env)`.

STRICT ENGINEERING REQUIREMENTS:
1. `self.observation_space` MUST be `spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)`. Always return `np.asarray(obs, dtype=np.float32)` with exact shape `(obs_dim,)` and no NaNs/Infs.
2. `self.action_space` MUST be `spaces.Discrete(n_actions)`.
3. Keep `compute_reward(self, state, action, next_state, info)` modular, returning `(float(total_reward), dict_of_term_floats)`.
4. Every `step()` MUST set `info["reward_breakdown"]`, `info["is_success"]` (boolean indicating whether the step/episode achieved the user's goal), and `info["arena_state"] = self.render_arena_state()`.
5. Implement `get_demonstrations(self, num_samples=200)` returning `(X_obs, y_actions)` using sensible domain heuristics so Warm-Start Behavioral Cloning gives the agent an immediate head start.
6. Implement `interactive_step(self, payload, policy_predict_fn)` so when a human clicks or interacts in the UI, it updates the state, queries `action_idx, action_probs = policy_predict_fn(obs)`, applies the agent's response, and returns a JSON dict with `arena_state`, `agent_action_label`, `action_probs`, and `status_text`.
7. Allowed imports ONLY: `math`, `random`, `numpy`, `pandas`, `gymnasium`, `typing`, `copy`, `collections`. Never use network calls, `subprocess`, `os.system`, or external game libraries like `pygame`.
"""


def load_synthesized_env_class(file_path: Path | str, class_name: str = "SynthesizedGymEnv") -> type[gym.Env]:
    """Dynamically loads the synthesized Gymnasium environment class from file."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Synthesized environment module not found at: {path}")

    unique_mod_name = f"synthesized_env_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(unique_mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not construct module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_mod_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"Module {path} does not export expected class '{class_name}'")

    return getattr(module, class_name)


class JITEnvironmentSynthesizer:
    """Synthesizes custom Gym environments tailored to an AgentBlueprintSpec."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self.client = llm_client or create_llm_client()

    def synthesize(
        self,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None = None,
        session_id: str | None = None,
        workspace_dir: str | Path | None = None,
    ) -> tuple[str, Path, type[gym.Env]]:
        """Generates, saves, and dynamically loads env_module.py."""
        # Determine output location
        sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
        if workspace_dir:
            out_dir = Path(workspace_dir)
        else:
            out_dir = Path("workspaces") / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "env_module.py"

        # Generate code string
        code_str = self.generate_env_code(blueprint, dataset)

        # Audit AST security before writing or executing module
        from rl_ide.synthesizer.verifier import audit_ast_security
        audit_ast_security(code_str)

        # Write to file
        file_path.write_text(code_str, encoding="utf-8")

        # Dynamically load the class
        env_cls = load_synthesized_env_class(file_path)
        return code_str, file_path, env_cls

    def generate_env_code(
        self,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None = None,
    ) -> str:
        """Synthesizes python code via live LLM or deterministic fallback."""
        if not isinstance(self.client, FallbackCodeSynthesizer):
            try:
                code = self._call_llm_code_gen(blueprint, dataset)
                if code:
                    return code
            except Exception as exc:
                raise RuntimeError(f"LLM Gymnasium environment synthesis failed: {exc}")

        # Deterministic Code Generation (used for offline Fallback in test mode)
        return self._generate_deterministic_env_code(blueprint, dataset)

    def _call_llm_code_gen(
        self,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None,
    ) -> str | None:
        prompt = (
            f"Synthesize the complete Python module `env_module.py` for this AgentBlueprintSpec:\n\n"
            f"{blueprint.model_dump_json(indent=2)}\n\n"
        )
        if dataset:
            sample_data = dataset.df.head(5).to_dict(orient="records") if hasattr(dataset, "df") else []
            prompt += f"Dataset Schema & Sample Records:\n{json.dumps(sample_data, default=str)}\n"

        prompt += "\nOutput ONLY valid Python code inside a ```python ... ``` block."
        raw = self.client.generate(prompt, system_prompt=ENV_BUILDER_SYSTEM_PROMPT)

        match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", raw)
        return match.group(1).strip() if match else raw.strip()

    def _generate_deterministic_env_code(
        self,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None,
    ) -> str:
        """Builds a high-performance, robust SynthesizedGymEnv Python module."""
        task_title = blueprint.task_title
        is_tictactoe = is_tictactoe_spec(blueprint)
        is_dataset = blueprint.data_source_type == "uploaded_dataset" or dataset is not None

        obs_dim = len(blueprint.observation_features)
        n_actions = len(blueprint.action_labels)
        action_labels = blueprint.action_labels
        obs_features = blueprint.observation_features

        if is_tictactoe:
            return self._build_tictactoe_code(blueprint)
        elif is_dataset:
            return self._build_dataset_code(blueprint, dataset)
        else:
            return self._build_general_sim_code(blueprint)

    def _build_tictactoe_code(self, blueprint: AgentBlueprintSpec) -> str:
        """Generates complete Tic-Tac-Toe Gym environment with self-play heuristics."""
        reward_terms = {t["name"]: float(t.get("weight", 1.0)) for t in blueprint.initial_reward_components}
        win_bonus = reward_terms.get("win_bonus", 10.0)
        draw_bonus = reward_terms.get("draw_bonus", 3.0)
        loss_penalty = reward_terms.get("loss_penalty", -10.0)
        illegal_penalty = reward_terms.get("illegal_move_penalty", -5.0)

        return f'''"""Auto-synthesized Gymnasium environment for Tic-Tac-Toe."""

from __future__ import annotations
import math
import random
from typing import Any, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SynthesizedGymEnv(gym.Env):
    """Auto-synthesized Gymnasium environment for: {blueprint.task_title}"""
    metadata = {{"render_modes": ["json"]}}

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.action_labels = {blueprint.action_labels}
        self.observation_features = {blueprint.observation_features}
        
        # 9 board cells (-1, 0, 1) + 1 turn indicator (+1=agent, -1=opponent)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        self.board = np.zeros(9, dtype=np.int8)
        self.current_turn = 1  # 1 = agent (X), -1 = opponent (O)
        self.step_count = 0
        self.game_status = "in_progress"
        self.winner = None
        self.last_move = None

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[:9] = self.board.astype(np.float32)
        obs[9] = float(self.current_turn)
        return obs

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board.fill(0)
        self.current_turn = 1
        self.step_count = 0
        self.game_status = "in_progress"
        self.winner = None
        self.last_move = None
        return self._get_obs(), {{"arena_state": self.render_arena_state()}}

    def _check_winner(self) -> int | None:
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for a, b, c in lines:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return int(self.board[a])
        if np.all(self.board != 0):
            return 0  # Draw
        return None

    def compute_reward(self, state: dict, action: int, next_state: dict, info: dict) -> tuple[float, dict[str, float]]:
        """Modular reward function. Rewritten across rounds by Autonomous Auto-Tuner."""
        breakdown = {{}}
        total = 0.0

        if info.get("is_illegal", False):
            breakdown["illegal_move_penalty"] = {illegal_penalty}
            total += {illegal_penalty}
            return float(total), breakdown

        winner = info.get("winner")
        if winner == 1:
            breakdown["win_bonus"] = {win_bonus}
            total += {win_bonus}
        elif winner == -1:
            breakdown["loss_penalty"] = {loss_penalty}
            total += {loss_penalty}
        elif winner == 0:
            breakdown["draw_bonus"] = {draw_bonus}
            total += {draw_bonus}
        else:
            breakdown["step_cost"] = -0.05
            total += -0.05

        return float(total), breakdown

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        self.step_count += 1
        self.last_move = action

        state_dict = {{"board": self.board.copy(), "turn": self.current_turn}}
        info: dict[str, Any] = {{"is_illegal": False, "is_success": False}}

        # Illegal move check
        if self.board[action] != 0:
            info["is_illegal"] = True
            info["winner"] = -1
            self.game_status = "agent_illegal_move"
            next_state_dict = {{"board": self.board.copy(), "turn": self.current_turn}}
            reward, breakdown = self.compute_reward(state_dict, action, next_state_dict, info)
            info["reward_breakdown"] = breakdown
            info["arena_state"] = self.render_arena_state()
            return self._get_obs(), reward, True, False, info

        # Agent plays X (+1)
        self.board[action] = 1
        res = self._check_winner()

        if res is not None:
            self.winner = res
            self.game_status = "agent_won" if res == 1 else ("draw" if res == 0 else "opponent_won")
            terminated = True
        else:
            # Opponent turn: heuristic defense / random play
            self.current_turn = -1
            empty_cells = np.where(self.board == 0)[0]
            if len(empty_cells) > 0:
                opp_move = self._heuristic_opponent_move(empty_cells)
                self.board[opp_move] = -1
                res = self._check_winner()
                if res is not None:
                    self.winner = res
                    self.game_status = "agent_won" if res == 1 else ("draw" if res == 0 else "opponent_won")
                    terminated = True
                else:
                    terminated = False
            else:
                self.winner = 0
                self.game_status = "draw"
                terminated = True
            self.current_turn = 1

        info["winner"] = self.winner
        info["is_success"] = (self.winner == 1 or self.winner == 0)

        next_state_dict = {{"board": self.board.copy(), "turn": self.current_turn}}
        reward, breakdown = self.compute_reward(state_dict, action, next_state_dict, info)
        info["reward_breakdown"] = breakdown
        info["arena_state"] = self.render_arena_state()

        return self._get_obs(), float(reward), terminated, False, info

    def _heuristic_opponent_move(self, empty_cells: np.ndarray) -> int:
        """Blocks immediate agent winning moves, otherwise plays tactical cell."""
        # 1. Check if opponent can win now
        for cell in empty_cells:
            self.board[cell] = -1
            if self._check_winner() == -1:
                self.board[cell] = 0
                return int(cell)
            self.board[cell] = 0

        # 2. Block agent winning move
        for cell in empty_cells:
            self.board[cell] = 1
            if self._check_winner() == 1:
                self.board[cell] = 0
                return int(cell)
            self.board[cell] = 0

        # 3. Prefer center
        if 4 in empty_cells and random.random() < 0.7:
            return 4

        return int(random.choice(empty_cells))

    def render_arena_state(self) -> dict[str, Any]:
        """Returns JSON-serializable telemetry dictionary for arena.html."""
        return {{
            "board": [int(x) for x in self.board],
            "current_turn": "agent" if self.current_turn == 1 else "opponent",
            "game_status": self.game_status,
            "winner": self.winner,
            "last_move": self.last_move,
            "step_count": self.step_count,
            "action_labels": self.action_labels,
        }}

    def interactive_step(self, payload: dict[str, Any], policy_predict_fn: Callable) -> dict[str, Any]:
        """Handles human play from arena.html."""
        if payload.get("reset"):
            self.reset()
            return {{
                "arena_state": self.render_arena_state(),
                "agent_action_label": "None",
                "action_probs": [1.0 / 9] * 9,
                "status_text": "Board reset. Your turn to play as O!",
            }}

        human_cell = payload.get("user_action")
        if human_cell is not None:
            human_cell = int(human_cell)
            if 0 <= human_cell < 9 and self.board[human_cell] == 0:
                self.board[human_cell] = -1
                if self._check_winner() is not None:
                    res = self._check_winner()
                    self.winner = res
                    self.game_status = "human_won" if res == -1 else "draw"
                    return {{
                        "arena_state": self.render_arena_state(),
                        "agent_action_label": "Game Over",
                        "action_probs": [0.0] * 9,
                        "status_text": f"Game Over: {{self.game_status}}",
                    }}

        # Agent responds
        obs = self._get_obs()
        action_idx, probs = policy_predict_fn(obs)
        action_idx = int(action_idx)

        # Apply agent move
        if self.board[action_idx] == 0:
            self.board[action_idx] = 1
            agent_label = self.action_labels[action_idx]
            status_text = f"Agent played {{agent_label}}"
        else:
            # Fallback to first empty cell
            empty = np.where(self.board == 0)[0]
            if len(empty) > 0:
                action_idx = int(empty[0])
                self.board[action_idx] = 1
                agent_label = self.action_labels[action_idx]
                status_text = f"Agent fallback played {{agent_label}}"
            else:
                agent_label = "Draw"
                status_text = "Board is full - Draw!"

        res = self._check_winner()
        if res is not None:
            self.winner = res
            self.game_status = "agent_won" if res == 1 else ("draw" if res == 0 else "human_won")
            status_text = f"Game Over: {{self.game_status}}"

        return {{
            "arena_state": self.render_arena_state(),
            "agent_action_label": agent_label,
            "action_probs": [float(p) for p in probs],
            "status_text": status_text,
        }}

    def get_demonstrations(self, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Heuristic demonstrations for Warm-Start Behavioral Cloning."""
        X = np.zeros((num_samples, 10), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.int64)

        for i in range(num_samples):
            # Generate plausible board
            b = np.zeros(9, dtype=np.int8)
            num_turns = random.randint(0, 4)
            for _ in range(num_turns):
                empty = np.where(b == 0)[0]
                if len(empty) >= 2:
                    b[random.choice(empty)] = 1
                    empty = np.where(b == 0)[0]
                    b[random.choice(empty)] = -1

            empty = np.where(b == 0)[0]
            if len(empty) == 0:
                b.fill(0)
                empty = np.arange(9)

            # Heuristic target action
            action = int(empty[0])
            if 4 in empty and random.random() < 0.6:
                action = 4

            obs = np.zeros(10, dtype=np.float32)
            obs[:9] = b.astype(np.float32)
            obs[9] = 1.0  # agent turn
            X[i] = obs
            y[i] = action

        return X, y
'''

    def _build_dataset_code(
        self,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None,
    ) -> str:
        """Generates custom data-driven Gymnasium environment stepping through records."""
        obs_features = blueprint.observation_features
        obs_dim = len(obs_features)
        action_labels = blueprint.action_labels
        n_actions = len(action_labels)

        # Extract sample records
        sample_records = []
        if dataset is not None and hasattr(dataset, "df") and dataset.df is not None:
            for _, row in dataset.df.head(100).iterrows():
                rec = {}
                for col in obs_features:
                    val = row.get(col, 0.0)
                    try:
                        rec[col] = float(val)
                    except (ValueError, TypeError):
                        rec[col] = float(hash(str(val)) % 100) / 10.0
                sample_records.append(rec)

        if not sample_records:
            # Fallback synthetic records
            for i in range(100):
                rec = {col: float(np.sin(i * 0.1 + j)) for j, col in enumerate(obs_features)}
                sample_records.append(rec)

        records_json = json.dumps(sample_records)

        return f'''"""Auto-synthesized Gymnasium environment for Data Inspection & Decision."""

from __future__ import annotations
import json
import math
import random
from typing import Any, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SynthesizedGymEnv(gym.Env):
    """Auto-synthesized Gymnasium environment for: {blueprint.task_title}"""
    metadata = {{"render_modes": ["json"]}}

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.action_labels = {action_labels}
        self.observation_features = {obs_features}
        self.obs_dim = {obs_dim}
        self.n_actions = {n_actions}

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        # Ingested dataset records
        self.records = dataset_records if dataset_records is not None else {records_json}
        if not self.records:
            self.records = [{{{", ".join(f"'{c}': 0.0" for c in obs_features)}}}]

        self.current_idx = 0
        self.step_count = 0
        self.correct_count = 0
        self.last_action = None
        self.last_prediction_truth = 0

    def _extract_obs(self, record: dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        for i, col in enumerate(self.observation_features):
            val = record.get(col, 0.0)
            try:
                vec[i] = float(val)
            except (ValueError, TypeError):
                vec[i] = float(hash(str(val)) % 100) / 10.0
        return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_idx = random.randint(0, len(self.records) - 1)
        self.step_count = 0
        self.correct_count = 0
        self.last_action = None

        record = self.records[self.current_idx]
        obs = self._extract_obs(record)
        return obs, {{"arena_state": self.render_arena_state()}}

    def compute_reward(self, state: dict, action: int, next_state: dict, info: dict) -> tuple[float, dict[str, float]]:
        """Modular reward function. Rewritten across rounds by Autonomous Auto-Tuner."""
        breakdown = {{}}
        total = 0.0

        is_correct = info.get("is_correct", False)
        if is_correct:
            breakdown["correct_decision"] = 1.0
            breakdown["confidence_bonus"] = 0.2
            total += 1.2
        else:
            breakdown["incorrect_penalty"] = -1.0
            total += -1.0

        return float(total), breakdown

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        self.step_count += 1
        self.last_action = action

        record = self.records[self.current_idx]
        # Ground truth pseudo-label derived from first numerical column or index
        first_feat = record.get(self.observation_features[0], 0.0)
        try:
            truth = int(float(first_feat)) % self.n_actions
        except (ValueError, TypeError):
            truth = self.current_idx % self.n_actions

        self.last_prediction_truth = truth
        is_correct = (action == truth)
        if is_correct:
            self.correct_count += 1

        info: dict[str, Any] = {{
            "is_correct": is_correct,
            "is_success": is_correct,
            "record_index": self.current_idx,
            "truth": truth,
        }}

        state_dict = {{"record": record, "action": action}}
        # Advance to next record
        self.current_idx = (self.current_idx + 1) % len(self.records)
        next_record = self.records[self.current_idx]
        next_obs = self._extract_obs(next_record)
        next_state_dict = {{"record": next_record}}

        reward, breakdown = self.compute_reward(state_dict, action, next_state_dict, info)
        info["reward_breakdown"] = breakdown
        info["arena_state"] = self.render_arena_state()

        terminated = (self.step_count >= min(25, len(self.records)))
        return next_obs, float(reward), terminated, False, info

    def render_arena_state(self) -> dict[str, Any]:
        """Returns JSON-serializable telemetry dictionary for arena.html."""
        rec = self.records[self.current_idx] if self.records else {{}}
        acc = (self.correct_count / max(1, self.step_count)) * 100.0
        return {{
            "record_index": self.current_idx,
            "total_records": len(self.records),
            "step_count": self.step_count,
            "accuracy_pct": round(acc, 1),
            "last_action_label": self.action_labels[self.last_action] if self.last_action is not None else "None",
            "current_record_preview": {{k: str(v)[:30] for k, v in list(rec.items())[:6]}},
            "action_labels": self.action_labels,
        }}

    def interactive_step(self, payload: dict[str, Any], policy_predict_fn: Callable) -> dict[str, Any]:
        """Handles human inspection from arena.html."""
        custom_record = payload.get("custom_params") or payload.get("record")
        if custom_record and isinstance(custom_record, dict):
            obs = self._extract_obs(custom_record)
            active_rec = custom_record
        else:
            rec_idx = int(payload.get("user_action", self.current_idx)) % len(self.records)
            self.current_idx = rec_idx
            active_rec = self.records[rec_idx]
            obs = self._extract_obs(active_rec)

        action_idx, probs = policy_predict_fn(obs)
        action_idx = int(action_idx)
        label = self.action_labels[action_idx]

        return {{
            "arena_state": self.render_arena_state(),
            "agent_action_label": label,
            "action_probs": [float(p) for p in probs],
            "status_text": f"Agent classified record as: {{label}} (confidence: {{round(float(probs[action_idx])*100, 1)}}%)",
        }}

    def get_demonstrations(self, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Heuristic demonstrations for Warm-Start Behavioral Cloning."""
        n = num_samples
        X = np.zeros((n, self.obs_dim), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)

        for i in range(n):
            rec = self.records[i % len(self.records)]
            obs = self._extract_obs(rec)
            first_feat = rec.get(self.observation_features[0], 0.0)
            try:
                truth = int(float(first_feat)) % self.n_actions
            except (ValueError, TypeError):
                truth = i % self.n_actions

            X[i] = obs
            y[i] = truth

        return X, y
'''

    def _build_general_sim_code(self, blueprint: AgentBlueprintSpec) -> str:
        """Generates general simulation 2D navigation Gym environment."""
        obs_features = blueprint.observation_features
        obs_dim = max(4, len(obs_features))
        action_labels = blueprint.action_labels
        n_actions = len(action_labels)

        return f'''"""Auto-synthesized Gymnasium environment for Continuous Simulation & Navigation."""

from __future__ import annotations
import math
import random
from typing import Any, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SynthesizedGymEnv(gym.Env):
    """Auto-synthesized Gymnasium environment for: {blueprint.task_title}"""
    metadata = {{"render_modes": ["json"]}}

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.action_labels = {action_labels}
        self.observation_features = {obs_features}
        self.obs_dim = {obs_dim}
        self.n_actions = {n_actions}

        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.target_pos = np.array([5.0, 5.0], dtype=np.float32)
        self.step_count = 0
        self.target_reached = False

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = self.agent_pos[0]
        obs[1] = self.agent_pos[1]
        obs[2] = self.target_pos[0]
        obs[3] = self.target_pos[1]
        if self.obs_dim > 4:
            dist = float(np.linalg.norm(self.target_pos - self.agent_pos))
            obs[4] = dist
        return obs

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.agent_pos = np.random.uniform(-5.0, 5.0, size=(2,)).astype(np.float32)
        self.target_pos = np.random.uniform(-8.0, 8.0, size=(2,)).astype(np.float32)
        self.step_count = 0
        self.target_reached = False
        return self._get_obs(), {{"arena_state": self.render_arena_state()}}

    def compute_reward(self, state: dict, action: int, next_state: dict, info: dict) -> tuple[float, dict[str, float]]:
        """Modular reward function. Rewritten across rounds by Autonomous Auto-Tuner."""
        breakdown = {{}}
        total = 0.0

        dist_delta = info.get("dist_delta", 0.0)
        breakdown["distance_delta"] = float(dist_delta * 2.0)
        total += float(dist_delta * 2.0)

        if info.get("is_success", False):
            breakdown["goal_bonus"] = 10.0
            total += 10.0

        breakdown["step_cost"] = -0.05
        total += -0.05

        return float(total), breakdown

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        self.step_count += 1

        prev_dist = float(np.linalg.norm(self.target_pos - self.agent_pos))
        # Move agent
        moves = [
            [-1.0, 0.0],  # Left
            [1.0, 0.0],   # Right
            [0.0, 1.0],   # Up
            [0.0, -1.0],  # Down
            [0.0, 0.0],   # Stay
        ]
        delta = moves[action % len(moves)]
        self.agent_pos[0] += delta[0] * 0.5
        self.agent_pos[1] += delta[1] * 0.5

        new_dist = float(np.linalg.norm(self.target_pos - self.agent_pos))
        dist_delta = prev_dist - new_dist

        self.target_reached = (new_dist < 1.0)
        terminated = self.target_reached or (self.step_count >= 50)

        info: dict[str, Any] = {{
            "dist_delta": dist_delta,
            "distance": new_dist,
            "is_success": self.target_reached,
        }}

        state_dict = {{"pos": self.agent_pos.copy()}}
        next_state_dict = {{"pos": self.agent_pos.copy()}}
        reward, breakdown = self.compute_reward(state_dict, action, next_state_dict, info)
        info["reward_breakdown"] = breakdown
        info["arena_state"] = self.render_arena_state()

        return self._get_obs(), float(reward), terminated, False, info

    def render_arena_state(self) -> dict[str, Any]:
        """Returns JSON-serializable telemetry dictionary for arena.html."""
        return {{
            "agent_pos": [float(self.agent_pos[0]), float(self.agent_pos[1])],
            "target_pos": [float(self.target_pos[0]), float(self.target_pos[1])],
            "distance": round(float(np.linalg.norm(self.target_pos - self.agent_pos)), 2),
            "step_count": self.step_count,
            "target_reached": self.target_reached,
            "action_labels": self.action_labels,
        }}

    def interactive_step(self, payload: dict[str, Any], policy_predict_fn: Callable) -> dict[str, Any]:
        """Handles human click / waypoint dragging from arena.html."""
        custom_target = payload.get("target_pos") or payload.get("custom_params", {{}}).get("target_pos")
        if custom_target and len(custom_target) >= 2:
            self.target_pos[0] = float(custom_target[0])
            self.target_pos[1] = float(custom_target[1])

        obs = self._get_obs()
        action_idx, probs = policy_predict_fn(obs)
        action_idx = int(action_idx)
        label = self.action_labels[action_idx]

        # Step agent with selected action
        self.step(action_idx)

        return {{
            "arena_state": self.render_arena_state(),
            "agent_action_label": label,
            "action_probs": [float(p) for p in probs],
            "status_text": f"Agent moved {{label}} toward waypoint",
        }}

    def get_demonstrations(self, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Heuristic demonstrations for Warm-Start Behavioral Cloning."""
        X = np.zeros((num_samples, self.obs_dim), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.int64)

        for i in range(num_samples):
            agent = np.random.uniform(-5.0, 5.0, size=(2,)).astype(np.float32)
            target = np.random.uniform(-8.0, 8.0, size=(2,)).astype(np.float32)
            diff = target - agent

            # Determine best cardinal direction
            if abs(diff[0]) > abs(diff[1]):
                act = 1 if diff[0] > 0 else 0
            else:
                act = 2 if diff[1] > 0 else 3

            obs = np.zeros(self.obs_dim, dtype=np.float32)
            obs[0] = agent[0]
            obs[1] = agent[1]
            obs[2] = target[0]
            obs[3] = target[1]
            if self.obs_dim > 4:
                obs[4] = float(np.linalg.norm(diff))

            X[i] = obs
            y[i] = act % self.n_actions

        return X, y
'''
