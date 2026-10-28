"""JIT Environment Synthesizer for RL-Native Agent IDE v4.0.

Synthesizes self-contained Gymnasium environment modules (env_module.py)
tailored to any AgentBlueprintSpec and IngestedDataset.
"""

from __future__ import annotations
import ast
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
from rl_ide.llm.client import BaseLLMClient, create_llm_client, FallbackCodeSynthesizer, get_active_llm_client
from rl_ide.llm.prompts import ENV_SYNTHESIZER_SYSTEM_PROMPT, ENV_BUILDER_SYSTEM_PROMPT


def extract_python_code(raw: str) -> str:
    """Extracts valid Python code from raw LLM output, handling markdown fences and truncated blocks."""
    if not raw:
        return ""
    text = raw.strip()

    # 1. Match standard closed markdown block
    match = re.search(r"```(?:python)?\s*\n([\s\S]*?)\n```", text)
    if match:
        return match.group(1).strip()

    # 2. Match unclosed markdown block (e.g. truncated output starting with ```python)
    match_unclosed = re.search(r"```(?:python)?\s*\n([\s\S]*)", text)
    if match_unclosed:
        candidate = match_unclosed.group(1).strip()
        candidate = re.sub(r"```+$", "", candidate).strip()
        return candidate

    # 3. Direct fence cleanup if code still starts with markdown backticks
    if text.startswith("```"):
        lines = text.split("\n")
        first_line = lines[0].strip()
        if first_line.startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return text


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


def instantiate_env(env_cls: type[gym.Env], dataset_records: list[dict[str, Any]] | None = None) -> gym.Env:
    """Safely instantiates a synthesized Gymnasium environment handling variable __init__ signatures."""
    try:
        return env_cls(dataset_records=dataset_records)
    except TypeError:
        try:
            return env_cls()
        except TypeError:
            return env_cls(dataset_records)


class JITEnvironmentSynthesizer:
    """Synthesizes custom Gym environments tailored to an AgentBlueprintSpec."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self._llm_client = llm_client

    @property
    def client(self) -> BaseLLMClient:
        if self._llm_client:
            return self._llm_client
        return get_active_llm_client()

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
        try:
            audit_ast_security(code_str)
        except Exception:
            # Fallback to robust deterministic code if AST security or syntax failed
            code_str = self._generate_deterministic_env_code(blueprint, dataset)
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
                if code and "class SynthesizedGymEnv" in code:
                    try:
                        ast.parse(code)
                        return code
                    except SyntaxError:
                        pass
            except Exception:
                pass

        # Deterministic Code Generation (used for offline Fallback in test mode or domain templates)
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

        prompt += "\nOutput ONLY valid Python code inside a ```python ... ``` block. Be concise and complete."
        raw = self.client.generate(prompt, system_prompt=ENV_BUILDER_SYSTEM_PROMPT)

        return extract_python_code(raw)

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
        win_bonus = min(10.0, max(-10.0, reward_terms.get("win_bonus", 10.0)))
        draw_bonus = min(10.0, max(-10.0, reward_terms.get("draw_bonus", 3.0)))
        loss_penalty = min(10.0, max(-10.0, reward_terms.get("loss_penalty", -10.0)))
        illegal_penalty = min(10.0, max(-10.0, reward_terms.get("illegal_move_penalty", -1.0)))

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

    _memo: dict[tuple, tuple[int, int]] = {{}}

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.action_labels = {blueprint.action_labels}
        self.observation_features = {blueprint.observation_features}
        
        # 9 board cells (-1=opponent, 0=empty, +1=agent) + 1 perspective indicator (+1=agent first, -1=opponent first)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        self.board = np.zeros(9, dtype=np.int8)
        self.current_turn = 1  # 1 = agent, -1 = opponent
        self.agent_plays_first = True
        self.opponent_profile = "tactical"  # "random" | "tactical" | "optimal"
        self.step_count = 0
        self.game_status = "in_progress"
        self.winner = None
        self.last_move = None

    def get_valid_actions(self) -> list[int]:
        """Returns list of valid empty cell indices."""
        return [int(i) for i in np.where(self.board == 0)[0]]

    def get_critical_actions(self) -> list[int] | None:
        """Returns critical immediate win, urgent defensive block, or non-losing actions."""
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        empty = self.get_valid_actions()
        if not empty:
            return None

        # 1. Immediate agent win
        win_moves = []
        for cell in empty:
            self.board[cell] = 1
            if self._check_winner() == 1:
                win_moves.append(cell)
            self.board[cell] = 0
        if win_moves:
            return win_moves

        # 2. Urgent defensive block against opponent immediate win
        block_moves = []
        for cell in empty:
            self.board[cell] = -1
            if self._check_winner() == -1:
                block_moves.append(cell)
            self.board[cell] = 0
        if block_moves:
            return block_moves

        # 3. Double-threat fork defense: filter out blunder moves that allow opponent a forced win
        non_losing_moves = []
        for cell in empty:
            self.board[cell] = 1
            opp_score, _ = self._minimax(tuple(int(x) for x in self.board), -1)
            self.board[cell] = 0
            if opp_score >= 0:
                non_losing_moves.append(cell)

        if non_losing_moves and len(non_losing_moves) < len(empty):
            return non_losing_moves

        return None

    def _minimax(self, b_tuple: tuple, player: int) -> tuple[int, int]:
        key = (b_tuple, player)
        if key in self._memo:
            return self._memo[key]

        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for p1, p2, p3 in lines:
            if b_tuple[p1] != 0 and b_tuple[p1] == b_tuple[p2] == b_tuple[p3]:
                return (b_tuple[p1], -1)
        if all(x != 0 for x in b_tuple):
            return (0, -1)

        best_score = -999 if player == 1 else 999
        best_move = -1
        for i in range(9):
            if b_tuple[i] == 0:
                nb = list(b_tuple)
                nb[i] = player
                score, _ = self._minimax(tuple(nb), -player)
                if player == 1:
                    if score > best_score:
                        best_score = score
                        best_move = i
                else:
                    if score < best_score:
                        best_score = score
                        best_move = i

        self._memo[key] = (best_score, best_move)
        return (best_score, best_move)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[:9] = self.board.astype(np.float32)
        obs[9] = 1.0 if self.agent_plays_first else -1.0
        return obs

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board.fill(0)
        self.step_count = 0
        self.game_status = "in_progress"
        self.winner = None
        self.last_move = None

        options = options or {{}}
        # First-turn randomization (50% agent first, 50% opponent first)
        if "agent_first" in options:
            self.agent_plays_first = bool(options["agent_first"])
        else:
            self.agent_plays_first = (random.random() < 0.5)

        # Multi-strategy opponent profile randomization
        if "opponent_profile" in options:
            self.opponent_profile = str(options["opponent_profile"])
        else:
            r = random.random()
            if r < 0.50:
                self.opponent_profile = "random"
            elif r < 0.80:
                self.opponent_profile = "tactical"
            else:
                self.opponent_profile = "optimal"

        self.current_turn = 1
        # If opponent plays first, opponent executes opening move
        if not self.agent_plays_first:
            empty = self.get_valid_actions()
            opp_move = self._heuristic_opponent_move(np.array(empty))
            self.board[opp_move] = -1
            self.last_move = opp_move

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

        if info.get("is_illegal", False) or info.get("illegal_move_attempted", False):
            breakdown["illegal_move_penalty"] = {illegal_penalty}
            total += {illegal_penalty}

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
        info: dict[str, Any] = {{"is_illegal": False, "illegal_move_attempted": False, "is_success": False}}

        valid_actions = self.get_valid_actions()
        if not valid_actions:
            self.winner = 0
            self.game_status = "draw"
            info["winner"] = 0
            info["is_success"] = True
            reward, breakdown = self.compute_reward(state_dict, action, state_dict, info)
            info["reward_breakdown"] = breakdown
            info["arena_state"] = self.render_arena_state()
            return self._get_obs(), float(reward), True, False, info

        # Invalid action check & sanitization
        if action not in valid_actions:
            info["is_illegal"] = True
            info["illegal_move_attempted"] = True
            critical = self.get_critical_actions()
            if critical:
                action = int(critical[0])
            elif 4 in valid_actions:
                action = 4
            else:
                action = int(valid_actions[0])

        # Agent plays (+1)
        self.board[action] = 1
        res = self._check_winner()

        if res is not None:
            self.winner = res
            self.game_status = "agent_won" if res == 1 else ("draw" if res == 0 else "opponent_won")
            terminated = True
        else:
            # Opponent turn
            self.current_turn = -1
            empty_cells = self.get_valid_actions()
            if len(empty_cells) > 0:
                opp_move = self._heuristic_opponent_move(np.array(empty_cells))
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
        """Returns opponent move based on selected behavior profile."""
        empty = [int(c) for c in empty_cells]
        if not empty:
            return 0

        profile = getattr(self, "opponent_profile", "tactical")
        if profile == "random":
            return int(random.choice(empty))

        if profile == "optimal":
            _, best_move = self._minimax(tuple(int(x) for x in self.board), -1)
            if best_move in empty:
                return int(best_move)

        # Tactical opponent: checks win, block, center, corners
        for cell in empty:
            self.board[cell] = -1
            if self._check_winner() == -1:
                self.board[cell] = 0
                return int(cell)
            self.board[cell] = 0

        for cell in empty:
            self.board[cell] = 1
            if self._check_winner() == 1:
                self.board[cell] = 0
                return int(cell)
            self.board[cell] = 0

        if 4 in empty and random.random() < 0.75:
            return 4

        corners = [c for c in [0, 2, 6, 8] if c in empty]
        if corners:
            return int(random.choice(corners))

        return int(random.choice(empty))

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
            "agent_plays_first": self.agent_plays_first,
            "opponent_profile": self.opponent_profile,
        }}

    def interactive_step(self, payload: dict[str, Any], policy_predict_fn: Callable) -> dict[str, Any]:
        """Handles human play from arena.html."""
        if payload.get("reset"):
            self.board.fill(0)
            self.step_count = 0
            self.game_status = "in_progress"
            self.winner = None
            self.last_move = None
            human_first = bool(payload.get("human_first", True))
            self.agent_plays_first = not human_first
            if not human_first:
                # Agent makes first move immediately
                obs = self._get_obs()
                action_idx, probs = policy_predict_fn(obs)
                action_idx = int(action_idx)
                valid_actions = self.get_valid_actions()
                if valid_actions and action_idx not in valid_actions:
                    action_idx = int(valid_actions[0])
                self.board[action_idx] = 1
                self.last_move = action_idx
                agent_label = self.action_labels[action_idx]
                return {{
                    "arena_state": self.render_arena_state(),
                    "agent_action_label": agent_label,
                    "action_probs": [float(p) for p in probs],
                    "status_text": f"Agent opened with {{agent_label}}. Your turn to play as O!",
                }}
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

        # Critical & Valid action masking
        valid_actions = self.get_valid_actions()
        critical_actions = self.get_critical_actions()
        target_actions = critical_actions if (critical_actions and len(critical_actions) > 0) else valid_actions

        if target_actions:
            masked_probs = np.zeros(9, dtype=np.float32)
            for va in target_actions:
                masked_probs[va] = probs[va] if va < len(probs) else 1.0
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
                action_idx = int(np.argmax(masked_probs))
            else:
                action_idx = int(target_actions[0])
            probs = masked_probs.tolist()

        # Apply agent move
        self.board[action_idx] = 1
        agent_label = self.action_labels[action_idx]
        status_text = f"Agent played {{agent_label}}"

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

    def get_demonstrations(self, num_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """Generates demonstrations for Warm-Start Behavioral Cloning covering both 1st and 2nd player positions."""
        X = np.zeros((num_samples, 10), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.int64)

        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]

        idx = 0
        while idx < num_samples:
            board = np.zeros(9, dtype=np.int8)
            agent_first = (random.random() < 0.5)
            profile = random.choice(["random", "tactical", "optimal"])

            if not agent_first:
                # Opponent takes opening move
                if profile == "random":
                    mv = random.choice(list(range(9)))
                elif profile == "tactical":
                    mv = 4 if random.random() < 0.7 else random.choice([0, 2, 6, 8])
                else:
                    _, mv = self._minimax(tuple(int(x) for x in board), -1)
                board[mv] = -1

            while idx < num_samples:
                empty = [int(i) for i in np.where(board == 0)[0]]
                if not empty:
                    break
                won = False
                for p1, p2, p3 in lines:
                    if board[p1] != 0 and board[p1] == board[p2] == board[p3]:
                        won = True
                        break
                if won:
                    break

                # Target Agent Move
                _, best_a = self._minimax(tuple(int(x) for x in board), 1)
                if best_a not in empty:
                    best_a = empty[0]

                obs = np.zeros(10, dtype=np.float32)
                obs[:9] = board.astype(np.float32)
                obs[9] = 1.0 if agent_first else -1.0
                X[idx] = obs
                y[idx] = int(best_a)
                idx += 1

                board[best_a] = 1

                won = False
                for p1, p2, p3 in lines:
                    if board[p1] != 0 and board[p1] == board[p2] == board[p3]:
                        won = True
                        break
                if won:
                    break

                opp_empty = [int(i) for i in np.where(board == 0)[0]]
                if not opp_empty:
                    break

                # Opponent response move
                if profile == "random":
                    opp_mv = random.choice(opp_empty)
                elif profile == "optimal":
                    _, opp_mv = self._minimax(tuple(int(x) for x in board), -1)
                    if opp_mv not in opp_empty:
                        opp_mv = opp_empty[0]
                else:
                    found = False
                    for c in opp_empty:
                        board[c] = -1
                        for p1, p2, p3 in lines:
                            if board[p1] == board[p2] == board[p3] == -1:
                                opp_mv = c
                                found = True
                                break
                        board[c] = 0
                        if found:
                            break
                    if not found:
                        for c in opp_empty:
                            board[c] = 1
                            for p1, p2, p3 in lines:
                                if board[p1] == board[p2] == board[p3] == 1:
                                    opp_mv = c
                                    found = True
                                    break
                            board[c] = 0
                            if found:
                                break
                    if not found:
                        if 4 in opp_empty and random.random() < 0.7:
                            opp_mv = 4
                        else:
                            corners = [c for c in [0, 2, 6, 8] if c in opp_empty]
                            opp_mv = random.choice(corners) if corners else random.choice(opp_empty)

                board[opp_mv] = -1
                for p1, p2, p3 in lines:
                    if board[p1] != 0 and board[p1] == board[p2] == board[p3]:
                        won = True
                        break
                if won:
                    break

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
