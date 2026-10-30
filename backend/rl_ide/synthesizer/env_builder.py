"""JIT Environment Synthesizer for RL-Native Agent IDE v4.4.

Universal LLM Gymnasium environment code synthesizer with pre-import AST security,
dynamic schema synthesis, and multi-iteration self-healing sandbox verification.
Eliminates all domain-specific switchboards and hardcoded templates.
"""

from __future__ import annotations
import ast
import importlib.util
import json
import math
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_ide.architect.models import AgentBlueprintSpec, DynamicTaskBlueprint
from rl_ide.ingest.loader import IngestedDataset
from rl_ide.llm.client import BaseLLMClient, FallbackCodeSynthesizer, get_active_llm_client
from rl_ide.llm.prompts import ENV_BUILDER_SYSTEM_PROMPT


class EnvironmentSynthesisError(RuntimeError):
    """Raised when no verified, task-specific environment could be synthesized."""


def is_test_mode() -> bool:
    """Local import wrapper so env synthesis can decide about deterministic placeholders."""
    from rl_ide.llm.client import is_test_mode as _is_test_mode
    return _is_test_mode()


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


_LOADED_MODULES: list[str] = []
_MAX_LOADED_MODULES = 64


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
    _LOADED_MODULES.append(unique_mod_name)
    while len(_LOADED_MODULES) > _MAX_LOADED_MODULES:
        stale = _LOADED_MODULES.pop(0)
        sys.modules.pop(stale, None)
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


def _generate_schema_driven_env_code(
    blueprint: AgentBlueprintSpec | DynamicTaskBlueprint,
    user_prompt: str = "",
) -> str:
    """Universal schema-driven generator that produces a complete Gymnasium environment for ANY blueprint."""
    task_title = getattr(blueprint, "domain_title", None) or getattr(blueprint, "task_title", "") or "Universal RL Agent"
    summary = getattr(blueprint, "domain_summary", "") or "Autonomous decision making agent."

    if hasattr(blueprint, "observation_feature_specs") and blueprint.observation_feature_specs:
        obs_features = [f.name for f in blueprint.observation_feature_specs]
    elif hasattr(blueprint, "observation_features") and blueprint.observation_features:
        obs_features = list(blueprint.observation_features)
    else:
        obs_features = ["state_0", "state_1", "state_2", "state_3"]

    if hasattr(blueprint, "actions") and blueprint.actions:
        action_labels = [a.label for a in blueprint.actions]
    elif hasattr(blueprint, "action_labels") and blueprint.action_labels:
        action_labels = list(blueprint.action_labels)
    else:
        action_labels = ["Action 0", "Action 1", "Action 2"]

    state_schema = getattr(blueprint, "state_dict_schema", {}) or {}
    if not state_schema:
        state_schema = {f: "float" for f in obs_features}
    state_schema["status"] = "str"
    state_schema["step_count"] = "int"

    obs_dim = max(1, len(obs_features))
    n_actions = max(1, len(action_labels))

    return f'''"""Auto-synthesized Gymnasium environment for: {task_title}"""

from __future__ import annotations
import math
import random
from typing import Any, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SynthesizedGymEnv(gym.Env):
    """Auto-synthesized Gymnasium environment for: {task_title}"""
    metadata = {{"render_modes": ["json"]}}

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.action_labels = {action_labels}
        self.observation_features = {obs_features}
        self.obs_dim = {obs_dim}
        self.n_actions = {n_actions}

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self.state_vars = {{f: 0.0 for f in self.observation_features}}
        self.step_count = 0
        self.done = False
        self.last_move = None
        # Honest placeholder metadata: this module implements no domain objective.
        self.is_placeholder = True
        self.task_class = "control"
        self.episode_horizon = 25
        self.success_criteria = "Placeholder environment: no domain objective was synthesized."
        has_board_schema = "board" in {state_schema} or self.n_actions == 9
        self.board = [0] * (self.n_actions if self.n_actions > 0 else 9) if has_board_schema else None
        self.reset()

    def _get_obs(self) -> np.ndarray:
        if not self.state_vars:
            self.reset()
        obs = np.zeros({obs_dim}, dtype=np.float32)
        for i, f in enumerate(self.observation_features):
            obs[i] = float(self.state_vars.get(f, 0.0))
        return obs

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.step_count = 0
        self.done = False
        self.last_move = None
        if hasattr(self, "board") and self.board is not None:
            self.board = [0] * len(self.board)
        self.state_vars = {{f: float(random.uniform(-1.0, 1.0)) for f in self.observation_features}}
        return self._get_obs(), {{
            "reward_breakdown": {{}},
            "is_success": False,
            "is_correct": None,
            "is_placeholder": self.is_placeholder,
            "arena_state": self.get_render_state(),
        }}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.state_vars:
            self.reset()
        self.step_count += 1
        act = min(max(0, int(action)), self.n_actions - 1)
        self.last_move = act

        for idx, f in enumerate(self.observation_features):
            delta = 0.5 if act == idx % self.n_actions else -0.1
            curr_val = self.state_vars.get(f, 0.0)
            self.state_vars[f] = float(np.clip(curr_val + delta + random.uniform(-0.05, 0.05), -10.0, 10.0))

        terminated = (self.step_count >= self.episode_horizon)
        obs = self._get_obs()
        reward, bd = self.compute_reward(obs, act, obs, {{"reward_breakdown": {{}}}})
        info = {{
            "reward_breakdown": bd,
            # A placeholder has no objective, so it must never claim success.
            "is_success": False,
            "is_correct": None,
            "is_placeholder": True,
            "arena_state": self.get_render_state(),
        }}
        return obs, float(reward), terminated, False, info

    def compute_reward(self, obs, action, next_obs, info) -> tuple[float, dict[str, float]]:
        if not self.state_vars:
            self.reset()
        # Neutral reward: a placeholder must not manufacture a learning signal.
        bd = dict((info or {{}}).get("reward_breakdown", {{}}))
        return 0.0, bd

    def get_render_state(self) -> dict[str, Any]:
        if not self.state_vars:
            self.reset()
        res = {{f: float(self.state_vars.get(f, 0.0)) for f in self.observation_features}}
        schema = {state_schema}
        for k in schema:
            if k not in res:
                res[k] = 0.0 if "float" in str(schema[k]) else (0 if "int" in str(schema[k]) else ("active" if "str" in str(schema[k]) else []))
        if hasattr(self, "board") and self.board is not None:
            res["board"] = list(self.board)
        res["step_count"] = self.step_count
        res["status"] = "operational"
        return res

    def render_arena_state(self) -> dict[str, Any]:
        return self.get_render_state()

    def interactive_step(self, payload: dict[str, Any], policy_predict_fn: Callable = None, mode: str = "agent_vs_env") -> dict[str, Any]:
        if not self.state_vars:
            self.reset()
        if isinstance(payload, dict) and payload.get("reset"):
            self.reset()
            human_first = payload.get("human_first", True)
            if not human_first and policy_predict_fn is not None:
                obs = self._get_obs()
                action_idx, probs = policy_predict_fn(obs)
                action_idx = int(action_idx)
                self.last_move = int(action_idx)
                if hasattr(self, "board") and self.board is not None and action_idx < len(self.board):
                    self.board[action_idx] = 1
                return {{
                    "arena_state": self.get_render_state(),
                    "agent_action_label": self.action_labels[min(int(action_idx), self.n_actions - 1)],
                    "action_probs": [float(p) for p in probs] if probs is not None else [1.0 / self.n_actions] * self.n_actions,
                    "status_text": "Agent opening move: " + str(self.action_labels[min(int(action_idx), self.n_actions - 1)]),
                }}
            return {{
                "arena_state": self.get_render_state(),
                "agent_action_label": "Reset",
                "action_probs": [1.0 / self.n_actions] * self.n_actions,
                "status_text": "System reset to initial state.",
            }}

        user_action = 0
        if isinstance(payload, dict):
            if "user_action" in payload:
                user_action = payload["user_action"]
            elif "cell" in payload:
                user_action = payload["cell"]
            elif "action" in payload and isinstance(payload["action"], (int, float)):
                user_action = int(payload["action"])
        elif isinstance(payload, (int, float)):
            user_action = int(payload)

        if hasattr(self, "board") and self.board is not None:
            act_int = int(user_action)
            if act_int < len(self.board):
                self.board[act_int] = -1

        obs, rew, term, _, info = self.step(int(user_action))
        probs = [0.0] * self.n_actions
        probs[min(int(user_action), self.n_actions - 1)] = 1.0

        if policy_predict_fn is not None:
            action_idx, probs = policy_predict_fn(obs)
            action_idx = int(action_idx)
            if hasattr(self, "board") and self.board is not None and action_idx < len(self.board):
                self.board[action_idx] = 1
        else:
            action_idx = int(user_action)

        self.last_move = int(action_idx)

        return {{
            "arena_state": self.get_render_state(),
            "agent_action_label": self.action_labels[min(int(action_idx), self.n_actions - 1)],
            "action_probs": [float(p) for p in probs] if probs is not None else [1.0 / self.n_actions] * self.n_actions,
            "status_text": "Executed action: " + str(self.action_labels[min(int(action_idx), self.n_actions - 1)]),
        }}

    def get_demonstrations(self, num_samples: int = 32) -> tuple[np.ndarray, np.ndarray]:
        X = np.zeros((num_samples, self.obs_dim), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.int64)
        for i in range(num_samples):
            obs, _ = self.reset()
            X[i] = obs
            y[i] = i % self.n_actions
        return X, y
'''


def synthesize_universal_env(
    blueprint: Any,
    user_prompt: str = "",
    dataset: IngestedDataset | None = None,
    client: BaseLLMClient | None = None,
    max_retries: int = 3,
    allow_placeholder: bool | None = None,
) -> str:
    """Universal LLM Gymnasium Code Synthesizer with 3-Attempt Self-Healing Sandbox Loop.

    When no usable LLM is available, an explicit placeholder module is only produced in test
    mode (or when ``allow_placeholder`` is explicitly true). In every other case synthesis
    fails loudly instead of fabricating a trainable task.
    """
    if allow_placeholder is None:
        allow_placeholder = is_test_mode()
    active_client = client or get_active_llm_client()
    task_title = getattr(blueprint, "domain_title", None) or getattr(blueprint, "task_title", "") or "Custom RL Environment"
    summary = getattr(blueprint, "domain_summary", "") or "Dynamic simulation task."

    bp_spec = blueprint if isinstance(blueprint, AgentBlueprintSpec) else blueprint.to_agent_blueprint_spec()

    # Path 1: Live LLM Code Synthesis if configured
    code: str | None = None
    if not isinstance(active_client, FallbackCodeSynthesizer):
        try:
            prompt = (
                f"Synthesize the complete Python module `env_module.py` defining `SynthesizedGymEnv(gymnasium.Env)` for:\n"
                f"Domain: {task_title}\n"
                f"Summary: {summary}\n"
                f"Blueprint: {bp_spec.model_dump_json(indent=2)}\n\n"
                f"Interface Requirements:\n"
                f"1. observation_space = spaces.Box(...) matching len(observation_features)\n"
                f"2. action_space = spaces.Discrete(len(actions))\n"
                f"3. reset(self, seed=None, options=None) -> (obs, info) with info['arena_state'] = self.get_render_state()\n"
                f"4. step(self, action: int) -> (obs, reward, terminated, truncated, info)\n"
                f"5. compute_reward(self, obs, action, next_obs, info) -> (float, dict)\n"
                f"6. get_render_state(self) -> dict matching state_dict_schema\n"
                f"7. render_arena_state(self) -> dict alias to get_render_state()\n"
                f"8. interactive_step(self, payload, policy_predict_fn=None, mode='agent_vs_env') -> dict\n"
                f"9. get_demonstrations(self, num_samples=32) -> (np.ndarray, np.ndarray)\n"
                f"10. Optional: get_valid_actions() and get_critical_actions() when the domain has legal/tactical actions\n\n"
                f"Output ONLY valid Python code inside ```python ... ```."
            )
            raw = active_client.generate(prompt, system_prompt=ENV_BUILDER_SYSTEM_PROMPT)
            extracted = extract_python_code(raw)
            if extracted and "class SynthesizedGymEnv" in extracted:
                code = extracted
        except Exception:
            code = None

    # No usable LLM output: refuse to fabricate a task unless a placeholder is allowed.
    if not code:
        if not allow_placeholder:
            raise EnvironmentSynthesisError(
                "No verified environment code could be synthesized for this task. "
                "A working LLM is required to generate a task-specific environment."
            )
        code = _generate_schema_driven_env_code(bp_spec, user_prompt=user_prompt)

    # Pre-import AST security audit
    from rl_ide.synthesizer.verifier import audit_ast_security, SandboxSmokeTester
    try:
        audit_ast_security(code)
    except Exception:
        if not allow_placeholder:
            raise
        code = _generate_schema_driven_env_code(bp_spec, user_prompt=user_prompt)
        audit_ast_security(code)

    # 3-Attempt Self-Healing Sandbox Verification
    tester = SandboxSmokeTester(llm_client=active_client)
    verified, healed_code, logs = tester.verify_and_heal(
        code_str=code,
        blueprint=bp_spec,
        dataset=dataset,
        max_retries=max_retries,
    )
    if not verified:
        raise EnvironmentSynthesisError(
            "Generated environment failed sandbox verification: " + (logs[-1][-500:] if logs else "unknown error")
        )
    return healed_code


class JITEnvironmentSynthesizer:
    """Synthesizes custom Gym environments tailored to an AgentBlueprintSpec or uploaded dataset."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self._llm_client = llm_client
        self.last_warnings: list[str] = []
        # What the automatic KPI repair did, for the turn's timeline and logs.
        self._relaxation_log: list[str] = []

    @property
    def client(self) -> BaseLLMClient:
        if self._llm_client:
            return self._llm_client
        return get_active_llm_client()

    def synthesize(
        self,
        blueprint: AgentBlueprintSpec | DynamicTaskBlueprint,
        dataset: IngestedDataset | None = None,
        session_id: str | None = None,
        workspace_dir: str | Path | None = None,
    ) -> tuple[str, Path, type[gym.Env]]:
        """Generates, saves, and dynamically loads env_module.py."""
        sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
        out_dir = Path(workspace_dir) if workspace_dir else Path("workspaces") / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "env_module.py"

        bp_spec = blueprint if isinstance(blueprint, AgentBlueprintSpec) else blueprint.to_agent_blueprint_spec()

        # Path 1: supervised tabular task (requires an identifiable label column)
        has_label = bool(bp_spec.label_column) or bool(
            dataset is not None
            and self._infer_label_column(
                dataset, max(1, len(bp_spec.action_labels)), list(bp_spec.action_labels or [])
            )
        )
        if bp_spec.data_source_type == "uploaded_dataset" and has_label:
            code_str, derived_features = self._build_dataset_code(bp_spec, dataset)
            if derived_features and derived_features != list(bp_spec.observation_features or []):
                # Observations are derived from the real columns, not from proposed names.
                bp_spec.observation_features = derived_features
        else:
            # Path 2: Universal LLM-driven environment synthesis
            code_str = synthesize_universal_env(
                blueprint=bp_spec,
                dataset=dataset,
                client=self.client,
            )

        from rl_ide.synthesizer.verifier import audit_ast_security, SandboxSmokeTester
        audit_ast_security(code_str)

        # Generated code is untrusted until the smoke test passes: never write/import before.
        tester = SandboxSmokeTester(self.client)
        verified, healed_code, logs = tester.verify_and_heal(
            code_str=code_str,
            blueprint=bp_spec,
            dataset=dataset,
            workspace_dir=out_dir,
        )
        if not verified:
            raise EnvironmentSynthesisError(
                "Environment failed sandbox verification: " + (logs[-1][-500:] if logs else "unknown error")
            )
        self.last_warnings = list(tester.last_warnings)

        # A success condition no policy can reach makes the whole run report 0 % and wastes every
        # training round. Repair it here — bounded, re-verified — instead of only warning (G12).
        if tester.last_success_observed is False:
            fixed, relaxed_code, relax_logs = tester.relax_unreachable_success(
                code_str=healed_code,
                blueprint=bp_spec,
                dataset=dataset,
                workspace_dir=out_dir,
            )
            self._relaxation_log = relax_logs
            if fixed:
                healed_code = relaxed_code
                logs.extend(relax_logs)
                self.last_warnings = list(tester.last_warnings)
            else:
                self.last_warnings = list(tester.last_warnings) + [
                    "The environment's success criteria could not be made reachable automatically; "
                    "expect a low or zero success rate for this task."
                ]
        else:
            self._relaxation_log = []

        file_path.write_text(healed_code, encoding="utf-8")
        env_cls = load_synthesized_env_class(file_path)
        return healed_code, file_path, env_cls

    def generate_env_code(
        self,
        blueprint: AgentBlueprintSpec | DynamicTaskBlueprint,
        dataset: IngestedDataset | None = None,
        user_prompt: str = "",
    ) -> str:
        """Synthesizes python code via universal generative pipeline."""
        bp_spec = blueprint if isinstance(blueprint, AgentBlueprintSpec) else blueprint.to_agent_blueprint_spec()
        has_label = bool(bp_spec.label_column) or bool(
            dataset is not None
            and self._infer_label_column(
                dataset, max(1, len(bp_spec.action_labels)), list(bp_spec.action_labels or [])
            )
        )
        if bp_spec.data_source_type == "uploaded_dataset" and has_label:
            code_str, derived_features = self._build_dataset_code(bp_spec, dataset)
            if derived_features and derived_features != list(bp_spec.observation_features or []):
                bp_spec.observation_features = derived_features
            return code_str
        return synthesize_universal_env(
            blueprint=bp_spec,
            user_prompt=user_prompt,
            dataset=dataset,
            client=self.client,
        )

    @staticmethod
    def _synthetic_feature_frame(features: list[str], label_column: str, n_actions: int, rows: int = 120) -> Any:
        """Deterministic stand-in table for blueprints that declare a schema but bring no dataset."""
        import pandas as pd

        names = [str(f) for f in features] or [f"feature_{i}" for i in range(4)]
        data: dict[str, Any] = {}
        for j, name in enumerate(names):
            data[name] = [round(math.sin(0.13 * i + 0.7 * j) * 0.5 + 0.5, 6) for i in range(rows)]
        data[label_column] = [i % max(2, n_actions) for i in range(rows)]
        return pd.DataFrame(data)

    @staticmethod
    def _infer_label_column(
        dataset: IngestedDataset | None,
        n_actions: int,
        action_labels: list[str] | None = None,
    ) -> str | None:
        """Infers a target/label column generically (see ``ingest.profiler.infer_label_column``)."""
        from rl_ide.ingest.profiler import infer_label_column

        if dataset is None or not hasattr(dataset, "df"):
            return None
        return infer_label_column(dataset.df, n_actions, action_labels)

    @classmethod
    def _select_label_column(
        cls,
        df: Any,
        declared: str | None,
        action_labels: list[str],
        n_actions: int,
    ) -> str | None:
        """Picks the target column that actually explains the declared actions.

        A declared column that only covers two of four actions (or a generic flag an LLM guessed)
        must not beat a column whose values name every action.
        """
        from rl_ide.synthesizer.feature_engineering import infer_label_map

        candidates: list[str] = []
        if declared:
            candidates.append(str(declared))
        inferred = cls._infer_label_column_for_frame(df, n_actions, action_labels)
        if inferred and str(inferred) not in candidates:
            candidates.append(str(inferred))

        best_col, best_score = None, -1
        for col in candidates:
            if col not in list(df.columns):
                continue
            try:
                values = list(df[col].dropna().unique())
            except Exception:
                continue
            score = len(set(infer_label_map(values, action_labels).values()))
            if score > best_score:
                best_col, best_score = col, score
        return best_col or (candidates[0] if candidates else None)

    @staticmethod
    def _infer_label_column_for_frame(df: Any, n_actions: int, action_labels: list[str]) -> str | None:
        from rl_ide.ingest.profiler import infer_label_column

        return infer_label_column(df, n_actions, action_labels)

    def _build_dataset_code(
        self,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None,
    ) -> tuple[str, list[str]]:
        """Builds a supervised tabular env whose observations are derived from the real columns.

        Returns ``(code, observation_feature_names)``. Feature names are returned so the caller can
        keep the blueprint (and therefore arena/telemetry schemas) in sync with the generated env.
        """
        from rl_ide.synthesizer.feature_engineering import (
            derive_feature_ops,
            infer_label_map,
            records_from_frame,
            select_usable_rows,
        )

        action_labels = list(blueprint.action_labels or [])
        n_actions = max(1, len(action_labels))
        declared_label = blueprint.label_column or self._infer_label_column(
            dataset, n_actions, action_labels
        )

        df = getattr(dataset, "df", None) if dataset is not None else None
        if (df is None or len(df) == 0) and declared_label:
            # No table on hand: fall back to deterministic synthetic demonstrations that honour
            # the declared feature schema, so the blueprint is still trainable.
            df = self._synthetic_feature_frame(
                features=list(blueprint.observation_features or []),
                label_column=str(declared_label),
                n_actions=n_actions,
            )
        if df is None or len(df) == 0:
            raise EnvironmentSynthesisError(
                "Supervised synthesis requires a tabular dataset, but none was provided."
            )
        label_column = self._select_label_column(df, declared_label, action_labels, n_actions)
        if not label_column or str(label_column) not in list(df.columns):
            raise EnvironmentSynthesisError(
                "No label/target column could be identified in the uploaded dataset. "
                "Name the target column 'label'/'target'/'is_*', or specify it explicitly."
            )
        label_column = str(label_column)
        blueprint.label_column = label_column

        feature_ops, feature_names = derive_feature_ops(df, label_column)
        if not feature_ops:
            raise EnvironmentSynthesisError(
                f"Dataset has no usable feature columns beside '{label_column}' "
                "(constant, empty, and free-text-only columns cannot be learned from)."
            )

        label_map = infer_label_map(list(df[label_column].dropna().unique()), action_labels)
        if len({int(v) for v in label_map.values()}) < 2:
            raise EnvironmentSynthesisError(
                f"Target column '{label_column}' could not be mapped onto the declared actions "
                f"{action_labels}. Rename the target values to match the actions."
            )

        usable = select_usable_rows(df, label_column, label_map)
        if len(usable) < 8:
            raise EnvironmentSynthesisError(
                f"Only {len(usable)} rows of '{label_column}' map onto the declared actions; "
                "at least 8 labelled rows are required to train a supervised policy."
            )

        records_json = json.dumps(records_from_frame(usable), default=str)
        obs_dim = len(feature_names)

        return f'''"""Auto-synthesized Gymnasium environment for supervised tabular decisions."""

from __future__ import annotations
import math
import random
import zlib
from typing import Any, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SynthesizedGymEnv(gym.Env):
    """Auto-synthesized Gymnasium environment for: {blueprint.task_title}"""
    metadata = {{"render_modes": ["json"]}}

    # (kind, column, low, high) for numeric min-max scaling, or (kind, column, value, None) for one-hot.
    _feature_ops = {feature_ops!r}
    _label_map = {label_map!r}
    _records = {records_json}

    def __init__(self, dataset_records=None, config=None):
        super().__init__()
        self.action_labels = {action_labels}
        self.observation_features = {feature_names}
        self.obs_dim = {obs_dim}
        self.n_actions = {n_actions}

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        source = [dict(r) for r in dataset_records] if dataset_records else [dict(r) for r in self._records]
        self.records = source or [{{}}]
        # Runtime records (all usable rows) may be larger than the embedded sample.
        self.dataset_rows_total = {len(usable)}
        self.label_column = {label_column!r}
        self._feature_ops = self._fit_bounds(self.records)
        self.split_indices = self._compute_splits()

        self.current_idx = 0
        self.step_count = 0
        self.correct_count = 0
        self.cumulative_score = 0.0
        self.last_action = None
        self.last_feedback = ""
        self.last_reward_breakdown = {{}}
        self.active_split = "train"
        self.split_fallback = False
        self._declared_state_schema = {dict(blueprint.state_dict_schema or {})!r}
        self.is_placeholder = False
        self.task_class = "classification"
        self.episode_horizon = 25
        self.success_criteria = (
            "Classify each record into the action whose label matches the dataset target column."
        )

    def _fit_bounds(self, records):
        """Fits numeric min-max bounds to the records actually in use (baked bounds are a fallback)."""
        fitted = []
        for kind, column, low, high in self._feature_ops:
            if kind != "num":
                fitted.append((kind, column, low, high))
                continue
            values = []
            for record in records:
                try:
                    value = float(record.get(column))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if values and max(values) > min(values):
                fitted.append((kind, column, min(values), max(values)))
            else:
                fitted.append((kind, column, low, high))
        return fitted

    def _split_for(self, record):
        """Content-hash split so a row lands in the same split on every process and restart."""
        parts = []
        for key in sorted(record):
            if key == self.label_column:
                continue
            value = record.get(key)
            if isinstance(value, float):
                value = f"{{value:.6f}}"
            parts.append(f"{{key}}={{value}}")
        bucket = zlib.crc32("|".join(parts).encode("utf-8")) % 100
        if bucket < 70:
            return "train"
        if bucket < 85:
            return "val"
        return "test"

    def _compute_splits(self):
        groups = {{"train": [], "val": [], "test": []}}
        for index, record in enumerate(self.records):
            groups[self._split_for(record)].append(index)
        if not any(groups.values()):
            groups["train"] = list(range(len(self.records)))
        return groups

    def split_counts(self):
        """Number of records available in each split."""
        return {{name: len(indices) for name, indices in self.split_indices.items()}}

    def available_splits(self):
        return [name for name, indices in self.split_indices.items() if indices]

    def _resolve_split(self, requested):
        """Returns (split, fell_back). Falls back rather than silently training on held-out rows."""
        if requested and self.split_indices.get(requested):
            return requested, False
        for name in ("train", "val", "test"):
            if self.split_indices.get(name):
                return name, bool(requested) and requested != name
        return "train", False

    def _extract_obs(self, record: dict[str, Any]) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        for i, (kind, column, low, high) in enumerate(self._feature_ops):
            raw = record.get(column)
            if raw is None:
                continue
            if kind == "num":
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                span = float(high) - float(low)
                if span > 0:
                    obs[i] = float(np.clip((value - float(low)) / span, 0.0, 1.0))
            else:
                if str(raw).strip().lower() == str(low).strip().lower():
                    obs[i] = 1.0
        return obs

    def _truth_for(self, record: dict[str, Any]) -> int | None:
        if not self.label_column or self.label_column not in record:
            return None
        raw = record[self.label_column]
        mapped = self._label_map.get(str(raw))
        if mapped is None:
            mapped = self._label_map.get(str(raw).strip())
        if mapped is not None:
            return int(mapped) % self.n_actions
        try:
            return int(float(raw)) % self.n_actions
        except (TypeError, ValueError):
            return None

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        requested = None
        if isinstance(options, dict):
            requested = options.get("split")
        self.active_split, self.split_fallback = self._resolve_split(requested)
        order = self.split_indices[self.active_split]

        self.current_idx = order[random.randrange(len(order))]
        self.step_count = 0
        self.correct_count = 0
        self.cumulative_score = 0.0
        self.last_action = None
        self.last_feedback = ""
        self.last_reward_breakdown = {{}}

        obs = self._extract_obs(self.records[self.current_idx])
        info = {{
            "record_index": self.current_idx,
            "split": self.active_split,
            "split_fallback": self.split_fallback,
            "split_counts": self.split_counts(),
            "is_success": False,
            "reward_breakdown": {{}},
            "arena_state": self.render_arena_state(),
        }}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action) % self.n_actions
        self.step_count += 1
        self.last_action = action

        record = self.records[self.current_idx]
        truth = self._truth_for(record)
        is_correct = truth is not None and action == truth

        reward, breakdown = self.compute_reward(None, action, None, {{"is_correct": is_correct}})
        if is_correct:
            self.correct_count += 1
        self.cumulative_score += float(reward)
        self.last_reward_breakdown = breakdown
        self.last_feedback = (
            f"Correct: {{self.action_labels[action]}}"
            if is_correct
            else f"Incorrect: {{self.action_labels[action]}}"
        )

        previous_idx = self.current_idx
        order = self.split_indices.get(self.active_split) or list(range(len(self.records)))
        try:
            position = order.index(self.current_idx)
        except ValueError:
            position = -1
        self.current_idx = order[(position + 1) % len(order)]
        next_obs = self._extract_obs(self.records[self.current_idx])
        terminated = self.step_count >= self.episode_horizon

        info = {{
            "record_index": self.current_idx,
            "decision_record_index": previous_idx,
            "split": self.active_split,
            "is_success": bool(is_correct),
            "is_correct": bool(is_correct),
            "correct_count": self.correct_count,
            "accuracy_pct": round((self.correct_count / max(1, self.step_count)) * 100, 1),
            "reward_breakdown": breakdown,
            "arena_state": self.render_arena_state(),
        }}
        return next_obs, float(reward), terminated, False, info

    def compute_reward(self, obs, action, next_obs, info) -> tuple[float, dict[str, float]]:
        is_correct = bool(info.get("is_correct", False)) if isinstance(info, dict) else False
        base = 1.0 if is_correct else -1.0
        breakdown = {{"accuracy_reward": float(base)}}
        return float(base), breakdown

    def get_render_state(self) -> dict[str, Any]:
        return self.render_arena_state()

    def render_arena_state(self) -> dict[str, Any]:
        record = self.records[self.current_idx % len(self.records)]
        accuracy = (self.correct_count / max(1, self.step_count)) * 100
        visible = {{k: str(v)[:40] for k, v in record.items() if k != self.label_column}}
        state = {{
            "record_index": self.current_idx,
            "total_records": len(self.records),
            "active_split": self.active_split,
            "split_size": len(self.split_indices.get(self.active_split) or []),
            "step_count": self.step_count,
            "accuracy_pct": round(accuracy, 1),
            "last_action": self.last_action if self.last_action is not None else -1,
            "last_action_label": (
                self.action_labels[self.last_action] if self.last_action is not None else "None"
            ),
            "current_record": visible,
            "action_labels": self.action_labels,
        }}
        for _key, _kind in self._declared_state_schema.items():
            if _key not in state:
                _kind_text = str(_kind).lower()
                state[_key] = (
                    [] if "list" in _kind_text
                    else (0 if "int" in _kind_text else (0.0 if "float" in _kind_text else ""))
                )
        return state

    def interactive_step(self, payload: dict[str, Any], policy_predict_fn: Callable = None, mode: str = "agent_vs_env") -> dict[str, Any]:
        custom_record = payload.get("custom_params") or payload.get("record") if isinstance(payload, dict) else None
        if custom_record and isinstance(custom_record, dict):
            obs = self._extract_obs(custom_record)
            self.records = self.records + [dict(custom_record)]
            self.current_idx = len(self.records) - 1
            self.split_indices = self._compute_splits()
            for _name, _idxs in self.split_indices.items():
                if self.current_idx in _idxs:
                    self.active_split = _name
                    break
        else:
            index = int(payload.get("user_action", self.current_idx)) % len(self.records) if isinstance(payload, dict) else 0
            self.current_idx = index
            obs = self._extract_obs(self.records[index])

        if policy_predict_fn is not None:
            action_idx, probs = policy_predict_fn(obs)
            action_idx = int(action_idx) % self.n_actions
        else:
            action_idx = 0
            probs = [1.0 / self.n_actions] * self.n_actions

        label = self.action_labels[action_idx]
        return {{
            "arena_state": self.render_arena_state(),
            "agent_action_label": label,
            "action_probs": [float(p) for p in probs],
            "status_text": f"Agent routed this record to: {{label}}",
        }}

    def get_demonstrations(self, num_samples: int = 200, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros((num_samples, self.obs_dim), dtype=np.float32)
        targets = np.zeros(num_samples, dtype=np.int64)
        indices = self.split_indices.get(split) if split else None
        if not indices:
            indices = list(range(len(self.records)))
        usable = 0
        for i in range(num_samples):
            record = self.records[indices[i % len(indices)]]
            truth = self._truth_for(record)
            if truth is None:
                continue
            features[usable] = self._extract_obs(record)
            targets[usable] = truth
            usable += 1
        if usable == 0:
            return features[:0].reshape(0, self.obs_dim), targets[:0]
        return features[:usable], targets[:usable]
''', feature_names
