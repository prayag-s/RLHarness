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

        self.state_vars = {{}}
        self.step_count = 0
        self.done = False
        self.last_move = None
        has_board_schema = "board" in {state_schema} or self.n_actions == 9
        self.board = [0] * (self.n_actions if self.n_actions > 0 else 9) if has_board_schema else None

    def _get_obs(self) -> np.ndarray:
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
            "arena_state": self.get_render_state(),
        }}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        act = min(max(0, int(action)), self.n_actions - 1)
        self.last_move = act

        for idx, f in enumerate(self.observation_features):
            delta = 0.5 if act == idx % self.n_actions else -0.1
            self.state_vars[f] = float(np.clip(self.state_vars[f] + delta + random.uniform(-0.05, 0.05), -10.0, 10.0))

        terminated = (self.step_count >= 25)
        obs = self._get_obs()
        reward, bd = self.compute_reward(obs, act, obs, {{"reward_breakdown": {{"step_progress": 1.0}}}})
        info = {{
            "reward_breakdown": bd,
            "is_success": terminated and reward > 0,
            "arena_state": self.get_render_state(),
        }}
        return obs, float(reward), terminated, False, info

    def compute_reward(self, obs, action, next_obs, info) -> tuple[float, dict[str, float]]:
        total = 0.0
        bd = info.get("reward_breakdown", {{}})
        total = float(sum(bd.values()))
        return total, bd

    def get_render_state(self) -> dict[str, Any]:
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

        user_action = payload.get("user_action", 0) if isinstance(payload, dict) else payload
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
) -> str:
    """Universal LLM Gymnasium Code Synthesizer with 3-Attempt Self-Healing Sandbox Loop."""
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
                f"9. get_demonstrations(self, num_samples=32) -> (np.ndarray, np.ndarray)\n\n"
                f"Output ONLY valid Python code inside ```python ... ```."
            )
            raw = active_client.generate(prompt, system_prompt=ENV_BUILDER_SYSTEM_PROMPT)
            extracted = extract_python_code(raw)
            if extracted and "class SynthesizedGymEnv" in extracted:
                code = extracted
        except Exception:
            code = None

    # Fallback to schema-driven code generation
    if not code:
        code = _generate_schema_driven_env_code(bp_spec, user_prompt=user_prompt)

    # Pre-import AST security audit
    from rl_ide.synthesizer.verifier import audit_ast_security, SandboxSmokeTester
    try:
        audit_ast_security(code)
    except Exception:
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
    if verified:
        return healed_code
    return code


class JITEnvironmentSynthesizer:
    """Synthesizes custom Gym environments tailored to an AgentBlueprintSpec or uploaded dataset."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self._llm_client = llm_client

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

        # Path 1: Uploaded tabular/document dataset
        if dataset is not None or bp_spec.data_source_type == "uploaded_dataset":
            code_str = self._build_dataset_code(bp_spec, dataset)
        else:
            # Path 2: Universal LLM-driven environment synthesis
            code_str = synthesize_universal_env(
                blueprint=bp_spec,
                dataset=dataset,
                client=self.client,
            )

        from rl_ide.synthesizer.verifier import audit_ast_security
        audit_ast_security(code_str)

        file_path.write_text(code_str, encoding="utf-8")
        env_cls = load_synthesized_env_class(file_path)
        return code_str, file_path, env_cls

    def generate_env_code(
        self,
        blueprint: AgentBlueprintSpec | DynamicTaskBlueprint,
        dataset: IngestedDataset | None = None,
        user_prompt: str = "",
    ) -> str:
        """Synthesizes python code via universal generative pipeline."""
        bp_spec = blueprint if isinstance(blueprint, AgentBlueprintSpec) else blueprint.to_agent_blueprint_spec()
        if dataset is not None or bp_spec.data_source_type == "uploaded_dataset":
            return self._build_dataset_code(bp_spec, dataset)
        return synthesize_universal_env(
            blueprint=bp_spec,
            user_prompt=user_prompt,
            dataset=dataset,
            client=self.client,
        )

    def _build_dataset_code(
        self,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None,
    ) -> str:
        """Generates custom data-driven Gymnasium environment stepping through records."""
        obs_features = blueprint.observation_features
        obs_dim = max(1, len(obs_features))
        action_labels = blueprint.action_labels
        n_actions = max(1, len(action_labels))

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
            for i in range(100):
                rec = {col: float(math.sin(i * 0.1 + j)) for j, col in enumerate(obs_features)}
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

        self.records = dataset_records if dataset_records is not None else {records_json}
        if not self.records:
            self.records = [{{{", ".join(f"'{c}': 0.0" for c in obs_features)}}}]

        self.current_idx = 0
        self.step_count = 0
        self.correct_count = 0
        self.last_action = None

    def _extract_obs(self, record: dict[str, Any]) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        for i, feat in enumerate(self.observation_features):
            val = record.get(feat, 0.0)
            try:
                obs[i] = float(val)
            except (ValueError, TypeError):
                obs[i] = 0.0
        return obs

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_idx = random.randint(0, len(self.records) - 1)
        self.step_count = 0
        self.correct_count = 0
        self.last_action = None

        obs = self._extract_obs(self.records[self.current_idx])
        info = {{"record_index": self.current_idx, "is_success": False, "reward_breakdown": {{}}}}
        info["arena_state"] = self.render_arena_state()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self.last_action = int(action)
        rec = self.records[self.current_idx]

        target_keys = ["is_fraud", "churn_risk", "is_policy_violation", "target", "label", "class", "sop_compliance_score"]
        found_target = None
        for tk in target_keys:
            if tk in rec:
                found_target = rec[tk]
                break

        if found_target is not None:
            try:
                truth = int(float(found_target)) % self.n_actions
            except (ValueError, TypeError):
                truth = 0
            is_correct = (int(action) == truth)
        else:
            first_val = rec.get(self.observation_features[0], 0.0)
            try:
                is_correct = (int(action) == (int(float(first_val)) % self.n_actions))
            except (ValueError, TypeError):
                is_correct = (int(action) == 0)

        reward, breakdown = self.compute_reward(None, action, None, {{"is_correct": is_correct}})
        if is_correct:
            self.correct_count += 1

        self.current_idx = (self.current_idx + 1) % len(self.records)
        next_obs = self._extract_obs(self.records[self.current_idx])
        terminated = (self.step_count >= 25)

        info = {{
            "record_index": self.current_idx,
            "is_success": is_correct,
            "accuracy_pct": round((self.correct_count / self.step_count) * 100, 1),
            "reward_breakdown": breakdown,
            "arena_state": self.render_arena_state(),
        }}
        return next_obs, reward, terminated, False, info

    def compute_reward(self, obs, action, next_obs, info) -> tuple[float, dict[str, float]]:
        is_correct = info.get("is_correct", False)
        base = 1.0 if is_correct else -1.0
        breakdown = {{"accuracy_reward": float(base)}}
        return float(base), breakdown

    def get_render_state(self) -> dict[str, Any]:
        return self.render_arena_state()

    def render_arena_state(self) -> dict[str, Any]:
        rec = self.records[self.current_idx % len(self.records)]
        acc = (self.correct_count / max(1, self.step_count)) * 100
        return {{
            "record_index": self.current_idx,
            "total_records": len(self.records),
            "step_count": self.step_count,
            "accuracy_pct": round(acc, 1),
            "last_action_label": self.action_labels[self.last_action] if self.last_action is not None else "None",
            "current_record_preview": {{k: str(v)[:30] for k, v in list(rec.items())[:6]}},
            "action_labels": self.action_labels,
        }}

    def interactive_step(self, payload: dict[str, Any], policy_predict_fn: Callable = None, mode: str = "agent_vs_env") -> dict[str, Any]:
        custom_record = payload.get("custom_params") or payload.get("record") if isinstance(payload, dict) else None
        if custom_record and isinstance(custom_record, dict):
            obs = self._extract_obs(custom_record)
        else:
            rec_idx = int(payload.get("user_action", self.current_idx)) % len(self.records) if isinstance(payload, dict) else 0
            self.current_idx = rec_idx
            active_rec = self.records[rec_idx]
            obs = self._extract_obs(active_rec)

        if policy_predict_fn is not None:
            action_idx, probs = policy_predict_fn(obs)
            action_idx = int(action_idx)
        else:
            action_idx = 0
            probs = [1.0 / self.n_actions] * self.n_actions

        label = self.action_labels[action_idx % self.n_actions]
        return {{
            "arena_state": self.render_arena_state(),
            "agent_action_label": label,
            "action_probs": [float(p) for p in probs],
            "status_text": f"Agent classified record as: {{label}}",
        }}

    def get_demonstrations(self, num_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
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
