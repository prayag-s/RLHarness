"""Sandbox Smoke Tester & Self-Healing Verifier for RL-Native Agent IDE v4.0.

Executes static AST security audits and dynamic runtime smoke tests on
synthesized Gymnasium environments, with automatic multi-round LLM self-healing.
"""

from __future__ import annotations
import ast
import json
import math
import re
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.ingest.loader import IngestedDataset
from rl_ide.llm.client import BaseLLMClient, create_llm_client, FallbackCodeSynthesizer, get_active_llm_client
from rl_ide.synthesizer.env_builder import load_synthesized_env_class, instantiate_env


class SecurityViolationError(Exception):
    """Raised when synthesized code contains prohibited modules or AST calls."""
    pass


FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "urllib", "requests",
    "httpx", "http", "ftplib", "builtins", "importlib", "pickle", "ctypes"
}

FORBIDDEN_CALLS = {
    "system", "popen", "spawn", "eval", "exec", "__import__",
    "remove", "unlink", "rmdir", "rmtree", "open"
}


def audit_ast_security(code_str: str) -> None:
    """Statically verifies that generated code imports and calls only safe modules."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise SyntaxError(f"Synthesized code failed syntax parsing: {e}")

    for node in ast.walk(tree):
        # 1. Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in FORBIDDEN_MODULES:
                    raise SecurityViolationError(f"Prohibited import: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in FORBIDDEN_MODULES:
                    raise SecurityViolationError(f"Prohibited from-import: '{node.module}'")

        # 2. Check forbidden function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    raise SecurityViolationError(f"Prohibited function call: '{node.func.id}()'")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS:
                    raise SecurityViolationError(f"Prohibited method call: '.{node.func.attr}()'")


class SandboxSmokeTester:
    """Verifies synthesized Gymnasium environments and repairs code via self-healing loop."""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self._llm_client = llm_client
        self.last_warnings: list[str] = []

    @property
    def client(self) -> BaseLLMClient:
        if self._llm_client:
            return self._llm_client
        return get_active_llm_client()

    def run_smoke_test(
        self,
        env_cls: type[gym.Env],
        dataset_records: list[dict[str, Any]] | None = None,
        expected_state_keys: list[str] | None = None,
        expected_obs_dim: int | None = None,
        expected_n_actions: int | None = None,
    ) -> dict[str, Any]:
        """Runs thorough runtime smoke test on a Gymnasium environment class."""
        # 1. Instantiate
        env = instantiate_env(env_cls, dataset_records=dataset_records)

        # 2. Check Spaces
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(f"env.observation_space must be gym.spaces.Box, got {type(env.observation_space)}")
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError(f"env.action_space must be gym.spaces.Discrete, got {type(env.action_space)}")

        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        # 2b. Declared contract: labels must align with the actual spaces
        if expected_obs_dim is not None and int(obs_shape[0]) != int(expected_obs_dim):
            raise ValueError(
                f"observation_space shape {obs_shape} does not match {expected_obs_dim} declared observation features"
            )
        if expected_n_actions is not None and int(n_actions) != int(expected_n_actions):
            raise ValueError(
                f"action_space.n {n_actions} does not match {expected_n_actions} declared actions"
            )
        labels = getattr(env, "action_labels", None)
        if labels is not None and len(labels) != n_actions:
            raise ValueError(f"env.action_labels has {len(labels)} entries but action_space.n is {n_actions}")

        declared_keys = [k for k in (expected_state_keys or []) if k not in ("status", "step_count")]

        # 3. Test Reset
        obs, info = env.reset()
        if not isinstance(obs, np.ndarray):
            raise TypeError(f"reset() must return numpy.ndarray observation, got {type(obs)}")
        if obs.shape != obs_shape:
            raise ValueError(f"reset() obs.shape {obs.shape} does not match observation_space.shape {obs_shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError("reset() observation contains NaNs or infinite values")
        if not isinstance(info, dict):
            raise TypeError(f"reset() info must be a dict, got {type(info)}")

        # 4. Step through 25 actions
        snapshot: list[dict[str, Any]] = []
        step_infos: list[dict[str, Any]] = []
        observed: list[np.ndarray] = [np.asarray(obs, dtype=np.float64)]
        for step_idx in range(25):
            action = env.action_space.sample()
            step_res = env.step(action)
            if len(step_res) != 5:
                raise ValueError(f"step() must return 5-tuple (obs, reward, terminated, truncated, info), got {len(step_res)}")
            step_obs, reward, term, trunc, step_info = step_res

            if step_obs.shape != obs_shape:
                raise ValueError(f"step() obs.shape {step_obs.shape} does not match {obs_shape}")
            if not np.all(np.isfinite(step_obs)):
                raise ValueError(f"step() observation contains NaNs or infinite values at step {step_idx}")
            observed.append(np.asarray(step_obs, dtype=np.float64))

            if not isinstance(reward, (float, int, np.floating)):
                raise TypeError(f"step() reward must be float/int, got {type(reward)}")
            if not math.isfinite(float(reward)):
                raise ValueError(f"step() reward is NaN or infinite at step {step_idx}")

            if not isinstance(term, (bool, np.bool_)):
                raise TypeError(f"step() terminated must be bool, got {type(term)}")
            if not isinstance(trunc, (bool, np.bool_)):
                raise TypeError(f"step() truncated must be bool, got {type(trunc)}")

            if not isinstance(step_info, dict):
                raise TypeError("step() info must be a dict")

            if "reward_breakdown" not in step_info:
                raise KeyError("step() info must contain 'reward_breakdown' dictionary")
            if not isinstance(step_info["reward_breakdown"], dict):
                raise TypeError("info['reward_breakdown'] must be a dictionary")

            if "is_success" not in step_info:
                raise KeyError("step() info must contain 'is_success' boolean")
            if not isinstance(step_info["is_success"], (bool, np.bool_)):
                raise TypeError("info['is_success'] must be a boolean")

            if "arena_state" not in step_info:
                raise KeyError("step() info must contain 'arena_state' dictionary")
            if not isinstance(step_info["arena_state"], dict):
                raise TypeError("info['arena_state'] must be a dictionary")
            # Verify JSON serializable
            try:
                json.dumps(step_info["arena_state"])
            except Exception as e:
                raise ValueError(f"info['arena_state'] is not JSON serializable: {e}")

            # Declared render-state contract (state_dict_schema is the shared source of truth)
            for key in declared_keys:
                if key not in step_info["arena_state"]:
                    raise KeyError(
                        f"arena_state is missing declared key '{key}' from state_dict_schema"
                    )
            render_state = dict(step_info["arena_state"])
            render_state.pop("step_count", None)
            snapshot.append(render_state)

            step_infos.append(step_info)
            if term or trunc:
                env.reset()

        # 5. Test get_demonstrations
        if not hasattr(env, "get_demonstrations"):
            raise AttributeError("env must implement get_demonstrations(num_samples)")
        X, y = env.get_demonstrations(num_samples=32)
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("get_demonstrations must return tuple of (np.ndarray, np.ndarray)")
        if X.ndim != 2 or X.shape[1] != obs_shape[0]:
            raise ValueError(f"Demonstration X shape {X.shape} must be (N, {obs_shape[0]})")
        if y.shape != (X.shape[0],):
            raise ValueError(f"Demonstration y shape {y.shape} must be ({X.shape[0]},)")
        if X.shape[0] < 1:
            raise ValueError("get_demonstrations returned no samples; a warm-start teacher is required")
        if not np.all(np.isfinite(X)):
            raise ValueError("Demonstrations X contains NaNs or infinite values")
        if not (np.all(y >= 0) and np.all(y < n_actions)):
            raise ValueError(f"Demonstrations y contains out-of-bounds actions (expected 0..{n_actions-1})")

        # 6. Test interactive_step
        if not hasattr(env, "interactive_step"):
            raise AttributeError("env must implement interactive_step(payload, policy_predict_fn)")

        def dummy_policy(obs):
            probs = [1.0 / n_actions] * n_actions
            return 0, probs

        interactive_res = env.interactive_step({}, dummy_policy)
        if not isinstance(interactive_res, dict):
            raise TypeError(f"interactive_step must return a dict, got {type(interactive_res)}")

        required_keys = ["arena_state", "agent_action_label", "action_probs", "status_text"]
        for k in required_keys:
            if k not in interactive_res:
                raise KeyError(f"interactive_step result missing required key: '{k}'")

        try:
            json.dumps(interactive_res)
        except Exception as e:
            raise ValueError(f"interactive_step result is not JSON serializable: {e}")

        # Declared state must actually change over an episode (a frozen render state is a bug).
        # Compare the whole sequence: short-horizon envs legitimately reset mid-probe, so checking
        # only the first/last snapshots would flag them as static.
        if len(snapshot) > 1:
            distinct = {json.dumps(item, sort_keys=True, default=str) for item in snapshot}
            if len(distinct) < 2:
                raise ValueError("arena_state never changes during an episode; declared state contract is static")

        # Dead observation vectors make a policy unlearnable: most dimensions must carry signal.
        obs_matrix = np.vstack(observed)
        degenerate = int(np.sum(np.std(obs_matrix, axis=0) < 1e-9))
        if obs_matrix.shape[1] > 0 and degenerate >= max(1, math.ceil(0.9 * obs_matrix.shape[1])):
            raise ValueError(
                f"{degenerate}/{obs_matrix.shape[1]} observation features are constant during the "
                "probe; the observation vector carries no learnable signal (check that feature "
                "names match real state fields)."
            )

        # Success reachability is reported as uncertainty, never silently assumed. A 25-step window
        # cannot see terminal-only success signals, so fall back to complete episodes.
        success_observed = any(bool(s.get("is_success")) for s in step_infos)
        if not success_observed and not getattr(env, "is_placeholder", False):
            success_observed = self._probe_success(env)

        warnings: list[str] = []
        if not success_observed and not getattr(env, "is_placeholder", False):
            warnings.append(
                "Success signal never observed across full probe episodes; the reported KPI may stay "
                "at 0% until the environment's success criteria are reachable."
            )

        return {
            "verified": True,
            "obs_dim": obs_shape[0],
            "n_actions": n_actions,
            "is_placeholder": bool(getattr(env, "is_placeholder", False)),
            "success_observed": success_observed,
            "warnings": warnings,
            "sample_arena_state": interactive_res["arena_state"],
        }

    @staticmethod
    def _probe_success(env: Any, episodes_per_policy: int = 3, max_steps: int = 200) -> bool:
        """Runs complete episodes to observe terminal-only success signals.

        Uses the environment's own ``get_critical_actions`` hint when available (many synthesized
        envs expose it) and falls back to random actions. Any observed success ends the probe.
        """
        horizon = getattr(env, "episode_horizon", None)
        if horizon is None:
            spec = getattr(env, "spec", None)
            horizon = getattr(spec, "max_episode_steps", None) if spec is not None else None
        try:
            horizon = int(horizon) if horizon is not None else 50
        except (TypeError, ValueError):
            horizon = 50
        horizon = max(1, min(horizon, max_steps))

        policies: list[Any] = []
        critical = getattr(env, "get_critical_actions", None)
        if callable(critical):
            def _critical_action() -> int:
                hinted = critical()
                return int(hinted[0]) if hinted else 0

            policies.append(_critical_action)
        policies.append(None)

        try:
            for policy in policies:
                for _ in range(episodes_per_policy):
                    _, info = env.reset()
                    if isinstance(info, dict) and info.get("is_success"):
                        return True
                    for _ in range(horizon):
                        action = policy() if policy is not None else int(env.action_space.sample())
                        _, _, terminated, truncated, info = env.step(action)
                        if isinstance(info, dict) and info.get("is_success"):
                            return True
                        if terminated or truncated:
                            break
        except Exception:
            return False
        return False

    def verify_and_heal(
        self,
        code_str: str,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None = None,
        workspace_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> tuple[bool, str, list[str]]:
        """Executes full verification with self-healing repair loop up to max_retries."""
        repair_logs: list[str] = []
        current_code = code_str

        # Ephemeral verification workspaces must not accumulate: previously every attempt left a
        # `workspaces/temp_verify_*` directory behind (609 of them on this machine).
        temporary_dir: tempfile.TemporaryDirectory | None = None
        if workspace_dir is None:
            temporary_dir = tempfile.TemporaryDirectory(prefix="rl_verify_")
            out_dir = Path(temporary_dir.name)
        else:
            out_dir = Path(workspace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        canonical_file = out_dir / "env_module.py"

        try:
            return self._verify_attempts(
                current_code=current_code,
                blueprint=blueprint,
                dataset=dataset,
                out_dir=out_dir,
                canonical_file=canonical_file,
                max_retries=max_retries,
                repair_logs=repair_logs,
            )
        finally:
            if temporary_dir is not None:
                temporary_dir.cleanup()

    def _verify_attempts(
        self,
        *,
        current_code: str,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None,
        out_dir: Path,
        canonical_file: Path,
        max_retries: int,
        repair_logs: list[str],
    ) -> tuple[bool, str, list[str]]:
        for attempt in range(max_retries + 1):
            try:
                # 1. AST Static Audit
                audit_ast_security(current_code)

                # 2. Invalidate caches and write unique attempt file to prevent .pyc stale reuse
                import importlib
                importlib.invalidate_caches()
                attempt_file = out_dir / f"env_module_try_{attempt}_{uuid.uuid4().hex[:6]}.py"
                attempt_file.write_text(current_code, encoding="utf-8")

                # 3. Dynamic Load
                env_cls = load_synthesized_env_class(attempt_file)

                # 4. Runtime Smoke Test
                records = dataset.df.to_dict(orient="records") if (dataset and hasattr(dataset, "df")) else None
                res = self.run_smoke_test(
                    env_cls,
                    dataset_records=records,
                    expected_state_keys=list(getattr(blueprint, "state_dict_schema", {}) or {}),
                    expected_obs_dim=len(getattr(blueprint, "observation_features", []) or []) or None,
                    expected_n_actions=len(getattr(blueprint, "action_labels", []) or []) or None,
                )

                # Canonicalize output file
                canonical_file.write_text(current_code, encoding="utf-8")

                self.last_warnings = list(res.get("warnings") or [])
                repair_logs.append(f"Attempt {attempt}: Verification PASSED 100% (25 steps, demos, interactive).")
                for warning in self.last_warnings:
                    repair_logs.append(f"Attempt {attempt} WARNING: {warning}")
                try:
                    attempt_file.unlink()
                except Exception:
                    pass
                return True, current_code, repair_logs

            except Exception as exc:
                tb = traceback.format_exc()
                log_msg = f"Attempt {attempt} FAILED: {exc}\n{tb}"
                repair_logs.append(log_msg)
                try:
                    if "attempt_file" in locals() and attempt_file.exists():
                        attempt_file.unlink()
                except Exception:
                    pass

                if attempt >= max_retries:
                    break

                # Self-Healing repair
                repair_logs.append(f"Initiating Self-Healing Repair (Attempt {attempt+1}/{max_retries})...")
                current_code = self._repair_code(current_code, str(exc), tb, blueprint, dataset)

        return False, current_code, repair_logs

    def _repair_code(
        self,
        broken_code: str,
        error_msg: str,
        full_traceback: str,
        blueprint: AgentBlueprintSpec,
        dataset: IngestedDataset | None,
    ) -> str:
        """Repairs buggy environment code via LLM or deterministic rules."""
        # 1. Check for simple deterministic bug fixes (e.g. division by zero)
        if "ZeroDivisionError" in error_msg or "division by zero" in error_msg.lower() or "/ 0" in broken_code:
            # Fix division by zero
            healed = re.sub(r"/\s*0(?!\d)", "/ 1.0", broken_code)
            healed = healed.replace("1 / 0", "1.0").replace("10 / 0", "10.0")
            if healed != broken_code:
                return healed

        if "shape mismatch" in error_msg.lower() or "does not match" in error_msg.lower():
            # If shape mismatch in obs or demos, let fallback synthesize clean code
            pass

        # 2. Try LLM repair if available
        if not isinstance(self.client, FallbackCodeSynthesizer):
            try:
                prompt = (
                    f"The following Gymnasium environment code failed verification checks:\n\n"
                    f"```python\n{broken_code}\n```\n\n"
                    f"Traceback Error:\n{full_traceback}\n\n"
                    f"Please repair the code. Ensure all engineering constraints are strictly met:\n"
                    f"- observation_space shape and return shape match\n"
                    f"- reset() and step() return float32 finite arrays\n"
                    f"- compute_reward() returns (float, dict)\n"
                    f"- get_demonstrations(32) returns (X_32_N, y_32)\n"
                    f"- interactive_step() returns valid dict\n\n"
                    f"Output the complete corrected Python code in ```python ... ```."
                )
                raw = self.client.generate(prompt, system_prompt="You are an Expert Python RL Debugger.")
                match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", raw)
                if match:
                    return match.group(1).strip()
            except Exception:
                pass

        # 3. No silent substitution: returning a *different* environment would hide the failure.
        return broken_code
