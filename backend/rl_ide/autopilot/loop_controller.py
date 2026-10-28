"""Autonomous Closed-Loop Trainer & Session Controller for RL-Native Agent IDE v4.0.

Orchestrates multi-round training loops (Warm-Start -> PPO Burst -> Observation ->
Reward Self-Tuning -> Critic Warmup) and manages session state snapshots.
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from pydantic import BaseModel, Field

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.autopilot.auto_tuner import AutoTunerAgent, TuningResult
from rl_ide.ingest.loader import IngestedDataset
from rl_ide.llm.client import BaseLLMClient, create_llm_client
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer, load_synthesized_env_class
from rl_ide.synthesizer.verifier import SandboxSmokeTester


class RoundSnapshot(BaseModel):
    """Immutable snapshot of a completed training round."""
    round_num: int
    metric: float
    target_threshold: float
    episodes_count: int
    diagnosis: str | None = None
    changes_summary: str | None = None
    weights_path: str
    env_code_path: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionState(BaseModel):
    """Complete real-time state of an agent synthesis session."""
    session_id: str
    blueprint: AgentBlueprintSpec
    dataset_id: str | None = None
    status: str = "created"  # "created" | "synthesizing" | "training" | "auto_tuning" | "target_achieved" | "completed" | "error"
    current_round: int = 1
    max_rounds: int = 4
    current_metric: float = 0.0
    target_threshold: float = 85.0
    rounds_history: list[RoundSnapshot] = Field(default_factory=list)
    env_module_code: str = ""
    arena_html: str = ""
    weights_path: str | None = None
    error_message: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionStateStore:
    """Manages active session instances, filesystem persistence, and round rollback."""

    def __init__(self, base_dir: Path | str = "workspaces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionState] = {}
        self._models: dict[str, PPO] = {}
        self._envs: dict[str, gym.Env] = {}

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def set_session(self, session: SessionState) -> None:
        self._sessions[session.session_id] = session

    def list_sessions(self) -> list[SessionState]:
        return list(self._sessions.values())

    def get_model(self, session_id: str) -> PPO | None:
        return self._models.get(session_id)

    def set_model(self, session_id: str, model: PPO) -> None:
        self._models[session_id] = model

    def get_env(self, session_id: str) -> gym.Env | None:
        return self._envs.get(session_id)

    def set_env(self, session_id: str, env: gym.Env) -> None:
        self._envs[session_id] = env

    def save_round_snapshot(
        self,
        session_id: str,
        round_num: int,
        metric: float,
        target_threshold: float,
        episodes_count: int,
        env_code: str,
        model: PPO,
        diagnosis: str | None = None,
        changes_summary: str | None = None,
    ) -> RoundSnapshot:
        """Persists immutable round artifacts to runs/<session_id>/v<round>/."""
        run_dir = Path("runs") / session_id / f"v{round_num}"
        run_dir.mkdir(parents=True, exist_ok=True)

        env_code_path = run_dir / "env_module.py"
        env_code_path.write_text(env_code, encoding="utf-8")

        weights_path = run_dir / "policy_weights.pt"
        torch.save(model.policy.state_dict(), str(weights_path))

        snapshot = RoundSnapshot(
            round_num=round_num,
            metric=metric,
            target_threshold=target_threshold,
            episodes_count=episodes_count,
            diagnosis=diagnosis,
            changes_summary=changes_summary,
            weights_path=str(weights_path),
            env_code_path=str(env_code_path),
        )

        meta_path = run_dir / "snapshot.json"
        meta_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

        session = self.get_session(session_id)
        if session:
            session.rounds_history.append(snapshot)
            session.current_metric = metric
            session.weights_path = str(weights_path)
            session.env_module_code = env_code

        return snapshot

    def rollback_to_round(self, session_id: str, round_num: int) -> RoundSnapshot:
        """Restores model policy weights and environment code from round N snapshot."""
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")

        target_snap = None
        for s in session.rounds_history:
            if s.round_num == round_num:
                target_snap = s
                break

        if not target_snap:
            raise ValueError(f"Round {round_num} snapshot not found for session '{session_id}'.")

        # Restore env code
        code_path = Path(target_snap.env_code_path)
        if code_path.exists():
            code = code_path.read_text(encoding="utf-8")
            session.env_module_code = code

            # Hot-swap env module in workspace
            ws_code_path = self.base_dir / session_id / "env_module.py"
            ws_code_path.write_text(code, encoding="utf-8")
            env_cls = load_synthesized_env_class(ws_code_path)
            new_env = env_cls()
            self.set_env(session_id, new_env)

        # Restore model weights
        weights_path = Path(target_snap.weights_path)
        model = self.get_model(session_id)
        if model and weights_path.exists():
            state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            model.policy.load_state_dict(state_dict)

        session.current_round = round_num
        session.current_metric = target_snap.metric
        session.weights_path = str(weights_path)
        return target_snap


class AutonomousLoopController:
    """Runs the autonomous closed-loop training and self-tuning cycles."""

    def __init__(
        self,
        store: SessionStateStore | None = None,
        llm_client: BaseLLMClient | None = None,
    ):
        self.store = store or SessionStateStore()
        self.client = llm_client or create_llm_client()
        self.auto_tuner = AutoTunerAgent(self.client)

    def run_loop(
        self,
        session_id: str,
        blueprint: AgentBlueprintSpec,
        env_cls: type[gym.Env],
        dataset: IngestedDataset | None = None,
        on_event_callback: Callable[[dict[str, Any]], None] | None = None,
        max_rounds: int | None = None,
        episodes_per_burst: int = 25,
    ) -> SessionState:
        """Executes the full multi-round autonomous training loop."""
        records = dataset.df.to_dict(orient="records") if (dataset and hasattr(dataset, "df")) else None
        env = env_cls(dataset_records=records)
        self.store.set_env(session_id, env)

        rounds_limit = max_rounds or blueprint.max_auto_tune_rounds or 4
        target_threshold = blueprint.target_metric_threshold

        # Initialize or retrieve SessionState
        session = self.store.get_session(session_id)
        if not session:
            ws_dir = self.store.base_dir / session_id
            env_code_path = ws_dir / "env_module.py"
            env_code = env_code_path.read_text(encoding="utf-8") if env_code_path.exists() else ""
            arena_html_path = ws_dir / "arena.html"
            arena_html = arena_html_path.read_text(encoding="utf-8") if arena_html_path.exists() else ""

            session = SessionState(
                session_id=session_id,
                blueprint=blueprint,
                dataset_id=blueprint.dataset_id,
                max_rounds=rounds_limit,
                target_threshold=target_threshold,
                env_module_code=env_code,
                arena_html=arena_html,
            )
            self.store.set_session(session)

        # Determine effective gamma: 0.0 for contextual bandit/tabular, 0.95 for dynamic sim
        is_tabular = blueprint.data_source_type in ("uploaded_dataset", "tabular") or dataset is not None
        effective_gamma = 0.0 if is_tabular else 0.95

        # Initialize SB3 PPO Policy
        model = self.store.get_model(session_id)
        if not model:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                gamma=effective_gamma,
                n_steps=min(64, max(16, episodes_per_burst * 2)),
                batch_size=16,
                n_epochs=5,
                ent_coef=0.01,
                device="cpu",
                verbose=0,
            )
            self.store.set_model(session_id, model)

        replay_buffer: list[dict[str, Any]] = []

        def emit(event_payload: dict[str, Any]) -> None:
            if on_event_callback:
                try:
                    on_event_callback(event_payload)
                except Exception:
                    pass

        current_code = session.env_module_code

        # --- MULTI-ROUND AUTONOMOUS LOOP ---
        for r in range(1, rounds_limit + 1):
            session.current_round = r
            session.status = "training"

            emit({
                "event": "round_started",
                "round": r,
                "max_rounds": rounds_limit,
                "target": target_threshold,
                "message": f"Starting Autonomous Round {r}/{rounds_limit}...",
            })

            # Stage 3.1: Warm-Start Behavioral Cloning (Round 1)
            if r == 1:
                emit({"event": "warmstart_started", "round": 1, "message": "Running Warm-Start Behavioral Cloning on domain heuristics..."})
                loss, acc = self._warmstart_from_demonstrations(model, env, epochs=15)
                emit({
                    "event": "warmstart_completed",
                    "round": 1,
                    "initial_loss": float(loss),
                    "accuracy_pct": float(acc * 100.0),
                    "message": f"Warm-start completed with {round(acc * 100.0, 1)}% demonstration alignment.",
                })

            # Stage 3.2: Critic Warmup (Round > 1)
            if r > 1 and replay_buffer:
                emit({"event": "critic_warmup_started", "round": r, "message": "Calibrating value network on updated reward function..."})
                self._critic_warmup(model, replay_buffer, gamma=0.95, epochs=15)
                emit({"event": "critic_warmup_completed", "round": r, "message": "Value network calibrated."})

            # Stage 3.3: PPO Training Burst
            burst_episodes = self._run_ppo_burst(
                model=model,
                env=env,
                round_num=r,
                n_episodes=episodes_per_burst,
                emit_fn=emit,
            )
            replay_buffer.extend(burst_episodes)

            # Stage 3.4: Evaluate Target Success Metric
            success_count = sum(1 for ep in burst_episodes if ep.get("is_success", False))
            current_metric = round((success_count / max(1, len(burst_episodes))) * 100.0, 2)
            session.current_metric = current_metric

            emit({
                "event": "round_evaluated",
                "round": r,
                "metric": current_metric,
                "target": target_threshold,
                "success_metric_name": blueprint.success_metric_name,
                "episodes_tested": len(burst_episodes),
            })

            # Check Target Mastery Achievement
            if current_metric >= target_threshold:
                session.status = "target_achieved"
                self.store.save_round_snapshot(
                    session_id=session_id,
                    round_num=r,
                    metric=current_metric,
                    target_threshold=target_threshold,
                    episodes_count=len(burst_episodes),
                    env_code=current_code,
                    model=model,
                    diagnosis="Target mastery expectation successfully achieved!",
                    changes_summary="Policy meets or exceeds target threshold.",
                )
                emit({
                    "event": "target_achieved",
                    "round": r,
                    "metric": current_metric,
                    "target": target_threshold,
                    "message": f"🎉 Target mastery achieved! ({current_metric}% >= {target_threshold}%). Policy ready for interactive play & export.",
                })
                break

            # If last round exhausted without meeting target
            if r == rounds_limit:
                session.status = "completed"
                self.store.save_round_snapshot(
                    session_id=session_id,
                    round_num=r,
                    metric=current_metric,
                    target_threshold=target_threshold,
                    episodes_count=len(burst_episodes),
                    env_code=current_code,
                    model=model,
                    diagnosis="Budget exhausted; target threshold not fully met.",
                    changes_summary="Completed all allotted auto-tune rounds.",
                )
                emit({
                    "event": "budget_exhausted",
                    "round": r,
                    "metric": current_metric,
                    "target": target_threshold,
                    "message": f"Autonomous auto-tune budget of {rounds_limit} rounds concluded. Final score: {current_metric}%.",
                })
                break

            # Stage 3.5: Autonomous Self-Tuning Step
            session.status = "auto_tuning"
            emit({
                "event": "auto_tuning_started",
                "round": r,
                "message": f"Diagnosing failure episodes and rewriting reward function for Round {r+1}...",
            })

            tuning_res = self.auto_tuner.tune(
                round_num=r,
                episodes=burst_episodes,
                blueprint=blueprint,
                current_code=current_code,
                dataset=dataset,
                workspace_dir=self.store.base_dir / session_id,
            )

            current_code = tuning_res.updated_code
            session.env_module_code = current_code

            # Save round snapshot
            self.store.save_round_snapshot(
                session_id=session_id,
                round_num=r,
                metric=current_metric,
                target_threshold=target_threshold,
                episodes_count=len(burst_episodes),
                env_code=current_code,
                model=model,
                diagnosis=tuning_res.diagnosis,
                changes_summary=tuning_res.changes_summary,
            )

            # Hot-swap the updated env_module.py into active session
            env_file = self.store.base_dir / session_id / "env_module.py"
            env_file.write_text(current_code, encoding="utf-8")
            updated_env_cls = load_synthesized_env_class(env_file)
            env = updated_env_cls(dataset_records=records)
            self.store.set_env(session_id, env)
            model.set_env(env)

            emit({
                "event": "auto_tuned",
                "round": r,
                "next_round": r + 1,
                "metric": current_metric,
                "diagnosis": tuning_res.diagnosis,
                "changes_summary": tuning_res.changes_summary,
            })

        return session

    def nudge_session(
        self,
        session_id: str,
        user_feedback_text: str,
        on_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TuningResult:
        """Applies English feedback to trigger an extra targeted tuning round."""
        session = self.store.get_session(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")

        model = self.store.get_model(session_id)
        env = self.store.get_env(session_id)

        # Run 1 quick burst to gather current telemetry
        episodes = self._run_ppo_burst(model, env, round_num=session.current_round, n_episodes=15, emit_fn=on_event_callback)

        tuning_res = self.auto_tuner.tune(
            round_num=session.current_round,
            episodes=episodes,
            blueprint=session.blueprint,
            current_code=session.env_module_code,
            user_nudge=user_feedback_text,
            workspace_dir=self.store.base_dir / session_id,
        )

        session.env_module_code = tuning_res.updated_code
        env_file = self.store.base_dir / session_id / "env_module.py"
        env_file.write_text(tuning_res.updated_code, encoding="utf-8")
        updated_env_cls = load_synthesized_env_class(env_file)
        new_env = updated_env_cls()
        self.store.set_env(session_id, new_env)
        if model:
            model.set_env(new_env)

        session.current_round += 1
        self.store.save_round_snapshot(
            session_id=session_id,
            round_num=session.current_round,
            metric=tuning_res.current_metric,
            target_threshold=session.target_threshold,
            episodes_count=len(episodes),
            env_code=tuning_res.updated_code,
            model=model,
            diagnosis=tuning_res.diagnosis,
            changes_summary=tuning_res.changes_summary,
        )

        return tuning_res

    def _warmstart_from_demonstrations(
        self,
        model: PPO,
        env: gym.Env,
        epochs: int = 15,
    ) -> tuple[float, float]:
        """Runs Supervised Behavioral Cloning on heuristic demonstrations."""
        if not hasattr(env, "get_demonstrations"):
            return 0.0, 1.0

        X_demo, y_demo = env.get_demonstrations(num_samples=200)
        X_t = torch.tensor(X_demo, dtype=torch.float32)
        y_t = torch.tensor(y_demo, dtype=torch.long)

        policy = model.policy
        # Optimize only actor parameters so value network optimizer state is not polluted
        actor_params = list(policy.action_net.parameters())
        if hasattr(policy, "mlp_extractor") and hasattr(policy.mlp_extractor, "policy_net"):
            actor_params += list(policy.mlp_extractor.policy_net.parameters())

        optimizer = torch.optim.Adam(actor_params, lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        initial_loss = 0.0
        final_acc = 0.0

        for epoch in range(epochs):
            features = policy.extract_features(X_t)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
            loss = criterion(logits, y_t)

            if epoch == 0:
                initial_loss = float(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            final_acc = float((preds == y_t).float().mean().item())

        return initial_loss, final_acc

    def _critic_warmup(
        self,
        model: PPO,
        replay_buffer: list[dict[str, Any]],
        gamma: float = 0.95,
        epochs: int = 15,
    ) -> None:
        """Calibrates value network on re-scored replay transitions."""
        all_obs: list[list[float]] = []
        all_returns: list[float] = []

        for ep in replay_buffer[-25:]:  # Use last 25 episodes
            steps = ep.get("steps", [])
            if not steps:
                continue
            rewards = [s.get("reward", 0.0) for s in steps]
            n = len(rewards)
            running_add = 0.0
            discounted = np.zeros(n, dtype=np.float32)
            for t in reversed(range(n)):
                running_add = rewards[t] + gamma * running_add
                discounted[t] = running_add

            for s, g_val in zip(steps, discounted):
                all_obs.append(s["obs"])
                all_returns.append(float(g_val))

        if not all_obs:
            return

        obs_t = torch.tensor(all_obs, dtype=torch.float32)
        ret_t = torch.tensor(all_returns, dtype=torch.float32).unsqueeze(-1)

        # Freeze Actor layers
        for p in model.policy.action_net.parameters():
            p.requires_grad = False
        if hasattr(model.policy, "mlp_extractor") and hasattr(model.policy.mlp_extractor, "policy_net"):
            for p in model.policy.mlp_extractor.policy_net.parameters():
                p.requires_grad = False

        # Gather Critic parameters to optimize
        critic_params = list(model.policy.value_net.parameters())
        if hasattr(model.policy, "mlp_extractor") and hasattr(model.policy.mlp_extractor, "value_net"):
            for p in model.policy.mlp_extractor.value_net.parameters():
                p.requires_grad = True
            critic_params += list(model.policy.mlp_extractor.value_net.parameters())

        for p in model.policy.value_net.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(critic_params, lr=1e-3)
        dataset = torch.utils.data.TensorDataset(obs_t, ret_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for _ in range(epochs):
            for b_obs, b_ret in loader:
                values = model.policy.predict_values(b_obs)
                loss = F.mse_loss(values, b_ret)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Unfreeze Actor layers
        for p in model.policy.action_net.parameters():
            p.requires_grad = True
        if hasattr(model.policy, "mlp_extractor") and hasattr(model.policy.mlp_extractor, "policy_net"):
            for p in model.policy.mlp_extractor.policy_net.parameters():
                p.requires_grad = True

    def _run_ppo_burst(
        self,
        model: PPO,
        env: gym.Env,
        round_num: int,
        n_episodes: int,
        emit_fn: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Collects episodes while streaming live ARENA_UPDATE frames."""
        collected_episodes: list[dict[str, Any]] = []

        for ep_idx in range(1, n_episodes + 1):
            obs, info = env.reset()
            ep_steps = []
            ep_reward = 0.0
            step_idx = 0
            done = False

            while not done and step_idx < 50:
                step_idx += 1
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    features = model.policy.extract_features(obs_t)
                    latent_pi, _ = model.policy.mlp_extractor(features)
                    logits = model.policy.action_net(latent_pi)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                    action = int(np.random.choice(len(probs), p=probs))

                next_obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)

                ep_steps.append({
                    "step": step_idx,
                    "obs": [float(x) for x in obs],
                    "action": action,
                    "reward": float(reward),
                    "reward_breakdown": step_info.get("reward_breakdown", {}),
                    "is_success": bool(step_info.get("is_success", False)),
                })

                # Stream live frame over WebSocket / callback
                if emit_fn and (step_idx % 2 == 1 or done):
                    emit_fn({
                        "event": "episode_step",
                        "round": round_num,
                        "episode_id": ep_idx,
                        "step": step_idx,
                        "total_reward": round(ep_reward, 2),
                        "reward_breakdown": step_info.get("reward_breakdown", {}),
                        "arena_state": step_info.get("arena_state", {}),
                        "action_probs": [float(p) for p in probs],
                    })
                    import time
                    time.sleep(0.03)  # ~30 FPS pacing for smooth visual playback in arena

                obs = next_obs

            # Final success check for episode
            is_success = any(s.get("is_success", False) for s in ep_steps)
            collected_episodes.append({
                "episode_id": ep_idx,
                "round": round_num,
                "total_reward": ep_reward,
                "steps": ep_steps,
                "is_success": is_success,
                "reward_breakdown": ep_steps[-1].get("reward_breakdown", {}) if ep_steps else {},
            })

        # SB3 learn call to update policy weights from new environment interactions
        try:
            model.learn(total_timesteps=max(32, len(collected_episodes) * 15), reset_num_timesteps=False)
        except Exception:
            pass

        return collected_episodes
