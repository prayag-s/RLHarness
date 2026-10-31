"""Autonomous Closed-Loop Trainer & Session Controller for RL-Native Agent IDE v4.0.

Orchestrates multi-round training loops (Warm-Start -> PPO Burst -> Observation ->
Reward Self-Tuning -> Critic Warmup) and manages session state snapshots.
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import json
import math
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import re
import shutil
import threading
from typing import Any, Callable
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from pydantic import BaseModel, Field, field_validator

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.autopilot.auto_tuner import AutoTunerAgent, TuningResult
from rl_ide.autopilot.crossfire_protocol import CrossfireProtocolStore
from rl_ide.autopilot.eval_corpus import EvalCorpusStore, greedy_predictor
from rl_ide.autopilot.verifier_gate import VerifierGate, VerifierSnapshot
from rl_ide.ingest.loader import IngestedDataset
from rl_ide.llm.client import BaseLLMClient, create_llm_client, get_active_llm_client
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer, load_synthesized_env_class, instantiate_env
from rl_ide.synthesizer.verifier import SandboxSmokeTester
from rl_ide.state.records_store import load_records, save_records


def parse_nudge_meta_command(text: str) -> dict[str, Any] | None:
    """Parses training loop control commands from user feedback prose.
    Examples:
      'tune 100 rounds, evaluate every 10 rounds'
      'train for 20 rounds'
      'run 50 episodes per round'
      'target 90%'
    """
    text_lower = text.lower()
    rounds_match = re.search(r'(?:tune|train|run|for)?\s*(\d+)\s*(?:more\s*)?rounds?', text_lower)
    episodes_match = re.search(r'(\d+)\s*episodes?', text_lower)
    # Only explicit mastery/goal phrasing may set the target; bare percentages are ignored.
    target_match = re.search(
        r'\b(?:target|goal|mastery|threshold|reach|achieve)\b\s*(?:metric|threshold|rate|goal|mastery)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%',
        text_lower,
    ) or re.search(
        r'\b(?:target|goal|mastery|threshold)\b\s*[:=]?\s*(\d+(?:\.\d+)?)\b',
        text_lower,
    )

    if rounds_match or episodes_match or target_match:
        return {
            "rounds": int(rounds_match.group(1)) if rounds_match else None,
            "episodes": int(episodes_match.group(1)) if episodes_match else None,
            "target": float(target_match.group(1)) if target_match else None,
        }
    return None


MAX_PPO_TIMESTEPS = 100_000


class TrainingInProgressError(RuntimeError):
    """Raised when a session already has an active training/tuning operation."""


def episode_step_limit(env: Any, default_ceiling: int = 200) -> int:
    """Returns the environment's declared horizon instead of a fixed 50-step cap."""
    horizon = getattr(env, "episode_horizon", None)
    if horizon is None:
        spec = getattr(env, "spec", None)
        horizon = getattr(spec, "max_episode_steps", None) if spec is not None else None
    try:
        horizon = int(horizon) if horizon is not None else default_ceiling
    except (TypeError, ValueError):
        horizon = default_ceiling
    return max(1, horizon)


def loop_new_terms(episodes: list[dict[str, Any]], baseline_keys: set[str]) -> list[str]:
    """Reward-term names that appear after an edit and never appeared before it."""
    seen: set[str] = set()
    for episode in episodes:
        for step in episode.get("steps", []) or []:
            seen.update((step.get("reward_breakdown") or {}).keys())
    return sorted(seen - set(baseline_keys))


def episode_metric_value(env: Any, steps: list[dict[str, Any]]) -> float:
    """Computes an honest per-episode metric.

    Classification tasks report mean per-step accuracy; control tasks report the terminal
    outcome. Step-level "at least one success" scoring is never used.
    """
    if not steps:
        return 0.0
    if getattr(env, "task_class", "control") == "classification":
        graded = [s for s in steps if s.get("is_correct") is not None]
        if not graded:
            return 0.0
        return 100.0 * sum(1 for s in graded if s["is_correct"]) / len(graded)
    return 100.0 if bool(steps[-1].get("is_success")) else 0.0


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
    episodes: list[dict[str, Any]] = Field(default_factory=list)
    env_module_code: str = ""
    arena_html: str = ""
    weights_path: str | None = None
    latest_model_zip_path: str | None = None
    scorecard: dict[str, Any] | None = None
    eval_corpus: dict[str, Any] | None = None
    crossfire: dict[str, Any] | None = None
    critique_gate: dict[str, Any] | None = None
    best_val_metric: float | None = None
    best_val_round: int | None = None
    mutation_pending: str | None = None
    error_message: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("env_module_code", "arena_html", mode="before")
    @classmethod
    def _coerce_none_to_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class SessionStateStore:
    """Manages active session instances, filesystem persistence, and round rollback."""

    def __init__(self, base_dir: Path | str = "workspaces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionState] = {}
        self._models: dict[str, PPO] = {}
        self._envs: dict[str, gym.Env] = {}

    def session_state_path(self, session_id: str) -> Path:
        return self.base_dir / session_id / "session_state.json"

    def save_session_state(self, session: SessionState) -> None:
        """Persists round history so a restarted (or per-round sandboxed) worker resumes.

        The episode buffer is truncated: it is a UI history, not resumable training state, and
        writing every step of every round would bloat the workspace on long runs.
        """
        path = self.session_state_path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = session.model_dump()
        payload["episodes"] = payload.get("episodes", [])[-40:]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_session_state(self, session_id: str) -> SessionState | None:
        path = self.session_state_path(session_id)
        if not path.exists():
            return None
        try:
            return SessionState.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def set_session(self, session: SessionState | None) -> None:
        if session is not None:
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

    def delete_model(self, session_id: str) -> None:
        self._models.pop(session_id, None)

    def delete_session(self, session_id: str) -> None:
        """Drops every in-memory reference for a deleted conversation."""
        self._sessions.pop(session_id, None)
        self._models.pop(session_id, None)
        self._envs.pop(session_id, None)

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

        model_zip_path = Path("runs") / session_id / "model.zip"
        try:
            model.save(str(model_zip_path))
        except Exception:
            pass

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
            if model_zip_path.exists():
                session.latest_model_zip_path = str(model_zip_path)
            session.env_module_code = env_code
            self.save_session_state(session)

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

        # Restore env code (only when the snapshot actually contains a module)
        code_path = Path(target_snap.env_code_path)
        if code_path.exists() and code_path.read_text(encoding="utf-8").strip():
            code = code_path.read_text(encoding="utf-8")
            session.env_module_code = code

            # Hot-swap env module in workspace
            ws_code_path = self.base_dir / session_id / "env_module.py"
            ws_code_path.write_text(code, encoding="utf-8")
            env_cls = load_synthesized_env_class(ws_code_path)
            new_env = instantiate_env(env_cls, dataset_records=load_records(self.base_dir / session_id))
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
        self._llm_client = llm_client
        self._auto_tuner = AutoTunerAgent(llm_client) if llm_client else None
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.corpus = EvalCorpusStore(self.store.base_dir)
        self.protocol = CrossfireProtocolStore(self.store.base_dir)
        self.gate = VerifierGate()
        self._best_outcomes: dict[str, list[float]] = {}

    def session_lock(self, session_id: str) -> threading.Lock:
        """Per-session lock guarding training, tuning and live inference against races."""
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            return lock

    def is_busy(self, session_id: str) -> bool:
        lock = self._locks.get(session_id)
        return bool(lock and lock.locked())

    @property
    def client(self) -> BaseLLMClient:
        if self._llm_client:
            return self._llm_client
        return get_active_llm_client()

    @property
    def auto_tuner(self) -> AutoTunerAgent:
        if self._auto_tuner:
            return self._auto_tuner
        return AutoTunerAgent(self.client)

    def run_loop(
        self,
        session_id: str,
        blueprint: AgentBlueprintSpec,
        env_cls: type[gym.Env],
        dataset: IngestedDataset | None = None,
        on_event_callback: Callable[[dict[str, Any]], None] | None = None,
        max_rounds: int | None = None,
        episodes_per_burst: int = 25,
        additional_rounds: int | None = None,
        skip_auto_tune: bool = False,
        finalize_scorecard: bool = True,
    ) -> SessionState:
        """Executes rounds under a per-session lock (see `_run_loop_locked`)."""
        lock = self.session_lock(session_id)
        if not lock.acquire(blocking=False):
            raise TrainingInProgressError(
                f"Session '{session_id}' already has an active training or tuning operation."
            )
        try:
            return self._run_loop_locked(
                session_id=session_id,
                blueprint=blueprint,
                env_cls=env_cls,
                dataset=dataset,
                on_event_callback=on_event_callback,
                max_rounds=max_rounds,
                episodes_per_burst=episodes_per_burst,
                additional_rounds=additional_rounds,
                skip_auto_tune=skip_auto_tune,
                finalize_scorecard=finalize_scorecard,
            )
        finally:
            lock.release()

    def _run_loop_locked(
        self,
        session_id: str,
        blueprint: AgentBlueprintSpec,
        env_cls: type[gym.Env],
        dataset: IngestedDataset | None = None,
        on_event_callback: Callable[[dict[str, Any]], None] | None = None,
        max_rounds: int | None = None,
        episodes_per_burst: int = 25,
        additional_rounds: int | None = None,
        skip_auto_tune: bool = False,
        finalize_scorecard: bool = True,
    ) -> SessionState:
        """Executes the full multi-round autonomous training loop, supporting continuous resumption."""
        records = dataset.df.to_dict(orient="records") if (dataset and hasattr(dataset, "df")) else None
        if records is None:
            records = load_records(self.store.base_dir / session_id)
        if records is not None:
            save_records(self.store.base_dir / session_id, records)
        env = instantiate_env(env_cls, dataset_records=records)
        self.store.set_env(session_id, env)

        # Initialize or retrieve SessionState (in-memory, else the durable copy on disk)
        session = self.store.get_session(session_id)
        if not session:
            session = self.store.load_session_state(session_id)
            if session is not None:
                self.store.set_session(session)
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
                max_rounds=max_rounds or blueprint.max_auto_tune_rounds or 4,
                target_threshold=blueprint.target_metric_threshold,
                env_module_code=env_code,
                arena_html=arena_html,
            )
            self.store.set_session(session)

        # Round semantics (unambiguous): run exactly N rounds in this call, accumulating on top
        # of any previously completed rounds. `session.max_rounds` is the cumulative target.
        completed_rounds = len(session.rounds_history)
        has_prior_rounds = completed_rounds > 0
        start_round = completed_rounds + 1

        if additional_rounds is not None:
            rounds_to_run = int(additional_rounds)
        elif max_rounds is not None:
            # `max_rounds` means "rounds to run in this call"; the session target accumulates.
            rounds_to_run = int(max_rounds)
        else:
            rounds_to_run = int(blueprint.max_auto_tune_rounds or 4)
        rounds_to_run = max(1, rounds_to_run)
        rounds_limit = completed_rounds + rounds_to_run

        session.max_rounds = rounds_limit
        target_threshold = blueprint.target_metric_threshold if blueprint.target_metric_threshold is not None else (session.target_threshold or 85.0)
        session.target_threshold = target_threshold

        # Determine effective gamma: 0.0 for contextual bandit/tabular, 0.95 for dynamic sim
        is_tabular = blueprint.data_source_type in ("uploaded_dataset", "tabular") or dataset is not None
        effective_gamma = 0.0 if is_tabular else 0.95

        # Initialize or reload SB3 PPO Policy
        model = self.store.get_model(session_id)
        if not model:
            # Check latest_model_zip_path or runs/<session_id>/model.zip
            model_zip_candidate = (
                Path(session.latest_model_zip_path)
                if session.latest_model_zip_path
                else (Path("runs") / session_id / "model.zip")
            )
            if model_zip_candidate and model_zip_candidate.exists():
                try:
                    model = PPO.load(str(model_zip_candidate), env=env, device="cpu")
                    self.store.set_model(session_id, model)
                except Exception as load_err:
                    print(f"Warning: Failed to load PPO from {model_zip_candidate}: {load_err}")

        # Check weights path if model still not loaded
        if not model and session.weights_path and Path(session.weights_path).exists():
            try:
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=1e-3,
                    gamma=effective_gamma,
                    n_steps=256,
                    batch_size=64,
                    n_epochs=10,
                    ent_coef=0.02,
                    device="cpu",
                    verbose=0,
                )
                state_dict = torch.load(session.weights_path, map_location="cpu", weights_only=True)
                model.policy.load_state_dict(state_dict)
                self.store.set_model(session_id, model)
            except Exception as w_err:
                print(f"Warning: Failed to load weights from {session.weights_path}: {w_err}")

        # Verify observation and action space compatibility
        if model:
            obs_match = getattr(model, "observation_space", None) == env.observation_space
            act_match = getattr(model, "action_space", None) == env.action_space
            if obs_match and act_match:
                model.set_env(env)
            else:
                # Spaces changed: fresh policy required
                model = None

        if not model:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                gamma=effective_gamma,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                ent_coef=0.02,
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

        # The corpus and the frozen rubric exist before the first training step; a claim scored
        # against a rubric written afterwards is not evidence.
        try:
            if not self.corpus.load(session_id):
                seeded = self.corpus.seed_from_validation_split(session_id, env)
                if seeded:
                    emit({
                        "event": "eval_corpus_seeded",
                        "round": session.current_round,
                        "cases": len(seeded),
                        "message": f"Seeded {len(seeded)} held-out behavioural cases into the eval corpus.",
                    })
            protocol = self.protocol.load(session_id)
            if protocol is None:
                protocol = self.protocol.open(
                    session_id,
                    rubric=self.DEFAULT_RUBRIC,
                    task_title=getattr(blueprint, "task_title", session_id),
                    task_class=str(getattr(env, "task_class", "control")),
                )
            self.protocol.record_training_start(session_id)
            val_seed_ids = [c.case_id for c in self.corpus.load(session_id) if c.source == "val_seed"]
            if val_seed_ids:
                self.protocol.register_holdout(session_id, val_seed_ids)
            report = self.corpus.evaluate_policy(session_id, greedy_predictor(model))
            session.eval_corpus = report.model_dump()
            session.crossfire = self.protocol.closure_report(session_id)
        except Exception as corpus_err:
            emit({
                "event": "eval_corpus_unavailable",
                "round": session.current_round,
                "message": f"Eval corpus unavailable: {type(corpus_err).__name__}: {corpus_err}",
            })

        best_val_metric: float | None = session.best_val_metric
        best_val_round: int | None = session.best_val_round
        best_corpus_passing: set[str] = set()
        best_entropy: float | None = None
        term_keys: set[str] | None = None
        mutation_kind: str | None = session.mutation_pending

        # --- MULTI-ROUND AUTONOMOUS LOOP ---
        for r in range(start_round, rounds_limit + 1):
            session.current_round = r
            session.status = "training"

            emit({
                "event": "round_started",
                "round": r,
                "max_rounds": rounds_limit,
                "target": target_threshold,
                "message": f"Starting Autonomous Round {r}/{rounds_limit}...",
            })

            # Stage 3.1: Warm-Start Behavioral Cloning (Round 1 ONLY, never when resuming past rounds)
            if r == 1 and not has_prior_rounds and not getattr(env, "is_placeholder", False):
                emit({"event": "warmstart_started", "round": 1, "message": "Running Warm-Start Behavioral Cloning on domain heuristics..."})
                loss, acc = self._warmstart_from_demonstrations(model, env, epochs=15, lr=3e-3, num_samples=150)
                emit({
                    "event": "warmstart_completed",
                    "round": 1,
                    "initial_loss": float(loss),
                    "accuracy_pct": float(acc * 100.0),
                    "message": f"Warm-start completed with {round(acc * 100.0, 1)}% demonstration alignment.",
                })
                # Critic Pre-Calibration pass
                emit({"event": "critic_warmup_started", "round": 1, "message": "Pre-calibrating Critic value network on warm-start baseline..."})
                self._critic_pre_calibration(model, env, gamma=effective_gamma, epochs=5)
                emit({"event": "critic_warmup_completed", "round": 1, "message": "Critic value network pre-calibrated."})

            # Stage 3.2: Critic Warmup (Round > 1)
            if r > 1 and replay_buffer:
                emit({"event": "critic_warmup_started", "round": r, "message": "Calibrating value network on updated reward function..."})
                self._critic_warmup(model, replay_buffer, gamma=effective_gamma, epochs=8)
                emit({"event": "critic_warmup_completed", "round": r, "message": "Value network calibrated."})

            # Stage 3.3: PPO Training Burst — the configured episode budget maps to timesteps with
            # an explicit, observable ceiling instead of a silent 1024-step cap.
            train_timesteps = max(256, int(episodes_per_burst) * 25)
            train_timesteps = min(train_timesteps, MAX_PPO_TIMESTEPS)
            try:
                model.learn(total_timesteps=train_timesteps, reset_num_timesteps=False)
            except Exception:
                pass

            # Collect representative rollouts for auto-tuning failure traces & throttled UI streaming
            burst_episodes = self._run_ppo_burst(
                model=model,
                env=env,
                round_num=r,
                n_episodes=min(20, episodes_per_burst),
                emit_fn=emit,
            )
            replay_buffer.extend(burst_episodes)

            # Stage 3.4: Evaluate Target Success Metric Strictly from Deterministic Rollouts.
            # Rounds are scored on the selection split (val, else train) so that the held-out
            # test split is only ever used for the final scorecard.
            eval_metric, eval_episodes, illegal_pct = self.evaluate_policy_deterministic(
                env=env,
                model=model,
                n_eval_episodes=50,
                emit_fn=emit,
                round_num=r,
                split=self._selection_split(env),
            )
            current_metric = eval_metric
            session.current_metric = current_metric
            session.episodes.extend(eval_episodes)

            emit({
                "event": "round_evaluated",
                "round": r,
                "metric": current_metric,
                "target": target_threshold,
                "success_metric_name": blueprint.success_metric_name,
                "episodes_tested": len(eval_episodes),
                "illegal_moves_pct": illegal_pct,
            })

            # ---- Independent verifier gate (Crossfire P3) -------------------
            val_outcomes = [float(ep.get("metric_value", 0.0)) for ep in eval_episodes]
            entropy_now = self._action_entropy(eval_episodes)
            term_stats = self._term_stats(eval_episodes, term_keys)
            corpus_passing: set[str] = set()
            corpus_report = None
            try:
                corpus_report = self.corpus.evaluate_policy(
                    session_id, greedy_predictor(model), baseline_passing_ids=best_corpus_passing
                )
                corpus_passing = self.corpus.passing_case_ids(session_id, corpus_report)
                session.eval_corpus = corpus_report.model_dump()
                emit({
                    "event": "eval_corpus_evaluated",
                    "round": r,
                    "pass_rate_pct": corpus_report.pass_rate_pct,
                    "passed": corpus_report.passed_cases,
                    "scored": corpus_report.scored_cases,
                    "regressed": corpus_report.regressed_case_ids,
                    "message": (
                        f"Golden corpus: {corpus_report.passed_cases}/{corpus_report.scored_cases} "
                        f"verified ({corpus_report.pass_rate_pct}%)."
                    ),
                })
            except Exception as corpus_err:
                emit({
                    "event": "eval_corpus_unavailable",
                    "round": r,
                    "message": f"Corpus evaluation failed: {type(corpus_err).__name__}: {corpus_err}",
                })

            if best_val_metric is None:
                best_val_metric, best_val_round = current_metric, r
                best_corpus_passing = corpus_passing
                best_entropy = entropy_now
                self._record_best(session, model, current_code, current_metric, val_outcomes)
                emit({
                    "event": "checkpoint_recorded",
                    "round": r,
                    "metric": current_metric,
                    "message": f"Baseline checkpoint recorded (round {r}, {current_metric}%).",
                })
            elif mutation_kind:
                snapshot = VerifierSnapshot(
                    mutation_kind=mutation_kind,
                    session_id=session_id,
                    env_code=current_code,
                    model_path=str(self._best_paths(session_id)[0]),
                    val_metric=best_val_metric,
                    val_outcomes=self._best_outcomes.get(session_id, []),
                    corpus_passing_ids=sorted(best_corpus_passing),
                    entropy=best_entropy,
                )
                decision = self.gate.decide(
                    snapshot,
                    val_metric_after=current_metric,
                    val_outcomes_after=val_outcomes,
                    corpus_passing_ids_after=corpus_passing,
                    corpus_pass_rate_after=(
                        corpus_report.pass_rate_pct if corpus_report else None
                    ),
                    term_stats=term_stats,
                    entropy_after=entropy_now,
                )
                session.critique_gate = decision.model_dump()
                mutation_kind = None
                session.mutation_pending = None
                if not decision.passed:
                    restored_model, restored_code = self._restore_best(session, env)
                    if restored_model is not None:
                        model = restored_model
                        if restored_code:
                            current_code = restored_code
                            try:
                                restored_env = instantiate_env(
                                    load_synthesized_env_class(
                                        self.store.base_dir / session_id / "env_module.py"
                                    ),
                                    dataset_records=load_records(self.store.base_dir / session_id),
                                )
                                env = restored_env
                                model.set_env(env)
                                self.store.set_env(session_id, env)
                            except Exception:
                                pass
                    current_metric = best_val_metric
                    session.current_metric = current_metric
                    emit({
                        "event": "mutation_rolled_back",
                        "round": r,
                        "metric": current_metric,
                        "triage": decision.triage.category if decision.triage else "unclassified",
                        "advice": decision.triage.advice if decision.triage else "",
                        "reason": decision.reason,
                        "message": (
                            f"Change rolled back: {decision.reason} "
                            f"({decision.triage.category if decision.triage else 'unclassified'})."
                        ),
                    })
                    continue
                emit({
                    "event": "mutation_retained",
                    "round": r,
                    "metric": current_metric,
                    "reason": decision.reason,
                    "message": f"Change retained: {decision.reason}",
                })

            if best_val_metric is not None and current_metric >= best_val_metric:
                best_val_metric, best_val_round = current_metric, r
                best_corpus_passing = corpus_passing
                best_entropy = entropy_now
                self._record_best(session, model, current_code, current_metric, val_outcomes)

            term_keys = self._term_keys(eval_episodes)

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

            # Plateau Detection: Only trigger AutoTunerAgent if metric fails to improve by +3.0%
            # over 2 consecutive rounds, or if current_metric < 25.0% after Round 2.
            prev_metric = session.rounds_history[-1].metric if session.rounds_history else 0.0
            metric_delta = current_metric - prev_metric

            should_tune = False
            if r >= 2 and current_metric < 25.0:
                should_tune = True
            elif r >= 3 and len(session.rounds_history) >= 2:
                delta_curr = current_metric - session.rounds_history[-1].metric
                delta_prev = session.rounds_history[-1].metric - session.rounds_history[-2].metric
                if delta_curr < 3.0 and delta_prev < 3.0:
                    should_tune = True

            if not should_tune:
                # Policy is progressing! Retain reward function and continue training
                session.status = "training"
                self.store.save_round_snapshot(
                    session_id=session_id,
                    round_num=r,
                    metric=current_metric,
                    target_threshold=target_threshold,
                    episodes_count=len(burst_episodes),
                    env_code=current_code,
                    model=model,
                    diagnosis=f"Round {r} progress: {current_metric}% (Delta: +{round(metric_delta, 1)}%). Policy progressing; reward function retained.",
                    changes_summary="Reward function retained: agent continues active improvement trajectory.",
                )
                emit({
                    "event": "training_continued",
                    "round": r,
                    "next_round": r + 1,
                    "metric": current_metric,
                    "delta": round(metric_delta, 2),
                    "message": f"Policy progressing (+{round(metric_delta, 1)}% delta). Retaining reward function for Round {r+1}...",
                })
                continue

            # Stage 3.5: Autonomous Self-Tuning Step (Plateau detected)
            session.status = "auto_tuning"
            emit({
                "event": "auto_tuning_started",
                "round": r,
                "message": f"Plateau detected at {current_metric}%. Diagnosing failure episodes and rewriting reward function for Round {r+1}...",
            })

            if skip_auto_tune:
                # The host owns the LLM between sandboxed rounds (it has the network and the
                # metered credentials); this worker only trains and reports evidence.
                session.status = "training"
                emit({
                    "event": "auto_tune_deferred",
                    "round": r,
                    "metric": current_metric,
                    "message": "Plateau detected; the host orchestrator will tune between rounds.",
                })
                continue

            if os.environ.get("RL_IDE_SANDBOX") == "1":
                # The worker has neither network nor provider keys, so the LLM tuner cannot run
                # there; skipping is honest and keeps the round alive.
                session.status = "training"
                emit({
                    "event": "auto_tune_skipped",
                    "round": r,
                    "metric": current_metric,
                    "message": "LLM auto-tuning is disabled inside the sandbox (no network); reward function retained.",
                })
                continue

            tuning_res = self.auto_tuner.tune(
                round_num=r,
                episodes=burst_episodes,
                blueprint=blueprint,
                current_code=current_code,
                dataset=dataset,
                workspace_dir=self.store.base_dir / session_id,
            )

            if not tuning_res.verified:
                # Unverified generated code must never be persisted or loaded.
                emit({
                    "event": "auto_tune_failed",
                    "round": r,
                    "metric": current_metric,
                    "message": "Auto-tuner output failed sandbox verification; reward function retained unchanged.",
                })
                self.store.save_round_snapshot(
                    session_id=session_id,
                    round_num=r,
                    metric=current_metric,
                    target_threshold=target_threshold,
                    episodes_count=len(burst_episodes),
                    env_code=current_code,
                    model=model,
                    diagnosis=tuning_res.diagnosis,
                    changes_summary="Auto-tune rejected: unverified code. Reward function retained.",
                )
                continue

            current_code = tuning_res.updated_code
            session.env_module_code = current_code
            mutation_kind = "reward"

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
            env = instantiate_env(updated_env_cls, dataset_records=records)
            self.store.set_env(session_id, env)

            if (
                getattr(model, "observation_space", None) == env.observation_space
                and getattr(model, "action_space", None) == env.action_space
            ):
                model.set_env(env)
            else:
                # Spaces changed: deliberately start a fresh policy rather than reuse mismatched weights.
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=1e-3,
                    gamma=effective_gamma,
                    n_steps=256,
                    batch_size=64,
                    n_epochs=10,
                    ent_coef=0.02,
                    device="cpu",
                    verbose=0,
                )
                self.store.set_model(session_id, model)
                emit({
                    "event": "policy_reset",
                    "round": r,
                    "message": "Action/observation space changed; a fresh policy was initialized.",
                })

            # Apply the tuner's hyperparameter recommendations (previously computed but unused).
            diffs = tuning_res.hyperparameter_diffs or {}
            if "ent_coef" in diffs:
                model.ent_coef = float(diffs["ent_coef"])
            if "learning_rate" in diffs:
                lr = float(diffs["learning_rate"])
                model.learning_rate = lr
                try:
                    for group in model.policy.optimizer.param_groups:
                        group["lr"] = lr
                except Exception:
                    pass

            emit({
                "event": "auto_tuned",
                "round": r,
                "next_round": r + 1,
                "metric": current_metric,
                "diagnosis": tuning_res.diagnosis,
                "changes_summary": tuning_res.changes_summary,
            })

        self.store.save_session_state(session)

        # Final held-out scorecard: the only number that describes the shipped policy.
        if not finalize_scorecard:
            return session
        try:
            self._finalize_scorecard(session=session, model=model, env=env, blueprint=blueprint, emit=emit)
        except Exception as scorecard_err:
            print(f"Warning: scorecard finalization failed for {session_id}: {scorecard_err}")
            emit({
                "event": "scorecard_failed",
                "round": session.current_round,
                "message": f"Scorecard unavailable: {scorecard_err}",
            })

        return session

    def nudge_session(
        self,
        session_id: str,
        user_feedback_text: str,
        on_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TuningResult:
        """Applies English feedback or executes training control meta-commands."""
        session = self.store.get_session(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")

        lock = self.session_lock(session_id)
        if not lock.acquire(blocking=False):
            raise TrainingInProgressError(
                f"Session '{session_id}' already has an active training or tuning operation."
            )
        try:
            return self._nudge_session_locked(
                session_id=session_id,
                user_feedback_text=user_feedback_text,
                on_event_callback=on_event_callback,
                session=session,
            )
        finally:
            lock.release()

    def _nudge_session_locked(
        self,
        session_id: str,
        user_feedback_text: str,
        on_event_callback: Callable[[dict[str, Any]], None] | None,
        session: SessionState,
    ) -> TuningResult:
        # Check for training control meta-commands (e.g. 'tune 100 rounds, evaluate every 10 rounds')
        meta = parse_nudge_meta_command(user_feedback_text)
        if meta and (meta["rounds"] is not None or meta["episodes"] is not None or meta["target"] is not None):
            extra_rounds = meta["rounds"] or 5
            new_max_rounds = session.current_round + extra_rounds
            session.max_rounds = new_max_rounds
            if meta["target"] is not None:
                session.target_threshold = meta["target"]
            eps_burst = meta["episodes"] or 25

            env = self.store.get_env(session_id)
            if env is None:
                ws_code_path = self.store.base_dir / session_id / "env_module.py"
                env_cls = load_synthesized_env_class(ws_code_path)
                env = instantiate_env(env_cls, dataset_records=load_records(self.store.base_dir / session_id))
                self.store.set_env(session_id, env)
            else:
                env_cls = type(env)

            updated_session = self._run_loop_locked(
                session_id=session_id,
                blueprint=session.blueprint,
                env_cls=env_cls,
                on_event_callback=on_event_callback,
                max_rounds=None,
                additional_rounds=extra_rounds,
                episodes_per_burst=eps_burst,
            )

            return TuningResult(
                round=updated_session.current_round,
                current_metric=updated_session.current_metric,
                target_threshold=updated_session.target_threshold,
                diagnosis=f"User control command executed: \"{user_feedback_text}\". Trained through Round {updated_session.current_round}.",
                changes_summary=f"Extended loop to {new_max_rounds} rounds. Metric achieved: {updated_session.current_metric}%.",
                updated_code=updated_session.env_module_code,
                verified=True,
            )

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

        if tuning_res.verified and tuning_res.applied and tuning_res.updated_code.strip():
            session.env_module_code = tuning_res.updated_code
            # The change is judged by the next training round; if it regresses the held-out metric
            # or the corpus, the loop restores the last approved checkpoint and reports why.
            session.mutation_pending = "reward"
            env_file = self.store.base_dir / session_id / "env_module.py"
            env_file.write_text(tuning_res.updated_code, encoding="utf-8")
            updated_env_cls = load_synthesized_env_class(env_file)
            new_env = instantiate_env(updated_env_cls, dataset_records=load_records(self.store.base_dir / session_id))
            self.store.set_env(session_id, new_env)
            if model and getattr(model, "observation_space", None) == new_env.observation_space and getattr(model, "action_space", None) == new_env.action_space:
                model.set_env(new_env)
        else:
            tuning_res.diagnosis += " (rejected: unverified code)"

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
        epochs: int = 40,
        lr: float = 3e-3,
        num_samples: int = 500,
    ) -> tuple[float, float]:
        """Runs Supervised Behavioral Cloning on heuristic demonstrations."""
        if not hasattr(env, "get_demonstrations"):
            return 0.0, 1.0

        X_demo, y_demo = env.get_demonstrations(num_samples=num_samples)
        if len(X_demo) == 0 or len(y_demo) == 0:
            return 0.0, 0.0
        X_t = torch.tensor(X_demo, dtype=torch.float32)
        y_t = torch.tensor(y_demo, dtype=torch.long)

        policy = model.policy
        # Optimize only actor parameters so value network optimizer state is not polluted
        actor_params = list(policy.action_net.parameters())
        if hasattr(policy, "mlp_extractor") and hasattr(policy.mlp_extractor, "policy_net"):
            actor_params += list(policy.mlp_extractor.policy_net.parameters())

        optimizer = torch.optim.Adam(actor_params, lr=lr)
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

    def _critic_pre_calibration(
        self,
        model: PPO,
        env: gym.Env,
        gamma: float = 0.95,
        epochs: int = 10,
        n_episodes: int = 30,
    ) -> None:
        """Calibrates Critic Value Network on rollout returns from warm-started policy."""
        transitions: list[dict[str, Any]] = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_steps = []
            done = False
            step_k = 0
            while not done and step_k < 50:
                step_k += 1
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    features = model.policy.extract_features(obs_t)
                    latent_pi, _ = model.policy.mlp_extractor(features)
                    logits = model.policy.action_net(latent_pi)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                    if hasattr(env, "get_valid_actions"):
                        valid = env.get_valid_actions()
                        if valid:
                            masked = np.zeros_like(probs)
                            for v in valid:
                                if v < len(probs):
                                    masked[v] = probs[v]
                            if masked.sum() > 0:
                                masked /= masked.sum()
                                act = int(np.argmax(masked))
                            else:
                                act = int(valid[0])
                        else:
                            act = 0
                    else:
                        act = int(np.argmax(probs))
                next_obs, reward, term, trunc, _ = env.step(act)
                ep_steps.append({"obs": [float(x) for x in obs], "reward": float(reward)})
                done = term or trunc
                obs = next_obs
            if ep_steps:
                transitions.append({"steps": ep_steps})

        if transitions:
            self._critic_warmup(model, transitions, gamma=gamma, epochs=epochs)

    def calibrate_critic(
        self,
        session_id: str,
        env: gym.Env | None = None,
        epochs: int = 15,
        n_episodes: int = 25,
    ) -> bool:
        """Calibrates Critic value head on existing policy weights without resetting Actor."""
        model = self.store.get_model(session_id)
        active_env = env or self.store.get_env(session_id)
        if not model or not active_env:
            return False
        session = self.store.get_session(session_id)
        effective_gamma = 0.0 if (session and session.blueprint.data_source_type in ("uploaded_dataset", "tabular")) else 0.95
        self._critic_pre_calibration(model, active_env, gamma=effective_gamma, epochs=epochs, n_episodes=n_episodes)
        return True

    def evaluate_policy_deterministic(
        self,
        env: gym.Env,
        model: PPO,
        n_eval_episodes: int = 50,
        emit_fn: Callable[[dict[str, Any]], None] | None = None,
        round_num: int = 1,
        split: str | None = None,
        phase: str = "eval",
    ) -> tuple[float, list[dict[str, Any]], float]:
        """Scores the policy exactly as it ships: unmasked, on the requested data split.

        Action masks are deliberately not applied. ``get_critical_actions()`` is a tactical
        oracle, and legality masks are only meaningful if training and the exported ONNX policy
        apply the identical mask — neither does. Illegal moves are counted and reported instead of
        being hidden by a mask.

        Returns: (metric_pct, eval_episodes, illegal_rate_pct)
        """
        eval_episodes: list[dict[str, Any]] = []
        success_count = 0
        illegal_count = 0

        profiles = ["random", "tactical", "optimal"]
        for ep_idx in range(1, n_eval_episodes + 1):
            profile = profiles[(ep_idx - 1) % len(profiles)]
            agent_first = (ep_idx % 2 == 1)
            reset_opts = {"opponent_profile": profile, "agent_first": agent_first}
            if split:
                reset_opts["split"] = split
            try:
                obs, info = env.reset(options=reset_opts)
            except (TypeError, ValueError):
                try:
                    obs, info = env.reset(options={"split": split}) if split else env.reset()
                except (TypeError, ValueError):
                    obs, info = env.reset()
            effective_split = str((info or {}).get("split") or split or "all")

            ep_steps = []
            ep_reward = 0.0
            step_idx = 0
            done = False
            ep_had_illegal = False
            step_limit = episode_step_limit(env)

            while not done and step_idx < step_limit:
                step_idx += 1
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    features = model.policy.extract_features(obs_t)
                    latent_pi, _ = model.policy.mlp_extractor(features)
                    logits = model.policy.action_net(latent_pi)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                # No action masking: the policy is scored exactly as it is exported.
                action = int(np.argmax(probs))
                action_probs_to_emit = probs.tolist()

                next_obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)

                if step_info.get("illegal_move_attempted") or step_info.get("is_illegal"):
                    ep_had_illegal = True

                ep_steps.append({
                    "step": step_idx,
                    "obs": [float(x) for x in obs],
                    "action": action,
                    "reward": float(reward),
                    "reward_breakdown": step_info.get("reward_breakdown", {}),
                    "is_success": bool(step_info.get("is_success", False)),
                    "is_correct": step_info.get("is_correct"),
                })

                ep_id = f"r{round_num}_{phase}_{effective_split}_{ep_idx}"
                # Stream top deterministic evaluation episodes to the UI Arena
                if emit_fn and ep_idx <= 5:
                    emit_fn({
                        "event": "episode_step",
                        "round": round_num,
                        "episode_id": ep_id,
                        "step": step_idx,
                        "total_reward": round(ep_reward, 2),
                        "reward_breakdown": step_info.get("reward_breakdown", {}),
                        "arena_state": step_info.get("arena_state", {}),
                        "action_probs": [float(p) for p in action_probs_to_emit],
                    })

                obs = next_obs

            if ep_had_illegal:
                illegal_count += 1

            episode_metric = episode_metric_value(env, ep_steps)
            if episode_metric >= 100.0:
                success_count += 1

            eval_episodes.append({
                "episode_id": ep_id,
                "round": round_num,
                "split": effective_split,
                "total_reward": ep_reward,
                "steps": ep_steps,
                "is_success": bool(ep_steps and ep_steps[-1].get("is_success")),
                "metric_value": round(episode_metric, 2),
                "task_class": getattr(env, "task_class", "control"),
                "reward_breakdown": ep_steps[-1].get("reward_breakdown", {}) if ep_steps else {},
            })

        if getattr(env, "task_class", "control") == "classification" and eval_episodes:
            metric = round(
                sum(ep["metric_value"] for ep in eval_episodes) / len(eval_episodes), 2
            )
        else:
            metric = round((success_count / max(1, n_eval_episodes)) * 100.0, 2)
        illegal_rate = round((illegal_count / max(1, n_eval_episodes)) * 100.0, 2)
        return metric, eval_episodes, illegal_rate

    @staticmethod
    def _selection_split(env: gym.Env) -> str | None:
        """Split used while training/selecting: validation when present, else train."""
        available = getattr(env, "available_splits", None)
        if callable(available):
            try:
                names = available()
            except Exception:
                return None
            if "val" in names:
                return "val"
            if "train" in names:
                return "train"
        return None

    @staticmethod
    def _held_out_split(env: gym.Env) -> tuple[str | None, dict[str, int]]:
        """Prefers the untouched test split, then val, then whatever remains."""
        counts = env.split_counts() if callable(getattr(env, "split_counts", None)) else {}
        for name in ("test", "val", "train"):
            if counts.get(name):
                return name, counts
        return None, counts

    def _random_policy_baseline(
        self,
        env: gym.Env,
        split: str | None,
        n_episodes: int = 20,
        seed: int = 7,
    ) -> dict[str, Any]:
        """Uniform-random-action baseline, scored with the same episode metric as the policy."""
        rng = np.random.default_rng(seed)
        n_actions = int(getattr(env.action_space, "n", 1) or 1)
        values: list[float] = []
        for _ in range(max(1, n_episodes)):
            try:
                obs, info = env.reset(options={"split": split}) if split else env.reset()
            except (TypeError, ValueError):
                obs, info = env.reset()
            steps: list[dict[str, Any]] = []
            for _step in range(episode_step_limit(env)):
                obs, _reward, terminated, truncated, info = env.step(int(rng.integers(0, n_actions)))
                steps.append({
                    "is_success": bool(info.get("is_success", False)),
                    "is_correct": info.get("is_correct"),
                })
                if terminated or truncated:
                    break
            values.append(episode_metric_value(env, steps))
        return {
            "name": "random_policy",
            "value_pct": round(sum(values) / max(1, len(values)), 2),
            "detail": "uniform random actions over full episodes",
            "n": len(values),
        }


    # ------------------------------------------------------------------
    # Crossfire P3: corpus seeding, verifier gate, checkpoint selection
    # ------------------------------------------------------------------
    DEFAULT_RUBRIC = [
        "Held-out accuracy must exceed every measured baseline.",
        "No previously approved behavioural case may regress.",
        "Held-out accuracy must not fall beyond tolerance after a reward or environment edit.",
    ]

    @staticmethod
    def _term_keys(episodes: list[dict[str, Any]]) -> set[str]:
        keys: set[str] = set()
        for episode in episodes:
            for step in episode.get("steps", []) or []:
                keys.update((step.get("reward_breakdown") or {}).keys())
        return keys

    @staticmethod
    def _term_stats(
        episodes: list[dict[str, Any]], baseline_keys: set[str] | None
    ) -> dict[str, Any] | None:
        """Activation statistics for reward terms the previous round never produced."""
        if baseline_keys is None:
            return None
        steps = 0
        hits = 0
        total = 0.0
        for episode in episodes:
            for step in episode.get("steps", []) or []:
                steps += 1
                breakdown = step.get("reward_breakdown") or {}
                new_values = [
                    float(v)
                    for k, v in breakdown.items()
                    if k not in baseline_keys and abs(float(v)) > 1e-6
                ]
                if new_values:
                    hits += 1
                total += sum(new_values)
        new_terms = loop_new_terms(episodes, baseline_keys)
        if not new_terms:
            return None
        return {
            "new_terms": new_terms,
            "activation_rate": round(hits / max(1, steps), 4),
            "mean_term_value": round(total / max(1, steps), 6),
        }

    @staticmethod
    def _action_entropy(episodes: list[dict[str, Any]]) -> float | None:
        """Normalised Shannon entropy of the evaluated action distribution (nats)."""
        counts: dict[int, int] = {}
        for episode in episodes:
            for step in episode.get("steps", []) or []:
                action = step.get("action")
                if action is not None:
                    counts[int(action)] = counts.get(int(action), 0) + 1
        total = sum(counts.values())
        if total <= 1 or len(counts) < 2:
            return 0.0
        entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
        return round(entropy, 4)

    def _best_paths(self, session_id: str) -> tuple[Path, Path]:
        ws = self.store.base_dir / session_id
        return ws / "best_val_model.zip", ws / "best_val_env_module.py"

    def _record_best(
        self,
        session: SessionState,
        model: PPO,
        env_code: str,
        metric: float,
        outcomes: list[float],
    ) -> None:
        self.store.base_dir.mkdir(parents=True, exist_ok=True)
        model_zip, env_file = self._best_paths(session.session_id)
        model_zip.parent.mkdir(parents=True, exist_ok=True)
        try:
            model.save(str(model_zip))
        except Exception:
            return
        env_file.write_text(env_code, encoding="utf-8")
        session.best_val_metric = float(metric)
        session.best_val_round = int(session.current_round)
        self._best_outcomes[session.session_id] = [float(x) for x in outcomes]

    def _restore_best(
        self, session: SessionState, env: gym.Env
    ) -> tuple[PPO | None, str | None]:
        model_zip, env_file = self._best_paths(session.session_id)
        if not model_zip.exists():
            return None, None
        try:
            restored = PPO.load(str(model_zip), env=env, device="cpu")
        except Exception:
            return None, None
        self.store.set_model(session.session_id, restored)
        code = env_file.read_text(encoding="utf-8") if env_file.exists() else session.env_module_code
        if code:
            session.env_module_code = code
            (self.store.base_dir / session.session_id / "env_module.py").write_text(
                code, encoding="utf-8"
            )
        return restored, code

    @staticmethod
    def _predict_latency(predict: Callable[[list[float]], int], obs_dim: int) -> dict[str, float]:
        """Percentile latency of one greedy decision (torch path; ONNX is timed at export)."""
        import time as _time

        sample = [0.0] * max(1, obs_dim)
        for _ in range(10):  # warm-up so the first-call allocation is not the p99
            predict(sample)
        timings: list[float] = []
        for _ in range(120):
            started = _time.perf_counter()
            predict(sample)
            timings.append((_time.perf_counter() - started) * 1000.0)
        timings.sort()

        def _pct(p: float) -> float:
            idx = min(len(timings) - 1, max(0, int(round(p * (len(timings) - 1)))))
            return round(timings[idx], 4)

        return {"p50_ms": _pct(0.50), "p99_ms": _pct(0.99), "engine": "torch"}

    def _finalize_scorecard(
        self,
        session: SessionState,
        model: PPO,
        env: gym.Env,
        blueprint: Any,
        emit: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Evaluates on the untouched split, adds baselines, and persists the scorecard."""
        from rl_ide.autopilot.scorecard import (
            build_scorecard,
            compute_economics_summary,
            logreg_baseline,
            majority_class_baseline,
            uniform_chance_pct,
            wilson_ci,
        )

        eval_split, split_counts = self._held_out_split(env)
        task_class = str(getattr(env, "task_class", "control"))

        # Best-on-validation selection: the customer receives the checkpoint that scored best on
        # held-out selection data, never a degraded late round.
        selection: dict[str, Any] = {
            "selected_round": session.current_round,
            "best_val_round": session.best_val_round,
            "best_val_metric": session.best_val_metric,
            "source": "final_round",
        }
        best_zip, _best_env = self._best_paths(session.session_id)
        if (
            session.best_val_round is not None
            and session.best_val_round != session.current_round
            and best_zip.exists()
        ):
            try:
                selected = PPO.load(str(best_zip), env=env, device="cpu")
                model = selected
                self.store.set_model(session.session_id, selected)
                selection["selected_round"] = session.best_val_round
                selection["source"] = "best_val_checkpoint"
                model.save(str(Path("runs") / session.session_id / "model.zip"))
                emit({
                    "event": "checkpoint_selected",
                    "round": session.best_val_round,
                    "metric": session.best_val_metric,
                    "message": (
                        f"Selected round {session.best_val_round} checkpoint "
                        f"({session.best_val_metric}% val) for scoring and export."
                    ),
                })
            except Exception as sel_err:
                selection["source"] = f"final_round (selection failed: {type(sel_err).__name__})"

        metric, episodes, illegal = self.evaluate_policy_deterministic(
            env=env,
            model=model,
            n_eval_episodes=50,
            round_num=session.current_round,
            split=eval_split,
            phase="holdout",
        )
        session.episodes.extend(episodes)

        if task_class == "classification":
            graded = [s for ep in episodes for s in ep.get("steps", []) if s.get("is_correct") is not None]
            interval = wilson_ci(sum(1 for s in graded if s.get("is_correct")), len(graded))
        else:
            wins = sum(1 for ep in episodes if float(ep.get("metric_value", 0.0)) >= 100.0)
            interval = wilson_ci(wins, len(episodes))

        n_actions = int(getattr(env, "n_actions", getattr(env.action_space, "n", 1)) or 1)
        baselines: list[dict[str, Any]] = [
            {
                "name": "uniform_random",
                "value_pct": uniform_chance_pct(n_actions),
                "detail": "1/k uniform guess",
                "n": None,
            }
        ]
        if task_class == "classification":
            for baseline_fn in (majority_class_baseline, logreg_baseline):
                entry = baseline_fn(env, eval_split or "train")
                if entry:
                    baselines.append(entry)
        else:
            baselines.append(self._random_policy_baseline(env, eval_split))

        # Corpus, economics and the pre-registered verdict are part of the scorecard, so a
        # closure probe cannot be reported without them.
        predict = greedy_predictor(model)
        corpus_report = None
        try:
            corpus_report = self.corpus.evaluate_policy(session.session_id, predict)
            session.eval_corpus = corpus_report.model_dump()
        except Exception as corpus_err:
            emit({
                "event": "eval_corpus_unavailable",
                "round": session.current_round,
                "message": f"Corpus evaluation failed: {type(corpus_err).__name__}: {corpus_err}",
            })

        crossfire: dict[str, Any] = {"closed": False, "verdict": "not_preregistered", "reasons": []}
        try:
            if corpus_report is not None:
                protocol = self.protocol.load(session.session_id)
                if protocol is not None:
                    self.protocol.register_holdout(
                        session.session_id,
                        [c.case_id for c in self.corpus.load(session.session_id)],
                    )
                best_baseline = max((b["value_pct"] for b in baselines), default=None)
                crossfire = self.protocol.evaluate(
                    session.session_id,
                    holdout_passed=corpus_report.passed_cases,
                    holdout_total=corpus_report.scored_cases,
                    policy_metric=metric,
                    baseline_metric=best_baseline,
                )
            session.crossfire = crossfire
        except Exception as cf_err:
            crossfire = {
                "closed": False,
                "verdict": "inconclusive",
                "reasons": [f"Protocol evaluation failed: {type(cf_err).__name__}: {cf_err}"],
            }

        obs_dim = int(np.prod(getattr(env, "observation_space", np.zeros(1)).shape)) if hasattr(env, "observation_space") else 1
        latency_stats = self._predict_latency(predict, obs_dim)
        economics = compute_economics_summary(
            session.session_id,
            onnx_latency_stats=latency_stats,
            feature_pipeline_ms=0.0,
        )
        economics["latency"]["engine"] = latency_stats.get("engine", "unknown")

        scorecard = build_scorecard(
            task_title=getattr(blueprint, "task_title", session.session_id),
            task_class=task_class,
            metric_name=getattr(blueprint, "success_metric_name", None) or "success rate",
            split_counts=split_counts,
            eval_split=eval_split or "all",
            split_fallback=bool(getattr(env, "split_fallback", False)),
            policy_metric_pct=metric,
            episodes=len(episodes),
            illegal_rate_pct=illegal,
            baselines=baselines,
            ci95_pct=interval,
            rows_total=getattr(env, "dataset_rows_total", None),
            embedded_rows=len(getattr(env, "_records", []) or []) or None,
            notes=[
                "Scored without action masking: the exported ONNX policy applies no mask, so "
                "neither does the evaluation.",
                "Latency is measured on the in-process policy; the ONNX figure is recorded by the "
                "exporter and excludes the feature pipeline.",
            ],
            eval_corpus=corpus_report.model_dump() if corpus_report else None,
            crossfire=crossfire,
            economics=economics,
            selection=selection,
        )
        session.scorecard = scorecard

        scorecard_path = self.store.base_dir / session.session_id / "scorecard.json"
        scorecard_path.parent.mkdir(parents=True, exist_ok=True)
        scorecard_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

        emit({
            "event": "scorecard_ready",
            "round": session.current_round,
            "scorecard": scorecard,
            "message": (
                f"Held-out evaluation on the '{scorecard['eval_split']}' split: "
                f"{scorecard['policy']['value_pct']}% "
                f"(baseline best {max((b['value_pct'] for b in baselines), default=0.0)}%)."
            ),
        })
        return scorecard

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
        """Collects representative episodes while throttling live ARENA_UPDATE frames to UI."""
        collected_episodes: list[dict[str, Any]] = []
        stream_stride = max(1, n_episodes // 15)

        for ep_idx in range(1, n_episodes + 1):
            obs, info = env.reset()
            ep_steps = []
            ep_reward = 0.0
            step_idx = 0
            done = False
            ep_id = f"r{round_num}_train_{ep_idx}"
            step_limit = episode_step_limit(env)
            should_stream = emit_fn is not None and (ep_idx % stream_stride == 0 or ep_idx <= 2 or ep_idx == n_episodes)

            while not done and step_idx < step_limit:
                step_idx += 1
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    features = model.policy.extract_features(obs_t)
                    latent_pi, _ = model.policy.mlp_extractor(features)
                    logits = model.policy.action_net(latent_pi)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                    valid_actions = env.get_valid_actions() if hasattr(env, "get_valid_actions") else None
                    if valid_actions:
                        masked = np.zeros_like(probs)
                        for a in valid_actions:
                            if a < len(probs):
                                masked[a] = probs[a]
                        if masked.sum() > 0:
                            masked /= masked.sum()
                            action = int(np.random.choice(len(masked), p=masked))
                        else:
                            action = int(valid_actions[0])
                    else:
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
                    "is_correct": step_info.get("is_correct"),
                })

                # Stream throttled live frame over WebSocket / callback
                if should_stream and (step_idx % 2 == 1 or done):
                    emit_fn({
                        "event": "episode_step",
                        "round": round_num,
                        "episode_id": ep_id,
                        "step": step_idx,
                        "total_reward": round(ep_reward, 2),
                        "reward_breakdown": step_info.get("reward_breakdown", {}),
                        "arena_state": step_info.get("arena_state", {}),
                        "action_probs": [float(p) for p in probs],
                    })
                    import time
                    time.sleep(0.005)

                obs = next_obs

            collected_episodes.append({
                "episode_id": ep_id,
                "round": round_num,
                "total_reward": ep_reward,
                "steps": ep_steps,
                "is_success": bool(ep_steps and ep_steps[-1].get("is_success")),
                "metric_value": round(episode_metric_value(env, ep_steps), 2),
                "reward_breakdown": ep_steps[-1].get("reward_breakdown", {}) if ep_steps else {},
            })

        return collected_episodes
