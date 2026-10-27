"""Training Runner with Warm-Start, SB3 PPO micro-bursts, and Critic Calibration."""

from __future__ import annotations
import copy
from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

from rl_ide.contracts.adapter import DomainAdapter, EpisodeLog
from rl_ide.reward.dsl import RewardDSL
from rl_ide.reward.evaluator import RewardEvaluator
from rl_ide.trainer.gym_env import AdapterGymEnv
from rl_ide.trainer.buffer import EpisodeBuffer
from rl_ide.state.store import TaskStateStore, TaskVersion


class TrainingRunner:
    """Orchestrates Warm-Start Behavioral Cloning, streaming PPO micro-bursts, and Critic Warmup."""

    def __init__(
        self,
        adapter: DomainAdapter,
        reward_dsl: RewardDSL,
        state_store: TaskStateStore | None = None,
        task_id: str = "default_task",
        hyperparameters: dict[str, Any] | None = None,
    ):
        self.adapter = adapter
        self.reward_dsl = reward_dsl
        self.state_store = state_store
        self.task_id = task_id
        self.hyperparameters = hyperparameters or {
            "learning_rate": 0.0003,
            "gamma": 0.90,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 10,
        }

        self.evaluator = RewardEvaluator(self.reward_dsl, schema=self.adapter.event_schema())
        self.gym_env = AdapterGymEnv(self.adapter, self.evaluator)
        self.buffer = EpisodeBuffer(max_episodes=200)

        # Wire gym_env callback to push completed episodes to buffer
        self._external_callback: Callable[[EpisodeLog], None] | None = None

        def _on_episode_finished(ep_log: EpisodeLog) -> None:
            self.buffer.append(ep_log)
            if self._external_callback:
                self._external_callback(ep_log)

        self.gym_env.on_episode_callback = _on_episode_finished

        # Initialize SB3 PPO on CPU (ensuring device consistency and fast tabular execution)
        self.model = PPO(
            "MlpPolicy",
            self.gym_env,
            learning_rate=float(self.hyperparameters.get("learning_rate", 3e-4)),
            gamma=float(self.hyperparameters.get("gamma", 0.90)),
            n_steps=int(self.hyperparameters.get("n_steps", 64)),
            batch_size=int(self.hyperparameters.get("batch_size", 32)),
            n_epochs=int(self.hyperparameters.get("n_epochs", 10)),
            ent_coef=0.01,
            device="cpu",
            verbose=0,
        )

    def warmstart_policy(
        self,
        epochs: int = 15,
        n_samples: int = 200,
        heuristic_flags_volatility: bool | None = None,
    ) -> dict[str, float]:
        """Supervised behavioral cloning pass on initial rules to avoid cold-start random actions."""
        states: list[list[float]] = []
        actions: list[int] = []

        # Check if initial DSL rewards volatility alerts
        if heuristic_flags_volatility is None:
            heuristic_flags_volatility = any(
                "volatility" in term.name and float(term.value) > 0
                for term in self.reward_dsl.reward.terms
            )

        # Sample demonstration steps from the adapter's domain
        if hasattr(self.adapter, "df"):
            df = getattr(self.adapter, "df")
            sample_indices = np.linspace(0, len(df) - 1, num=min(n_samples, len(df)), dtype=int)
            labels_list = getattr(self.adapter, "ACTION_LABELS", None)
            for idx in sample_indices:
                row = df.iloc[idx]
                if "true_label" in row and labels_list is not None:
                    lbl = str(row["true_label"])
                    act = labels_list.index(lbl) if lbl in labels_list else 0
                else:
                    is_anomaly = bool(row.get("true_anomaly", False))
                    is_high_vol = bool(row.get("volatility", 0.0) > 0.005)
                    if heuristic_flags_volatility and is_high_vol:
                        act = 1
                    elif is_anomaly:
                        act = 1
                    else:
                        act = 0
                obs_vec = getattr(self.adapter, "_get_obs_vector")(idx)
                states.append(obs_vec)
                actions.append(act)
        else:
            # Fallback: step through adapter to collect states
            obs, _ = self.adapter.reset(seed=42)
            states.append(obs)
            actions.append(0)
            for _ in range(n_samples - 1):
                res = self.adapter.step(0)
                states.append(res.obs_vector)
                actions.append(1 if res.events.get("truth", {}).get("true_anomaly") else 0)
                if res.terminated:
                    self.adapter.reset()

        states_t = torch.tensor(states, dtype=torch.float32, device="cpu")
        actions_t = torch.tensor(actions, dtype=torch.long, device="cpu")

        policy = self.model.policy
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        initial_loss: float = 0.0
        final_acc: float = 0.0

        for epoch in range(epochs):
            features = policy.extract_features(states_t)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
            loss = criterion(logits, actions_t)

            if epoch == 0:
                initial_loss = float(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            final_acc = float((preds == actions_t).float().mean().item())

        return {
            "initial_loss": initial_loss,
            "final_loss": float(loss.item()),
            "accuracy": final_acc,
        }

    def run_burst(
        self,
        n_episodes: int = 30,
        on_episode_callback: Callable[[EpisodeLog], None] | None = None
    ) -> list[EpisodeLog]:
        """Runs a micro-burst of PPO training, streaming each completed episode."""
        self._external_callback = on_episode_callback
        episodes_before = len(self.buffer)

        # Calculate steps per episode
        ep_len = getattr(self.adapter, "episode_length", 100)
        total_steps = n_episodes * ep_len

        # Execute SB3 learn
        self.model.learn(total_timesteps=total_steps, reset_num_timesteps=False)

        # Retrieve episodes collected during this burst
        all_episodes = self.buffer.all()
        burst_episodes = all_episodes[episodes_before:]
        return burst_episodes

    def apply_reward_diff(
        self,
        new_dsl: RewardDSL,
        critique_text: str | None = None
    ) -> TaskVersion | None:
        """Applies a confirmed Reward DSL diff:
        1. Saves snapshot to TaskStateStore.
        2. Hot-swaps RewardEvaluator and re-scores EpisodeBuffer.
        3. Runs Critic Warmup for 15 mini-epochs on the re-scored buffer.
        """
        # 1. Save snapshot to store if configured
        snapshot_version = None
        if self.state_store is not None:
            ckpt_path = self.state_store.base_dir / f"checkpoint_{self.task_id}.pt"
            torch.save(self.model.policy.state_dict(), str(ckpt_path))

            # Retrieve TaskConfig if exists
            from rl_ide.compiler.models import TaskConfig
            current_ver = self.state_store.get_current_version(self.task_id)
            if current_ver:
                cfg = current_ver.task_config
                cfg.reward_dsl = new_dsl
            else:
                cfg = TaskConfig(
                    task_id=self.task_id,
                    adapter_name=type(self.adapter).__name__,
                    reward_dsl=new_dsl,
                )

            snapshot_version = self.state_store.save_version(
                task_id=self.task_id,
                task_config=cfg,
                reward_dsl=new_dsl,
                applied_critique=critique_text,
                policy_checkpoint_path=str(ckpt_path),
            )

        # 2. Hot-swap RewardEvaluator & re-score buffer
        self.reward_dsl = new_dsl
        self.evaluator = RewardEvaluator(new_dsl, schema=self.adapter.event_schema())
        self.gym_env.set_evaluator(self.evaluator)
        self.buffer.rescore(self.evaluator)

        # 3. Critic Warmup (15 epochs)
        warmup_epochs = new_dsl.horizon.critic_warmup_epochs_on_diff or 15
        self._warmup_critic(epochs=warmup_epochs)

        return snapshot_version

    def _warmup_critic(self, epochs: int = 15) -> None:
        """Calibrates value network (vf) on re-scored buffer transitions without updating actor weights."""
        all_episodes = self.buffer.all()
        if not all_episodes:
            return

        all_obs: list[list[float]] = []
        all_returns: list[float] = []
        gamma = self.reward_dsl.horizon.gamma

        for ep in all_episodes:
            rewards = ep.step_rewards
            n = len(rewards)
            # Compute discounted returns G_t
            discounted = np.zeros(n, dtype=np.float32)
            running_add = 0.0
            for t in reversed(range(n)):
                running_add = rewards[t] + gamma * running_add
                discounted[t] = running_add

            for step, g_val in zip(ep.steps, discounted):
                all_obs.append(step.obs_vector)
                all_returns.append(float(g_val))

        if not all_obs:
            return

        obs_t = torch.tensor(all_obs, dtype=torch.float32, device="cpu")
        returns_t = torch.tensor(all_returns, dtype=torch.float32, device="cpu").unsqueeze(-1)

        # Freeze Actor parameters
        for p in self.model.policy.action_net.parameters():
            p.requires_grad = False

        # Unfreeze Value network
        for p in self.model.policy.value_net.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.model.policy.value_net.parameters(), lr=1e-3)
        dataset = torch.utils.data.TensorDataset(obs_t, returns_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for _ in range(epochs):
            for b_obs, b_ret in loader:
                values = self.model.policy.predict_values(b_obs)
                loss = F.mse_loss(values, b_ret)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Unfreeze Actor parameters
        for p in self.model.policy.action_net.parameters():
            p.requires_grad = True
