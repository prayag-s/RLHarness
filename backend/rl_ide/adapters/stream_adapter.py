"""CSV Stream Adapter with synthetic financial / sensor time-series generation."""

from __future__ import annotations
from collections import deque
from typing import Any
import numpy as np
import pandas as pd
from rl_ide.contracts.adapter import (
    ObservationSpec,
    ActionSpec,
    StepResult,
    DomainAdapter,
)


def generate_synthetic_stream(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generates a deterministic synthetic time-series stream.
    
    Produces columns:
        timestamp, price, return_1m, volume, volatility, rolling_vol_90p, z_score, true_anomaly
        
    Design invariant:
        High volatility spikes are normal unless accompanied by extreme volume / z-score anomalies.
    """
    rng = np.random.default_rng(seed)

    # Base timestamps (1-minute intervals)
    base_time = pd.Timestamp("2026-01-01 09:30:00")
    timestamps = [str(base_time + pd.Timedelta(minutes=i)) for i in range(n_rows)]

    # Generate baseline returns with fat tails (t-distribution / normal mixture)
    returns = rng.normal(0.0001, 0.004, size=n_rows)
    volumes = rng.normal(10000.0, 1500.0, size=n_rows).clip(min=1000.0)

    # Inject periodic high volatility clusters that are NORMAL (e.g. market open/lunch chop)
    # Every ~100 rows, inject high return volatility for 10-15 rows
    cluster_starts = [100, 250, 420, 600, 780, 910]
    for start in cluster_starts:
        if start < n_rows:
            length = min(15, n_rows - start)
            # Volatility is high, but volume is normal and no structural anomaly
            returns[start : start + length] = rng.normal(0.0002, 0.015, size=length)

    # Calculate rolling volatility (std over 20 periods)
    vol_series = pd.Series(returns).rolling(window=20, min_periods=1).std().bfill().fillna(0.004)
    # 90th percentile threshold over 50-step window
    rolling_vol_90p = vol_series.rolling(window=50, min_periods=10).quantile(0.90).bfill().fillna(0.012)

    # Rolling mean for z-score
    rolling_mean = pd.Series(returns).rolling(window=20, min_periods=1).mean().bfill().fillna(0.0)
    z_scores = ((pd.Series(returns) - rolling_mean) / (vol_series + 1e-8)).fillna(0.0).values

    # Flags for true anomalies
    true_anomalies = np.zeros(n_rows, dtype=bool)

    # Inject true anomalies: extreme volume (4x-8x) + extreme z-score (|z| > 3.0)
    # Deterministic anomaly indices across the stream
    anomaly_indices = [65, 180, 340, 520, 710, 850]
    for idx in anomaly_indices:
        if idx < n_rows:
            # Huge jump return + 5x-8x volume spike
            sign = 1 if rng.random() > 0.5 else -1
            returns[idx] = sign * rng.uniform(0.035, 0.065)
            volumes[idx] = volumes[idx] * rng.uniform(4.5, 8.0)
            true_anomalies[idx] = True

    # Re-calculate metrics incorporating the true anomalies
    vol_series = pd.Series(returns).rolling(window=20, min_periods=1).std().bfill().fillna(0.004)
    rolling_vol_90p = vol_series.rolling(window=50, min_periods=10).quantile(0.90).bfill().fillna(0.012)
    rolling_mean = pd.Series(returns).rolling(window=20, min_periods=1).mean().bfill().fillna(0.0)
    z_scores = ((pd.Series(returns) - rolling_mean) / (vol_series + 1e-8)).fillna(0.0).values

    # Ensure anomalies have |z_score| > 2.5
    for idx in anomaly_indices:
        if idx < n_rows:
            if abs(z_scores[idx]) < 2.5:
                z_scores[idx] = 3.5 if returns[idx] > 0 else -3.5

    # Compute price trajectory from returns
    prices = np.zeros(n_rows, dtype=float)
    prices[0] = 100.0
    for i in range(1, n_rows):
        prices[i] = max(0.5, prices[i - 1] * (1.0 + returns[i]))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "return_1m": returns,
        "volume": volumes,
        "volatility": vol_series.values,
        "rolling_vol_90p": rolling_vol_90p.values,
        "z_score": z_scores,
        "true_anomaly": true_anomalies,
    })

    return df


class CSVStreamAdapter:
    """Stateful MDP Adapter for streaming time-series anomaly detection."""

    FEATURE_NAMES = [
        "price",
        "return_1m",
        "volume",
        "volatility",
        "rolling_vol_90p",
        "z_score",
        "cooldown",
        "steps_since_last_alert",
        "rolling_alert_rate",
    ]

    ACTION_LABELS = ["normal", "anomaly"]

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        episode_length: int = 100,
        cooldown_steps: int = 3,
        rolling_window: int = 20,
    ):
        self.df = df if df is not None else generate_synthetic_stream(n_rows=1000, seed=42)
        self.episode_length = episode_length
        self.cooldown_steps = cooldown_steps
        self.rolling_window = rolling_window

        self.episode_id = 0
        self.current_idx = 0
        self.step_in_episode = 0

        # Stateful MDP tracking
        self.cooldown = 0
        self.steps_since_last_alert = 0
        self.alert_history: deque[int] = deque(maxlen=rolling_window)

    def observations(self) -> ObservationSpec:
        return ObservationSpec(
            vector_dim=len(self.FEATURE_NAMES),
            feature_names=self.FEATURE_NAMES,
            horizon_type="stateful_mdp",
            default_gamma=0.90,
        )

    def actions(self) -> ActionSpec:
        return ActionSpec(
            action_type="discrete",
            labels=self.ACTION_LABELS,
        )

    def event_schema(self) -> dict[str, str]:
        return {
            "obs.price": "float",
            "obs.return_1m": "float",
            "obs.volume": "float",
            "obs.volatility": "float",
            "obs.rolling_vol_90p": "float",
            "obs.z_score": "float",
            "action.label": "str",
            "action.action_idx": "int",
            "state.cooldown": "int",
            "state.steps_since_last_alert": "int",
            "state.rolling_alert_rate": "float",
            "truth.has_label": "bool",
            "truth.label": "str",
            "truth.true_anomaly": "bool",
        }

    def _get_obs_vector(self, row_idx: int) -> list[float]:
        row = self.df.iloc[row_idx]
        alert_rate = (
            sum(self.alert_history) / max(1, len(self.alert_history))
            if self.alert_history
            else 0.0
        )
        return [
            float(row["price"]),
            float(row["return_1m"]),
            float(row["volume"]),
            float(row["volatility"]),
            float(row["rolling_vol_90p"]),
            float(row["z_score"]),
            float(self.cooldown),
            float(self.steps_since_last_alert),
            float(alert_rate),
        ]

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        self.step_in_episode = 0
        self.cooldown = 0
        self.steps_since_last_alert = 0
        self.alert_history.clear()

        if seed is not None:
            rng = np.random.default_rng(seed)
            max_start = max(0, len(self.df) - self.episode_length - 1)
            self.current_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        else:
            # Sequence deterministically through dataset by episode_id
            max_start = max(1, len(self.df) - self.episode_length)
            self.current_idx = (self.episode_id * self.episode_length) % max_start

        self.episode_id += 1
        obs_vector = self._get_obs_vector(self.current_idx)
        return obs_vector, {"episode_id": self.episode_id, "start_index": self.current_idx}

    def step(self, action_idx: int) -> StepResult:
        if action_idx < 0 or action_idx >= len(self.ACTION_LABELS):
            raise ValueError(f"action_idx {action_idx} out of range [0, {len(self.ACTION_LABELS)-1}]")

        action_label = self.ACTION_LABELS[action_idx]
        is_alert = (action_label == "anomaly")

        row = self.df.iloc[self.current_idx]
        is_true_anomaly = bool(row["true_anomaly"])
        true_label = "anomaly" if is_true_anomaly else "normal"

        obs = {
            "price": float(row["price"]),
            "return_1m": float(row["return_1m"]),
            "volume": float(row["volume"]),
            "volatility": float(row["volatility"]),
            "rolling_vol_90p": float(row["rolling_vol_90p"]),
            "z_score": float(row["z_score"]),
        }

        truth = {
            "has_label": True,
            "label": true_label,
            "true_anomaly": is_true_anomaly,
        }

        action = {
            "action_idx": int(action_idx),
            "label": action_label,
        }

        # MDP state updates
        if is_alert:
            self.cooldown = self.cooldown_steps
            self.steps_since_last_alert = 0
            self.alert_history.append(1)
        else:
            if self.cooldown > 0:
                self.cooldown -= 1
            self.steps_since_last_alert += 1
            self.alert_history.append(0)

        rolling_alert_rate = sum(self.alert_history) / max(1, len(self.alert_history))

        state = {
            "cooldown": int(self.cooldown),
            "steps_since_last_alert": int(self.steps_since_last_alert),
            "rolling_alert_rate": float(rolling_alert_rate),
        }

        events = {
            "obs": obs,
            "action": action,
            "state": state,
            "truth": truth,
        }

        self.step_in_episode += 1
        self.current_idx += 1

        terminated = (
            self.step_in_episode >= self.episode_length
            or self.current_idx >= len(self.df)
        )
        truncated = False

        # Next obs vector
        next_row_idx = self.current_idx if not terminated else self.current_idx - 1
        next_obs = self._get_obs_vector(next_row_idx)

        return StepResult(
            obs_vector=next_obs,
            events=events,
            terminated=terminated,
            truncated=truncated,
            info={"step": self.step_in_episode, "episode_id": self.episode_id},
        )

    def render_frame(
        self,
        episode_id: int,
        step_result: StepResult,
        total_reward: float,
        breakdown: dict[str, float]
    ) -> dict[str, Any]:
        return {
            "episode_id": episode_id,
            "step": self.step_in_episode,
            "obs": step_result.events.get("obs", {}),
            "action": step_result.events.get("action", {}),
            "state": step_result.events.get("state", {}),
            "truth": step_result.events.get("truth", {}),
            "total_reward": total_reward,
            "breakdown": breakdown,
            "terminated": step_result.terminated,
        }
