"""AutoDataAdapter: Synthesizes a compliant DomainAdapter from any IngestedDataset or DataFrame."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from rl_ide.contracts.adapter import (
    ObservationSpec,
    ActionSpec,
    StepResult,
    DomainAdapter,
)
from rl_ide.ingest.profiler import (
    SchemaProfiler,
    DatasetProfile,
    AutoFeatureVectorizer,
)
from rl_ide.ingest.loader import IngestedDataset


class AutoDataAdapter:
    """Universal DomainAdapter dynamically synthesized for any SQL table, CSV/Excel file, or document stream."""

    def __init__(
        self,
        dataset: IngestedDataset | pd.DataFrame | None = None,
        action_labels: list[str] | None = None,
        episode_length: int = 25,
        target_col: str | None = None,
        horizon_type: str = "contextual_bandit",
    ):
        if dataset is None:
            dataset = pd.DataFrame({
                "amount": [100.0, 5500.0, 20.0, 12000.0],
                "risk_score": [0.1, 0.9, 0.2, 0.85],
                "has_po": [True, False, True, False],
                "department": ["sales", "engineering", "sales", "ops"],
                "target": ["pass", "flag", "pass", "flag"],
            })

        if isinstance(dataset, IngestedDataset):
            self.dataset = dataset
            self.df = dataset.df.copy()
            self.role = dataset.role
            self.profile = dataset.profile or SchemaProfiler().profile(self.df, target_col=target_col)
        else:
            self.dataset = None
            self.df = dataset.copy()
            self.role = "tabular"
            self.profile = SchemaProfiler().profile(self.df, target_col=target_col)

        if len(self.df) == 0:
            raise ValueError("Cannot initialize AutoDataAdapter with an empty dataset.")

        self.episode_length = min(max(1, episode_length), len(self.df))
        self.target_col = target_col or self.profile.target_candidate
        self.horizon_type = horizon_type

        # UI mode determination
        if self.role in ("document_records", "knowledge_rules"):
            self.ui_mode = "document_reader"
        else:
            self.ui_mode = "tabular_inspector"

        # Initialize feature vectorizer
        self.vectorizer = AutoFeatureVectorizer(self.profile, self.df)

        # Inferred or explicit action labels
        if action_labels:
            self.action_labels = list(action_labels)
        elif self.target_col and self.profile.get_column(self.target_col):
            t_col = self.profile.get_column(self.target_col)
            if t_col and t_col.categories and len(t_col.categories) <= 10:
                self.action_labels = [str(c) for c in t_col.categories]
            else:
                self.action_labels = ["flag", "pass"]
        else:
            self.action_labels = ["flag", "pass"]

        self.rng = np.random.default_rng(42)
        self.episode_id = 0
        self.current_step = 0
        self.current_row_idx = 0
        self.start_idx = 0

    def _get_obs_vector(self, idx: int) -> list[float]:
        """Provides direct index-based observation vector for behavioral cloning warmstart."""
        return self.vectorizer.get_obs_at(idx)

    def observations(self) -> ObservationSpec:
        """Returns observation specification for SB3 policy networks."""
        return ObservationSpec(
            vector_dim=self.vectorizer.vector_dim,
            feature_names=self.vectorizer.feature_names,
            horizon_type=self.horizon_type,
            default_gamma=0.0 if self.horizon_type == "contextual_bandit" else 0.90,
        )

    def actions(self) -> ActionSpec:
        """Returns action specification."""
        return ActionSpec(
            action_type="discrete",
            labels=self.action_labels,
        )

    def event_schema(self) -> dict[str, str]:
        """Returns dynamic schema mapping for AST Reward DSL validation."""
        return self.profile.event_schema()

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        """Resets the environment for a new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_id += 1
        self.current_step = 0

        # Pick random starting slice in dataset
        max_start = max(0, len(self.df) - self.episode_length)
        self.start_idx = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        self.current_row_idx = self.start_idx

        obs_vector = self.vectorizer.get_obs_at(self.current_row_idx)
        return obs_vector, {
            "episode_id": self.episode_id,
            "step": 0,
            "row_idx": self.current_row_idx,
            "total_rows": len(self.df),
            "ui_mode": self.ui_mode,
        }

    def step(self, action_idx: int) -> StepResult:
        """Executes a single step given action index."""
        clamped_idx = max(0, min(action_idx, len(self.action_labels) - 1))
        action_label = self.action_labels[clamped_idx]

        # Extract current row data
        row = self.df.iloc[self.current_row_idx]

        # Formulate obs event dict for AST evaluator
        obs_dict: dict[str, Any] = {}
        for col_prof in self.profile.columns:
            raw_val = row.get(col_prof.name)
            if col_prof.python_type == "float":
                try:
                    obs_dict[col_prof.sanitized_name] = float(raw_val) if not pd.isna(raw_val) else 0.0
                except (ValueError, TypeError):
                    obs_dict[col_prof.sanitized_name] = 0.0
            elif col_prof.python_type == "int":
                try:
                    obs_dict[col_prof.sanitized_name] = int(raw_val) if not pd.isna(raw_val) else 0
                except (ValueError, TypeError):
                    obs_dict[col_prof.sanitized_name] = 0
            elif col_prof.python_type == "bool":
                obs_dict[col_prof.sanitized_name] = bool(raw_val) and not pd.isna(raw_val) and str(raw_val).lower() not in ("0", "false", "no")
            else:
                obs_dict[col_prof.sanitized_name] = str(raw_val)[:150] if not pd.isna(raw_val) else ""

        # Formulate truth event dict
        truth_dict: dict[str, Any] = {
            "has_label": False,
            "label": "",
            "target": 0.0,
            "is_anomaly": False,
        }
        if self.target_col and self.target_col in row.index:
            t_val = row[self.target_col]
            if not pd.isna(t_val):
                truth_dict["has_label"] = True
                truth_dict["label"] = str(t_val)
                try:
                    truth_dict["target"] = float(t_val)
                except (ValueError, TypeError):
                    truth_dict["target"] = 0.0
                if str(t_val).lower() in ("true", "1", "anomaly", "fraud", "urgent", "flag"):
                    truth_dict["is_anomaly"] = True

        state_dict = {
            "step_in_episode": int(self.current_step),
            "row_idx": int(self.current_row_idx),
            "total_rows": int(len(self.df)),
        }

        events = {
            "obs": obs_dict,
            "action": {
                "label": action_label,
                "action_idx": int(clamped_idx),
            },
            "state": state_dict,
            "truth": truth_dict,
        }

        # Step progression
        self.current_step += 1
        self.current_row_idx = (self.current_row_idx + 1) % len(self.df)

        terminated = self.current_step >= self.episode_length
        truncated = False

        next_obs_vec = self.vectorizer.get_obs_at(self.current_row_idx)

        record_summary = {
            k: (v if not pd.isna(v) else None)
            for k, v in row.to_dict().items()
        }

        info = {
            "step": self.current_step,
            "episode_id": self.episode_id,
            "action_label": action_label,
            "record": record_summary,
            "ui_mode": self.ui_mode,
        }
        if truth_dict["has_label"]:
            info["is_correct"] = (action_label == truth_dict["label"])

        return StepResult(
            obs_vector=next_obs_vec,
            events=events,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def render_frame(
        self,
        episode_id: int,
        step_result: StepResult,
        total_reward: float,
        breakdown: dict[str, float],
    ) -> dict[str, Any]:
        """Produces structured polymorphic UI telemetry."""
        return {
            "ui_mode": self.ui_mode,
            "episode_id": episode_id,
            "step": step_result.info.get("step", self.current_step),
            "obs": step_result.events.get("obs", {}),
            "action": step_result.events.get("action", {}),
            "state": step_result.events.get("state", {}),
            "truth": step_result.events.get("truth", {}),
            "record": step_result.info.get("record", {}),
            "total_reward": float(total_reward),
            "breakdown": breakdown,
            "terminated": step_result.terminated,
        }
