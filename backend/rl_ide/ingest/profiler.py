"""Schema Profiler, AutoFeatureVectorizer, and DocumentRuleExtractor for universal data ingestion."""

from __future__ import annotations
import re
from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import HashingVectorizer


def sanitize_column_name(col: str) -> str:
    """Sanitizes raw column names into valid Python identifiers for the AST Reward DSL."""
    name = str(col).strip()
    # Replace non-alphanumeric chars with underscore
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Collapse consecutive underscores
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "col"
    # Ensure it starts with a letter or underscore, not a digit
    if clean[0].isdigit():
        clean = f"f_{clean}"
    return clean


class ColumnProfile(BaseModel):
    """Profile of a single ingested dataset column."""
    name: str
    sanitized_name: str
    data_type: str  # "numeric_continuous" | "numeric_discrete" | "categorical_low_card" | "text_freeform" | "boolean" | "timestamp" | "target_candidate"
    python_type: str  # "float" | "int" | "str" | "bool"
    num_unique: int
    missing_count: int
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    std_val: float | None = None
    categories: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """Comprehensive schema profile of an ingested dataset."""
    columns: list[ColumnProfile]
    row_count: int
    target_candidate: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    @property
    def column_types(self) -> dict[str, str]:
        """Provides a direct dictionary mapping column name -> detected data_type."""
        return {col.name: col.data_type for col in self.columns}

    def get_column(self, name_or_sanitized: str) -> ColumnProfile | None:
        for col in self.columns:
            if col.name == name_or_sanitized or col.sanitized_name == name_or_sanitized:
                return col
        return None

    def event_schema(self) -> dict[str, str]:
        """Builds {field_path: type_str} for RewardValidator AST check."""
        schema: dict[str, str] = {
            "action.label": "str",
            "action.action_idx": "int",
            "state.step_in_episode": "int",
            "state.row_idx": "int",
            "state.total_rows": "int",
            "truth.has_label": "bool",
            "truth.label": "str",
            "truth.target": "float",
            "truth.is_anomaly": "bool",
        }
        for col in self.columns:
            schema[f"obs.{col.sanitized_name}"] = col.python_type
        return schema


class SchemaProfiler:
    """Profiles datasets, infers semantic types, and builds vectorization specifications."""

    TARGET_NAME_PATTERNS = [
        r"^label$", r"^target$", r"^status$", r"^outcome$", r"^action$",
        r"^class$", r"^true_anomaly$", r"^is_fraud$", r"^is_anomaly$",
        r"^decision$", r"^category$", r"^priority$"
    ]

    def profile(self, df: pd.DataFrame, target_col: str | None = None) -> DatasetProfile:
        columns: list[ColumnProfile] = []
        n_rows = len(df)
        detected_target = target_col

        # Sanitized column mapping to prevent collisions
        sanitized_seen: set[str] = set()

        for raw_col in df.columns:
            col_series = df[raw_col]
            sanitized = sanitize_column_name(str(raw_col))
            suffix = 1
            orig_sanitized = sanitized
            while sanitized in sanitized_seen:
                sanitized = f"{orig_sanitized}_{suffix}"
                suffix += 1
            sanitized_seen.add(sanitized)

            num_unique = int(col_series.nunique(dropna=True))
            missing = int(col_series.isna().sum())

            # Detect if this is a target label candidate
            is_target = False
            if target_col is not None and (raw_col == target_col or sanitized == target_col):
                is_target = True
                detected_target = raw_col
            elif target_col is None and detected_target is None:
                for pat in self.TARGET_NAME_PATTERNS:
                    if re.search(pat, str(raw_col), re.IGNORECASE):
                        is_target = True
                        detected_target = raw_col
                        break

            # Classify type
            dtype = col_series.dtype
            min_val, max_val, mean_val, std_val = None, None, None, None
            categories: list[str] = []

            if is_target:
                data_type = "target_candidate"
                python_type = "str" if col_series.dtype == object else "float"
                if num_unique <= 50:
                    categories = [str(x) for x in col_series.dropna().unique().tolist()]
            elif pd.api.types.is_bool_dtype(dtype) or (num_unique == 2 and set(col_series.dropna().unique()).issubset({0, 1, True, False, "0", "1", "true", "false", "True", "False"})):
                data_type = "boolean"
                python_type = "bool"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                data_type = "timestamp"
                python_type = "str"
            elif pd.api.types.is_numeric_dtype(dtype):
                valid_num = col_series.dropna()
                if len(valid_num) > 0:
                    min_val = float(valid_num.min())
                    max_val = float(valid_num.max())
                    mean_val = float(valid_num.mean())
                    std_val = float(valid_num.std()) if len(valid_num) > 1 else 0.0
                    if np.isnan(std_val):
                        std_val = 0.0

                if pd.api.types.is_integer_dtype(dtype) and num_unique <= 20:
                    data_type = "numeric_discrete"
                    python_type = "int"
                    categories = [str(x) for x in valid_num.unique().tolist()]
                else:
                    data_type = "numeric_continuous"
                    python_type = "float"
            else:
                # String / Object
                non_null = col_series.dropna().astype(str)
                avg_len = float(non_null.str.len().mean()) if len(non_null) > 0 else 0.0

                if avg_len > 35 or num_unique > 50:
                    data_type = "text_freeform"
                    python_type = "str"
                else:
                    data_type = "categorical_low_card"
                    python_type = "str"
                    categories = [str(x) for x in non_null.unique()[:30].tolist()]

            columns.append(ColumnProfile(
                name=str(raw_col),
                sanitized_name=sanitized,
                data_type=data_type,
                python_type=python_type,
                num_unique=num_unique,
                missing_count=missing,
                min_val=min_val,
                max_val=max_val,
                mean_val=mean_val,
                std_val=std_val,
                categories=categories,
            ))

        feature_cols = [
            col.name for col in columns
            if col.data_type != "target_candidate" and col.data_type != "timestamp"
        ]

        return DatasetProfile(
            columns=columns,
            row_count=n_rows,
            target_candidate=str(detected_target) if detected_target is not None else None,
            feature_columns=feature_cols,
            summary={"num_features": len(feature_cols), "num_rows": n_rows},
        )


class AutoFeatureVectorizer:
    """Transforms raw structured tabular and text records into normalized float32 observation vectors."""

    def __init__(self, profile: DatasetProfile, df: pd.DataFrame):
        self.profile = profile
        self.feature_names: list[str] = []
        self._numeric_stats: dict[str, tuple[float, float]] = {}  # col -> (mean, std)
        self._cat_vocab: dict[str, list[str]] = {}  # col -> categories
        self._text_vectorizers: dict[str, HashingVectorizer] = {}

        # 1. Inspect and setup each feature column
        for col_name in profile.feature_columns:
            col_prof = profile.get_column(col_name)
            if not col_prof:
                continue

            if col_prof.data_type in ("numeric_continuous", "numeric_discrete"):
                mean = col_prof.mean_val if col_prof.mean_val is not None else 0.0
                std = col_prof.std_val if (col_prof.std_val is not None and col_prof.std_val > 1e-6) else 1.0
                self._numeric_stats[col_name] = (mean, std)
                self.feature_names.append(col_prof.sanitized_name)

            elif col_prof.data_type == "boolean":
                self.feature_names.append(col_prof.sanitized_name)

            elif col_prof.data_type == "categorical_low_card":
                cats = col_prof.categories[:20]  # Cap at top 20 categories
                self._cat_vocab[col_name] = cats
                for cat in cats:
                    safe_cat = re.sub(r"[^a-zA-Z0-9_]", "_", str(cat))
                    self.feature_names.append(f"{col_prof.sanitized_name}_is_{safe_cat}")

            elif col_prof.data_type == "text_freeform":
                # Deterministic 32-dim hashing vectorizer
                vec = HashingVectorizer(n_features=32, alternate_sign=False, norm="l2")
                self._text_vectorizers[col_name] = vec
                for i in range(32):
                    self.feature_names.append(f"{col_prof.sanitized_name}_hash_{i}")

        self.vector_dim = max(len(self.feature_names), 1)
        if len(self.feature_names) == 0:
            self.feature_names = ["dummy_bias"]
            self.vector_dim = 1

        # Pre-cache transformed matrices for full df for high throughput stepping
        self._cached_obs_matrix = self.transform_df(df)

    def transform_row(self, row: dict[str, Any] | pd.Series) -> list[float]:
        """Vectorizes a single record row into float32 values."""
        feats: list[float] = []

        for col_name in self.profile.feature_columns:
            col_prof = self.profile.get_column(col_name)
            if not col_prof:
                continue

            raw_val = row.get(col_name) if isinstance(row, dict) else row[col_name]

            if col_prof.data_type in ("numeric_continuous", "numeric_discrete"):
                try:
                    val = float(raw_val) if (raw_val is not None and not pd.isna(raw_val)) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                mean, std = self._numeric_stats[col_name]
                norm_val = float(np.clip((val - mean) / std, -5.0, 5.0))
                feats.append(norm_val)

            elif col_prof.data_type == "boolean":
                val = 1.0 if bool(raw_val) and not pd.isna(raw_val) and str(raw_val).lower() not in ("0", "false", "no") else 0.0
                feats.append(val)

            elif col_prof.data_type == "categorical_low_card":
                cats = self._cat_vocab.get(col_name, [])
                str_val = str(raw_val) if raw_val is not None and not pd.isna(raw_val) else ""
                for cat in cats:
                    feats.append(1.0 if str_val == cat else 0.0)

            elif col_prof.data_type == "text_freeform":
                text = str(raw_val) if raw_val is not None and not pd.isna(raw_val) else ""
                vec = self._text_vectorizers[col_name]
                h_arr = vec.transform([text]).toarray()[0].tolist()
                feats.extend(h_arr)

        if not feats:
            feats = [1.0]

        return feats

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """Batch vectorizes an entire DataFrame into a 2D float32 array."""
        rows_list = []
        for _, row in df.iterrows():
            rows_list.append(self.transform_row(row))
        if not rows_list:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        return np.array(rows_list, dtype=np.float32)

    def get_obs_at(self, idx: int) -> list[float]:
        """Retrieves fast pre-computed observation vector for row index."""
        if 0 <= idx < len(self._cached_obs_matrix):
            return self._cached_obs_matrix[idx].tolist()
        return [0.0] * self.vector_dim


class CandidateRule(BaseModel):
    """Business rule candidate extracted from unstructured document policies."""
    name: str
    plain_text: str
    when_expr: str
    value: float
    importance: str  # "Must Do (+10)" | "High Priority (+5)" | "Avoid (-5)" | "Strict Ban (-10)"
    category: str = "compliance"

    @property
    def description(self) -> str:
        return self.plain_text

    @property
    def ast_expression(self) -> str:
        return self.when_expr

    @property
    def suggested_weight(self) -> float:
        return float(self.value)

    @property
    def rule_id(self) -> str:
        return self.name

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        d["description"] = self.plain_text
        return d


class DocumentRuleExtractor:
    """Extracts candidate business rules and AST DSL templates from policy documents."""

    @classmethod
    def extract_rules(cls, text_content: str, known_columns: list[str] | None = None) -> list[CandidateRule]:
        rules: list[CandidateRule] = []
        lines = [line.strip() for line in text_content.splitlines() if line.strip()]

        col_set = set(known_columns or [])

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Rule Pattern 1: Dollar / Currency threshold rules (e.g. "Invoices over $5,000 require VP approval" or "Flag > 5000")
            dollar_match = re.search(r"(\$|usd|\bamount\b|\bvalue\b)[\s:]*([0-9,]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
            if dollar_match:
                try:
                    num_str = dollar_match.group(2).replace(",", "")
                    threshold = float(num_str)
                    if threshold > 0:
                        when_target = "obs.dollar_amount" if "dollar_amount" in col_set else "obs.max_dollar_value"
                        if "amount" in col_set:
                            when_target = "obs.amount"

                        rules.append(CandidateRule(
                            name=f"flag_high_value_threshold_{int(threshold)}",
                            plain_text=f"Flag transactions exceeding ${threshold:,.0f} for mandatory review",
                            when_expr=f"{when_target} > {threshold} and action.label == 'flag'",
                            value=8.0,
                            importance="Must Do (+10)",
                            category="financial",
                        ))
                except ValueError:
                    pass

            # Rule Pattern 2: Missing PO / Unverified Purchase Order
            if "missing po" in line_lower or "without po" in line_lower or "no purchase order" in line_lower or "po verification" in line_lower:
                when_clause = "obs.has_po == False" if "has_po" in col_set else "obs.risk_keyword_count > 0"
                rules.append(CandidateRule(
                    name="reject_missing_po",
                    plain_text="Strictly penalize approving transactions without an authorized PO",
                    when_expr=f"{when_clause} and action.label == 'approve'",
                    value=-15.0,
                    importance="Strict Ban (-10)",
                    category="compliance",
                ))

            # Rule Pattern 3: Escalations for angry / high-risk cases
            if "escalate" in line_lower or "vip" in line_lower or "high risk" in line_lower or "critical" in line_lower:
                rules.append(CandidateRule(
                    name="escalate_critical_risk",
                    plain_text="Escalate high-risk and urgent cases promptly",
                    when_expr="obs.risk_keyword_count >= 2 and action.label == 'escalate'",
                    value=6.0,
                    importance="High Priority (+5)",
                    category="urgency",
                ))

        # Default fallback rule if no pattern matched
        if not rules:
            rules.append(CandidateRule(
                name="correct_action_match",
                plain_text="Reward accurate classification matching business criteria",
                when_expr="truth.has_label == True and action.label == truth.label",
                value=5.0,
                importance="High Priority (+5)",
                category="accuracy",
            ))

        return rules
