"""Dataset-agnostic feature derivation for supervised tabular environments.

Synthesized tabular environments must not trust observation names proposed by an LLM: those
names rarely match the uploaded columns, which silently produces all-zero observations. Instead
the observation vector is derived deterministically from the dataset itself:

* numeric columns  -> min-max normalised to ``[0, 1]``
* boolean columns  -> ``0/1``
* categorical/text columns with few distinct values -> one-hot over the most frequent values
* free-text / identifier columns -> skipped (no learnable signal without embeddings)

The derivation is embedded into the generated ``env_module.py`` as a small list of ops so the
environment can transform raw records at runtime, including records supplied after a restart or
inside an exported bundle.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

MAX_CATEGORIES = 12
MAX_FEATURES = 64
MAX_ROWS = 120

_STOP_TOKENS = {"to", "the", "a", "an", "of", "for", "and", "or", "agent", "route", "action"}


def normalize_token(text: Any) -> str:
    """Lower-cases and collapses any non-alphanumeric run into a single underscore."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _tokens(text: Any) -> set[str]:
    return {t for t in normalize_token(text).split("_") if t and t not in _STOP_TOKENS}


def is_numeric_like(value: Any) -> bool:
    """True when ``value`` can be interpreted as a finite number."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    try:
        return math.isfinite(float(str(value)))
    except (TypeError, ValueError):
        return False


def label_keys(value: Any) -> list[str]:
    """Builds the lookup keys used by generated environments for a raw label value."""
    keys = [str(value)]
    if is_numeric_like(value):
        num = float(str(value)) if not isinstance(value, (int, float)) else float(value)
        if num.is_integer():
            keys.append(str(int(num)))
        keys.append(str(num))
    stripped = str(value).strip()
    if stripped != keys[0]:
        keys.append(stripped)
    return list(dict.fromkeys(keys))


def infer_label_map(values: Iterable[Any], action_labels: list[str]) -> dict[str, int]:
    """Maps each distinct target value onto an action index using name similarity.

    Exact (normalised) matches win, then unique token overlap. Values that cannot be tied to
    an action are dropped, which keeps the policy from learning an arbitrary label ordering.
    """
    labels = [str(a) for a in action_labels]
    if not labels:
        return {}
    norm_actions = [normalize_token(a) for a in labels]
    action_tokens = [_tokens(a) for a in labels]

    mapping: dict[str, int] = {}
    for raw in values:
        token = normalize_token(raw)
        idx: int | None = None

        if token in norm_actions:
            idx = norm_actions.index(token)
        elif is_numeric_like(raw):
            as_int = int(float(str(raw)))
            if 0 <= as_int < len(labels):
                idx = as_int
        else:
            best_score, best_idx = 0.0, None
            for i, toks in enumerate(action_tokens):
                probe = _tokens(raw)
                if not probe or not toks:
                    continue
                overlap = len(probe & toks) / len(probe | toks)
                if overlap > best_score:
                    best_score, best_idx = overlap, i
            if best_idx is not None and best_score >= 0.34:
                idx = best_idx

        if idx is None:
            continue
        for key in label_keys(raw):
            mapping[key] = idx
    return mapping


def sanitize_name(name: str) -> str:
    """Produces a readable, unique-safe observation name."""
    cleaned = re.sub(r"\s+", "_", str(name).strip())
    return cleaned or "feature"


def derive_feature_ops(
    df: Any,
    label_column: str,
    max_features: int = MAX_FEATURES,
    max_categories: int = MAX_CATEGORIES,
) -> tuple[list[tuple[str, str, Any, Any]], list[str]]:
    """Derives ``(feature_ops, feature_names)`` from ``df`` excluding the label column.

    ``feature_ops`` entries are ``("num", column, low, high)`` or ``("cat", column, value, None)``.
    """
    import pandas as pd  # local import keeps module import cheap

    ops: list[tuple[str, str, Any, Any]] = []
    names: list[str] = []
    seen: set[str] = set()

    for col in df.columns:
        if str(col) == str(label_column):
            continue
        series = df[col]
        non_null = series.dropna()
        if len(non_null) == 0:
            continue
        try:
            nunique = int(non_null.nunique())
        except Exception:
            continue
        if nunique < 2:
            continue

        is_bool = pd.api.types.is_bool_dtype(series)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_bool:
            ops.append(("num", str(col), 0.0, 1.0))
            name = sanitize_name(col)
            if name not in seen:
                seen.add(name)
                names.append(name)
        elif is_numeric:
            try:
                low = float(non_null.min())
                high = float(non_null.max())
            except (TypeError, ValueError):
                continue
            if not math.isfinite(low) or not math.isfinite(high) or high <= low:
                continue
            ops.append(("num", str(col), low, high))
            name = sanitize_name(col)
            if name not in seen:
                seen.add(name)
                names.append(name)
        else:
            if nunique > max_categories:
                continue
            counts = non_null.astype(str).value_counts()
            for category in list(counts.index)[:max_categories]:
                ops.append(("cat", str(col), str(category), None))
                name = sanitize_name(f"{col}={category}")
                if name not in seen:
                    seen.add(name)
                    names.append(name)

        if len(ops) >= max_features:
            break

    return ops[:max_features], names[:max_features]


def select_usable_rows(df: Any, label_column: str, label_map: dict[str, int]) -> Any:
    """Keeps only rows whose target value maps onto a declared action."""
    if not label_map:
        return df.iloc[0:0]
    mask = df[label_column].map(lambda v: str(v) in label_map)
    return df[mask]


def records_from_frame(df: Any, limit: int = MAX_ROWS) -> list[dict[str, Any]]:
    """Serialises rows into JSON-safe dicts (raw values, so labels and arenas stay meaningful)."""
    records: list[dict[str, Any]] = []
    for _, row in df.head(limit).iterrows():
        rec: dict[str, Any] = {}
        for col, value in row.items():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except (ValueError, TypeError):
                    value = str(value)
            if isinstance(value, float) and not math.isfinite(value):
                value = 0.0
            rec[str(col)] = value
        records.append(rec)
    return records
