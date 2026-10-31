"""Redaction of learner traces before they leave the process boundary.

Round evidence has to reach the tuner, and the tuner is a hosted LLM. Observation vectors are
derived from customer rows, so shipping them verbatim would quietly undo the zero-exfiltration
guarantee that the sandbox exists to protect. What the tuner actually needs is the decision
shape - which actions were taken, which reward terms fired, how they scored - not the raw
feature values. Raw vectors are replaced by per-dimension ranges.
"""

from __future__ import annotations

from typing import Any, Iterable

MAX_REDACTED_EPISODES = 15
MAX_OBS_DIMS = 32
DECIMALS = 4


def _obs_summary(vectors: list[list[float]]) -> dict[str, Any]:
    """Per-dimension range instead of the values themselves."""
    if not vectors:
        return {"dims": 0}
    width = min(MAX_OBS_DIMS, min(len(v) for v in vectors))
    summary: dict[str, Any] = {"dims": width, "truncated_dims": max(0, len(vectors[0]) - width)}
    for dim in range(width):
        column = [float(v[dim]) for v in vectors if len(v) > dim]
        if not column:
            continue
        summary[f"d{dim}"] = [
            round(min(column), DECIMALS),
            round(max(column), DECIMALS),
            round(sum(column) / len(column), DECIMALS),
        ]
    return summary


def redact_episodes(
    episodes: Iterable[dict[str, Any]], *, max_episodes: int = MAX_REDACTED_EPISODES
) -> list[dict[str, Any]]:
    """Keeps diagnostic fields, drops raw observations."""
    redacted: list[dict[str, Any]] = []
    for episode in list(episodes)[-max_episodes:]:
        steps_out: list[dict[str, Any]] = []
        for step in episode.get("steps", []) or []:
            steps_out.append(
                {
                    "step": step.get("step"),
                    "action": step.get("action"),
                    "reward": round(float(step.get("reward", 0.0) or 0.0), DECIMALS),
                    "reward_breakdown": {
                        str(k): round(float(v), DECIMALS)
                        for k, v in (step.get("reward_breakdown") or {}).items()
                    },
                    "is_success": bool(step.get("is_success", False)),
                    "is_correct": step.get("is_correct"),
                }
            )
        redacted.append(
            {
                "episode_id": episode.get("episode_id"),
                "round": episode.get("round"),
                "metric_value": episode.get("metric_value"),
                "is_success": bool(episode.get("is_success", False)),
                "total_reward": round(float(episode.get("total_reward", 0.0) or 0.0), DECIMALS),
                "steps": steps_out,
                "obs_summary": _obs_summary(
                    [step.get("obs", []) for step in episode.get("steps", []) or [] if step.get("obs")]
                ),
            }
        )
    return redacted
