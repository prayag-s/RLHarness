"""Wording and classification for thread entries — one place, so the transcript stays coherent.

Every surface that used to narrate separately (activity feed, telemetry panel, coach cards) now
speaks through these functions. They are pure: given an event or a studio response, they return the
entries to append. Nothing here reads or writes the database.
"""

from __future__ import annotations

from typing import Any


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_of(event: dict[str, Any]) -> int:
    return int(event.get("round") or event.get("round_num") or 1)


#: Events that are live frames or plumbing, never transcript entries.
SILENT_EVENTS = {
    "episode_step",
    "pong",
    "drift",
    "job_completed",
    "sandbox_bypassed",
}


def entry_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Maps one websocket/studio event onto a single thread entry, or ``None`` to stay silent."""
    name = str(event.get("event") or "")
    if not name or name in SILENT_EVENTS:
        return None

    if name == "job_stage":
        stage = str(event.get("stage") or "working").replace("_", " ")
        return {
            "actor": "studio",
            "kind": "stage",
            "status": "running",
            "title": f"Studio · {stage}",
            "body": str(event.get("detail") or ""),
        }

    if name == "job_queued":
        kind = str(event.get("kind") or "training")
        if kind != "training":
            return None
        return {
            "actor": "trainer",
            "kind": "round",
            "status": "running",
            "title": "Training queued",
            "body": str(event.get("detail") or "The trainer will take the next turn."),
        }

    if name == "round_started":
        r = _round_of(event)
        return {
            "actor": "trainer",
            "kind": "round",
            "status": "running",
            "title": f"Round {r} started",
            "body": str(event.get("message") or "Autonomous training burst executing."),
            "anchor": {"round": r},
        }

    if name == "round_evaluated":
        r = _round_of(event)
        metric = _num(event.get("metric", event.get("current_metric")), 0.0)
        target = _num(event.get("target", event.get("target_threshold")), 85.0)
        achieved = metric >= target
        return {
            "actor": "trainer",
            "kind": "round",
            "status": "success" if achieved else "info",
            "title": f"Round {r} evaluated · {metric:.1f}%",
            "body": str(event.get("message") or f"Target is {target:.1f}%."),
            "anchor": {"round": r, "metric": metric},
            "data": {"round": r, "metric": metric, "target": target},
        }

    if name == "auto_tuned":
        r = _round_of(event)
        diagnosis = str(
            event.get("diagnosis") or (event.get("tuning_result") or {}).get("diagnosis") or "Reward function updated."
        )
        changes = str(
            event.get("changes_summary") or (event.get("tuning_result") or {}).get("changes_summary") or ""
        )
        return {
            "actor": "trainer",
            "kind": "tune",
            "status": "warning",
            "title": f"Auto-tuner · Round {r}",
            "body": f"{diagnosis} {changes}".strip(),
            "anchor": {"round": r},
        }

    if name in {"warmstart_completed", "critic_warmup_completed"}:
        label = "Behavioural warm-start complete" if name == "warmstart_completed" else "Value critic calibrated"
        return {
            "actor": "trainer",
            "kind": "round",
            "status": "info",
            "title": label,
            "body": str(event.get("message") or ""),
        }

    if name == "target_achieved":
        metric = _num(event.get("metric", event.get("current_metric")), 0.0)
        return {
            "actor": "trainer",
            "kind": "result",
            "status": "success",
            "title": f"Target achieved · {metric:.1f}%",
            "body": str(event.get("message") or "The agent cleared the target metric."),
            "anchor": {"metric": metric},
        }

    if name == "budget_exhausted":
        metric = _num(event.get("metric", event.get("current_metric")), 0.0)
        return {
            "actor": "trainer",
            "kind": "result",
            "status": "info",
            "title": f"Budget exhausted · {metric:.1f}%",
            "body": str(event.get("message") or "All auto-tune rounds are complete."),
            "anchor": {"metric": metric},
        }

    if name == "arena_updated":
        kind = str(event.get("arena_kind") or "bespoke")
        return {
            "actor": "studio",
            "kind": "arena",
            "status": "warning" if kind == "fallback" else "success",
            "title": "Arena preview" if kind == "preview" else "Arena updated",
            "body": str(
                event.get("message")
                or (
                    "Deterministic preview rendered while the bespoke canvas generates."
                    if kind == "preview"
                    else "Arena layout and styling hot-reloaded."
                )
            ),
        }

    if name == "scorecard_ready":
        return {
            "actor": "gatekeeper",
            "kind": "scorecard",
            "status": "success",
            "title": "Held-out scorecard",
            "body": str(event.get("message") or "Evaluation finished on unseen rows."),
            "actions": [{"label": "Review evidence", "kind": "open_trust"}],
        }

    if name == "drift_warning":
        drift = event.get("drift") or {}
        return {
            "actor": "gatekeeper",
            "kind": "verdict",
            "status": "warning",
            "title": "Drift warning",
            "body": str(drift.get("recommendation") or event.get("message") or "Live inputs moved away from the training baseline."),
            "actions": [{"label": "Review evidence", "kind": "open_trust"}],
        }

    if name in {"training_error", "error"}:
        return {
            "actor": "system",
            "kind": "error",
            "status": "warning",
            "title": "Training error",
            "body": str(event.get("message") or "An error occurred during autonomous training."),
        }

    if name == "job_failed":
        return {
            "actor": "system",
            "kind": "error",
            "status": "warning",
            "title": "Turn failed",
            "body": str(event.get("message") or "The job did not finish."),
        }

    if name == "job_cancelled":
        return {
            "actor": "system",
            "kind": "result",
            "status": "info",
            "title": "Stopped",
            "body": "You stopped this turn.",
        }

    if name == "eval_corpus_seeded":
        return {
            "actor": "gatekeeper",
            "kind": "scorecard",
            "status": "info",
            "title": "Held-out corpus seeded",
            "body": str(event.get("message") or ""),
        }

    return None


def coach_entries(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One entry per open coach finding, carrying the item so the card can act on it."""
    if not report:
        return []
    entries: list[dict[str, Any]] = []
    for item in report.get("items") or []:
        if item.get("status") == "resolved":
            continue
        severity = str(item.get("severity") or "tip")
        status = {"blocker": "blocked", "warning": "warning", "tip": "info"}.get(severity, "info")
        actions = [
            {
                "label": str(action.get("label") or "Apply"),
                "kind": str(action.get("kind") or "send_message"),
                "payload": action.get("payload") or {},
            }
            for action in item.get("actions") or []
        ]
        entries.append(
            {
                "actor": "coach",
                "kind": "finding",
                "status": status,
                "title": str(item.get("title") or "Coach finding"),
                "body": str(item.get("why") or ""),
                "actions": actions,
                "data": {
                    "rule_id": item.get("rule_id"),
                    "severity": severity,
                    "stage": item.get("stage"),
                    "item": item,
                },
            }
        )
    return entries


def entries_from_studio_response(response: Any) -> list[dict[str, Any]]:
    """The studio's own contribution to a turn: its reply, questions, and coach findings."""
    entries: list[dict[str, Any]] = []

    reply = getattr(response, "assistant_message", "") or getattr(response, "message", "")
    if reply:
        entries.append(
            {
                "actor": "studio",
                "kind": "answer",
                "status": "success",
                "title": "Studio",
                "body": str(reply),
            }
        )

    questions = getattr(response, "questions", None) or []
    if questions:
        body = "\n".join(
            f"{i + 1}. {getattr(q, 'question', None) or (q.get('question') if isinstance(q, dict) else '')}"
            for i, q in enumerate(questions)
        )
        entries.append(
            {
                "actor": "studio",
                "kind": "question",
                "status": "success",
                "title": "Studio needs a decision",
                "body": body,
                "data": {
                    "questions": [
                        q.model_dump() if hasattr(q, "model_dump") else q for q in questions
                    ]
                },
            }
        )

    guidance = getattr(response, "guidance", None)
    entries.extend(coach_entries(guidance))
    return entries


def settle_state(response: Any) -> str:
    """What the conversation's turn state becomes once a studio response is fully handled.

    A blocking coach finding or an outstanding question is a decision only the user can make; a
    training run keeps the turn open, because the trainer now owns it.
    """
    guidance = getattr(response, "guidance", None) or {}
    items = guidance.get("items") or []
    if any(i.get("severity") == "blocker" and i.get("status") != "acknowledged" for i in items):
        return "blocked"
    if getattr(response, "questions", None):
        return "decision_required"
    return "idle"
