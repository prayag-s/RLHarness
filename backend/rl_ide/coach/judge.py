"""The LLM half of the coach: prioritise and narrate, never measure.

The judge is intentionally narrow. It sees a small facts dict produced by ``facts.py`` and may
only return findings whose numbers already appear in that dict. Anything else is discarded before
it reaches the user, so a hallucinated statistic cannot become product copy.

It also never returns a blocker. A structural blocker (empty data, no actions, zero budget) is a
fact, not an opinion; letting a model hard-block a run would make the studio fragile.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from pydantic import ValidationError

from rl_ide.llm.prompts import with_english_rule
from rl_ide.coach.models import (
    CoachStage,
    GuidanceAction,
    GuidanceItem,
    Severity,
)

COACH_JUDGE_SYSTEM_PROMPT = with_english_rule("""You are the Journey Coach in an RL agent IDE. You review one
artefact at one stage of a user's journey and tell them the single most useful thing to change.

You are given FACTS computed by deterministic code. They are the only source of numbers, column
names, metrics and actions. Never invent one.

Rules:
- Return at most 3 findings, most important first. Return none if the artefact is sound.
- Only report a finding if it would change what the user does next.
- Never use severity "blocker"; structural blockers are computed outside you.
- Every finding must carry exactly one concrete action from the allowed kinds.
- Keep the title under 90 characters and the reason under 320 characters.
- Plain language, no markdown, no scolding.

Allowed action kinds and payloads:
- {"kind":"apply_params","label":"...","payload":{"max_auto_tune_rounds":3,"episodes_per_round":200,"target_metric_threshold":85}}
- {"kind":"use_objective","label":"...","payload":{"objective":"<a complete rewritten objective>"}}
- {"kind":"send_message","label":"...","payload":{"message":"<an instruction the user can send>"}}
- {"kind":"open_guide","label":"...","payload":{"section":"5.1"}}
- {"kind":"run_episodes","label":"...","payload":{"episodes_per_round":200}}

Respond with strict JSON only:
{"items":[{"rule_id":"llm.<slug>","severity":"warning|tip","title":"...","why":"...","action":{...}}]}
""")

_STAGE_BRIEF = {
    CoachStage.OBJECTIVE: (
        "Judge whether this objective is complete, measurable and internally consistent. "
        "Check that the stated success metric can be observed, that the prohibition (if any) has a "
        "reward or constraint behind it, and that the observation set can support the decision. "
        "If a rewrite would materially help, return one use_objective action."
    ),
    CoachStage.DATA: (
        "Judge whether this dataset can answer the stated objective. Check the target, the "
        "distribution, leakage and whether the recorded outcome matches what the agent is asked "
        "to decide."
    ),
    CoachStage.PREVIEW: (
        "Judge whether a human could tell success from failure by looking at this arena. Name "
        "the one missing readout or control that would make the preview honest."
    ),
    CoachStage.RESULTS: (
        "Explain what the evidence means and what to do next. Use only the facts: held-out "
        "status, interval width, baselines, corpus regressions, Crossfire reasons. Do not "
        "congratulate and do not invent a cause."
    ),
    CoachStage.EXPORT: (
        "Judge whether this bundle can be audited and deployed safely. The scorecard, Crossfire "
        "verdict and deployment parity are the only evidence."
    ),
}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_SMALL_COUNTS = {"0", "1", "2", "3", "4", "5"}


def facts_digest(facts: dict[str, Any]) -> str:
    payload = json.dumps(facts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _allowed_numbers(facts: dict[str, Any]) -> set[str]:
    text = json.dumps(facts, default=str)
    allowed = set(_NUMBER.findall(text))
    # Percentages are stored as 85.0 while prose says 85%; accept both spellings.
    for value in list(allowed):
        try:
            number = float(value)
        except ValueError:
            continue
        allowed.add(str(int(number)) if number.is_integer() else f"{number:g}")
        allowed.add(f"{number:g}")
    allowed |= _SMALL_COUNTS
    return allowed


def _unsupported_number(text: str, allowed: set[str]) -> str | None:
    for raw in _NUMBER.findall(text or ""):
        normalized = raw.lstrip("+")
        if normalized in allowed:
            continue
        try:
            number = float(normalized)
        except ValueError:
            return raw
        if number.is_integer():
            normalized = str(int(number))
        else:
            normalized = f"{number:g}"
        if normalized not in allowed:
            return raw
    return None


def _parse_action(raw: Any) -> GuidanceAction | None:
    if not isinstance(raw, dict):
        return None
    try:
        return GuidanceAction.model_validate(raw)
    except ValidationError:
        return None


def _clean_items(
    raw_items: Iterable[Any],
    *,
    stage: CoachStage,
    facts: dict[str, Any],
) -> list[GuidanceItem]:
    allowed = _allowed_numbers(facts)
    digest = facts_digest(facts)
    items: list[GuidanceItem] = []
    seen: set[str] = set()

    for index, raw in enumerate(list(raw_items)[:3]):
        if not isinstance(raw, dict):
            continue
        title = " ".join(str(raw.get("title") or "").split())[:120]
        why = " ".join(str(raw.get("why") or "").split())[:400]
        if len(title) < 8 or len(why) < 20:
            continue
        if _unsupported_number(title, allowed) or _unsupported_number(why, allowed):
            continue

        action = _parse_action(raw.get("action"))
        if action is None:
            continue

        severity_raw = str(raw.get("severity") or "tip").lower()
        # "blocker" is downgraded rather than dropped: the finding may still be useful, but a
        # model must never be the thing that stops a run.
        severity = Severity.WARNING if severity_raw in ("warning", "blocker") else Severity.TIP

        rule_id = str(raw.get("rule_id") or f"llm.finding_{index + 1}").strip()
        if not rule_id.startswith("llm."):
            rule_id = f"llm.{re.sub(r'[^a-z0-9_]+', '_', rule_id.lower()).strip('_') or 'finding'}"
        if rule_id in seen:
            continue
        seen.add(rule_id)

        items.append(
            GuidanceItem(
                rule_id=rule_id,
                stage=stage,
                severity=severity,
                title=title,
                why=why,
                evidence={"facts_digest": digest},
                actions=[action],
                source="llm",
            )
        )
    return items


def judge(
    stage: CoachStage,
    facts: dict[str, Any],
    client: Any,
    *,
    prompt: str | None = None,
) -> list[GuidanceItem]:
    """Calls the model once and validates every finding against the facts.

    Any exception is the caller's to handle: the service treats a failed judge as "rules only",
    never as a failed checkpoint.
    """
    brief = _STAGE_BRIEF.get(stage, "Review this artefact.")
    user_prompt = (
        f"STAGE: {stage.value}\n"
        f"WHAT TO JUDGE: {brief}\n"
        f"FACTS:\n{json.dumps(facts, indent=2, default=str)}\n"
    )
    if prompt:
        user_prompt += f"\nORIGINAL OBJECTIVE:\n{prompt[:2000]}\n"
    user_prompt += "\nReturn strict JSON only."

    payload = client.generate_json(
        user_prompt,
        system_prompt=COACH_JUDGE_SYSTEM_PROMPT,
        timeout=45.0,
    )
    if not isinstance(payload, dict):
        return []
    return _clean_items(payload.get("items") or [], stage=stage, facts=facts)
