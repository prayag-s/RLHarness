"""Journey Coach service: gather facts, run rules, optionally let the model prioritise.

The service is the only place that decides when a model is spent. Rules always run and are free.
The judge runs once per artefact change, cached by content hash, capped per conversation, and
fails open: a provider outage downgrades the coach to rules-only instead of breaking the studio.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

from rl_ide.coach import judge as judge_module
from rl_ide.coach.facts import (
    dataset_facts,
    export_facts,
    objective_facts,
    parameter_facts,
    preview_facts,
    result_facts,
)
from rl_ide.coach.models import (
    SEVERITY_ORDER,
    CoachBlocked,
    CoachReport,
    CoachStage,
    GuidanceItem,
    Severity,
)
from rl_ide.coach.rules import (
    dataset_items,
    export_items,
    objective_items,
    parameter_items,
    preview_items,
    result_items,
)
from rl_ide.coach.store import CoachStore, get_coach_store

DEFAULT_MAX_LLM_CALLS = 12
DEFAULT_BLOCKER_MODE = "hard"

_STAGE_SEQUENCE = {
    CoachStage.OBJECTIVE: 0,
    CoachStage.DATA: 1,
    CoachStage.PARAMETERS: 2,
    CoachStage.PREVIEW: 3,
    CoachStage.RESULTS: 4,
    CoachStage.EXPORT: 5,
}

_LLM_STAGES = {
    CoachStage.OBJECTIVE,
    CoachStage.DATA,
    CoachStage.PREVIEW,
    CoachStage.RESULTS,
}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "with",
    "this", "that", "it", "you", "your", "be", "will", "was", "were", "has", "have",
}


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no", "")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def blockers_mode() -> str:
    """``hard`` blocks execution until a blocker is fixed; ``advisory`` only warns."""
    raw = os.environ.get("RL_IDE_COACH_BLOCKERS", DEFAULT_BLOCKER_MODE).strip().lower()
    return raw if raw in ("hard", "advisory") else DEFAULT_BLOCKER_MODE


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def _dedupe(items: list[GuidanceItem]) -> list[GuidanceItem]:
    """Rules win. An LLM restatement of a rule is noise, not a second opinion."""
    seen_ids: set[str] = set()
    kept: list[GuidanceItem] = []
    rule_token_sets: list[set[str]] = []

    for item in items:
        if item.rule_id in seen_ids:
            continue
        tokens = _tokens(item.title)
        if item.source == "llm":
            duplicate = False
            for rule_tokens in rule_token_sets:
                if not tokens or not rule_tokens:
                    continue
                overlap = len(tokens & rule_tokens) / max(1, len(tokens | rule_tokens))
                if overlap >= 0.5:
                    duplicate = True
                    break
            if duplicate:
                continue
        seen_ids.add(item.rule_id)
        kept.append(item)
        if item.source == "rule":
            rule_token_sets.append(tokens)
    return kept


class JourneyCoach:
    def __init__(
        self,
        client: Any = None,
        store: CoachStore | None = None,
        use_llm: bool | None = None,
    ):
        self._client = client
        self.store = store or get_coach_store()
        self._use_llm = use_llm

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        from rl_ide.llm.client import get_active_llm_client

        return get_active_llm_client()

    @property
    def hard_mode(self) -> bool:
        return blockers_mode() == "hard"

    def llm_enabled(self) -> bool:
        if self._use_llm is not None:
            return self._use_llm
        return _env_flag("RL_IDE_COACH_LLM", True)

    def _max_llm_calls(self) -> int:
        return _env_int("RL_IDE_COACH_MAX_LLM_CALLS", DEFAULT_MAX_LLM_CALLS)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _not_ready(
        self, stage: CoachStage, conversation_id: str | None, message: str
    ) -> CoachReport:
        """A checkpoint with no artefact yet is a tip, not a blocker, and is not persisted."""
        return CoachReport(
            stage=stage,
            conversation_id=conversation_id,
            items=[
                GuidanceItem(
                    rule_id=f"{stage.value}.not_ready",
                    stage=stage,
                    severity=Severity.TIP,
                    title="Nothing to evaluate yet",
                    why=message,
                    evidence={},
                )
            ],
        )

    def evaluate(
        self,
        stage: CoachStage,
        *,
        conversation_id: str | None = None,
        prompt: str = "",
        blueprint: Any = None,
        dataset: Any = None,
        params: dict[str, Any] | None = None,
        scorecard: dict[str, Any] | None = None,
        conversation: dict[str, Any] | None = None,
        arena_html: str | None = None,
        env_code: str | None = None,
        deployment: dict[str, Any] | None = None,
        use_llm: bool | None = None,
    ) -> CoachReport:
        if stage == CoachStage.OBJECTIVE and not (prompt or "").strip() and blueprint is None:
            return self._not_ready(
                stage, conversation_id, "Describe what the agent should decide, then evaluate again."
            )
        if stage == CoachStage.DATA and dataset is None:
            return self._not_ready(
                stage, conversation_id, "Attach a dataset to check its target, balance and leakage."
            )
        if stage == CoachStage.PARAMETERS and blueprint is None:
            return self._not_ready(
                stage, conversation_id, "Build a blueprint before checking the training budget."
            )

        facts: dict[str, Any]
        items: list[GuidanceItem]
        llm_prompt = prompt or None

        if stage == CoachStage.OBJECTIVE:
            objective = objective_facts(prompt, blueprint, dataset)
            facts = objective.to_dict()
            items = objective_items(objective)
        elif stage == CoachStage.DATA:
            ds_facts = dataset_facts(dataset)
            supervised = False
            if blueprint is not None or prompt:
                supervised = objective_facts(prompt, blueprint, dataset).supervised_metric
            facts = ds_facts.to_dict()
            facts["supervised_objective"] = supervised
            items = dataset_items(ds_facts, supervised_objective=supervised)
        elif stage == CoachStage.PARAMETERS:
            param_facts = parameter_facts(blueprint, params, dataset, prompt)
            facts = param_facts.to_dict()
            items = parameter_items(param_facts)
        elif stage == CoachStage.PREVIEW:
            preview = preview_facts(blueprint, arena_html, env_code)
            interactive_expected = False
            if isinstance(blueprint, dict):
                interactive_expected = bool(blueprint.get("interactive_mode_description"))
            elif blueprint is not None:
                interactive_expected = bool(
                    getattr(blueprint, "interactive_mode_description", "")
                )
            facts = preview.to_dict()
            facts["interactive_expected"] = interactive_expected
            items = preview_items(preview, interactive_expected=interactive_expected)
        elif stage == CoachStage.RESULTS:
            results = result_facts(scorecard, conversation)
            facts = results.to_dict()
            items = result_items(results)
        elif stage == CoachStage.EXPORT:
            export = export_facts(scorecard, deployment)
            facts = export.to_dict()
            items = export_items(export)
        else:  # pragma: no cover - the enum is exhaustive
            raise ValueError(f"Unknown coach stage: {stage}")

        llm_items, llm_used, llm_reason = self._judge(
            stage,
            facts,
            conversation_id=conversation_id,
            prompt=llm_prompt,
            use_llm=use_llm,
        )
        items = _dedupe([*items, *llm_items])
        items.sort(key=lambda i: (SEVERITY_ORDER[i.severity], _STAGE_SEQUENCE[i.stage]))

        if conversation_id:
            statuses = self.store.record(
                conversation_id, items, stages={stage.value}
            )
            for item in items:
                item.status = statuses.get(item.fingerprint(), "open")  # type: ignore[assignment]

        return CoachReport(
            stage=stage,
            items=items,
            conversation_id=conversation_id,
            llm_used=llm_used,
            llm_skipped_reason=llm_reason,
        )

    def evaluate_preflight(
        self,
        *,
        conversation_id: str | None = None,
        prompt: str = "",
        blueprint: Any = None,
        dataset: Any = None,
        params: dict[str, Any] | None = None,
        use_llm: bool | None = None,
    ) -> CoachReport:
        """Evaluates objective, data and parameters together before a run is paid for."""
        if not (prompt or "").strip() and blueprint is None:
            return self._not_ready(
                CoachStage.OBJECTIVE,
                conversation_id,
                "Describe what the agent should decide before a build is evaluated.",
            )
        objective = objective_facts(prompt, blueprint, dataset)
        items: list[GuidanceItem] = list(objective_items(objective))

        if dataset is not None:
            ds_facts = dataset_facts(dataset)
            items.extend(
                dataset_items(ds_facts, supervised_objective=objective.supervised_metric)
            )

        param_facts = parameter_facts(blueprint, params, dataset, prompt)
        items.extend(parameter_items(param_facts))

        llm_items, llm_used, llm_reason = self._judge(
            CoachStage.OBJECTIVE,
            objective.to_dict(),
            conversation_id=conversation_id,
            prompt=prompt or None,
            use_llm=use_llm,
        )
        items = _dedupe([*items, *llm_items])
        items.sort(key=lambda i: (SEVERITY_ORDER[i.severity], _STAGE_SEQUENCE[i.stage]))

        stages = {CoachStage.OBJECTIVE.value, CoachStage.PARAMETERS.value}
        if dataset is not None:
            stages.add(CoachStage.DATA.value)
        if conversation_id:
            statuses = self.store.record(conversation_id, items, stages=stages)
            for item in items:
                item.status = statuses.get(item.fingerprint(), "open")  # type: ignore[assignment]

        return CoachReport(
            stage=None,
            items=items,
            conversation_id=conversation_id,
            llm_used=llm_used,
            llm_skipped_reason=llm_reason,
        )

    # ------------------------------------------------------------------
    # Enforcement and acknowledgement
    # ------------------------------------------------------------------

    def enforce(self, report: CoachReport) -> CoachReport:
        if self.hard_mode and report.blockers:
            raise CoachBlocked(report)
        return report

    def acknowledge(
        self,
        conversation_id: str,
        rule_id: str,
        *,
        fingerprint: str | None = None,
        note: str = "",
    ) -> list[dict[str, Any]]:
        matches = [
            item
            for item in self.store.items(conversation_id)
            if item["rule_id"] == rule_id
            and (fingerprint is None or item["fingerprint"] == fingerprint)
            and item["status"] == "open"
        ]
        if not matches:
            raise KeyError(f"No open coach finding '{rule_id}' for this conversation.")
        if self.hard_mode and any(item["severity"] == "blocker" for item in matches):
            raise PermissionError("Blockers must be fixed, not acknowledged.")
        return self.store.acknowledge(
            conversation_id, rule_id, fingerprint=fingerprint, note=note
        )

    # ------------------------------------------------------------------
    # Judge plumbing
    # ------------------------------------------------------------------

    def _judge(
        self,
        stage: CoachStage,
        facts: dict[str, Any],
        *,
        conversation_id: str | None,
        prompt: str | None,
        use_llm: bool | None,
    ) -> tuple[list[GuidanceItem], bool, str | None]:
        enabled = self.llm_enabled() if use_llm is None else use_llm
        if stage not in _LLM_STAGES:
            return [], False, "stage is rule-only"
        if not enabled:
            return [], False, "disabled"

        key = self.store.cache_key(stage.value, facts, prompt)
        cached = self.store.cache_get(key)
        if cached is not None:
            try:
                return [GuidanceItem.model_validate(item) for item in cached], True, None
            except Exception:
                pass  # a corrupt cache entry must never break a checkpoint

        if conversation_id and not self.store.try_reserve_llm_call(
            conversation_id, self._max_llm_calls()
        ):
            return [], False, "coach LLM budget exhausted for this conversation"

        try:
            items = judge_module.judge(stage, facts, self.client, prompt=prompt)
        except Exception as exc:  # noqa: BLE001 - fail open by design
            return [], False, f"{type(exc).__name__}: {exc}"

        self.store.cache_put(key, items)
        return items, True, None


_COACH: JourneyCoach | None = None


def get_coach() -> JourneyCoach:
    global _COACH
    if _COACH is None:
        _COACH = JourneyCoach()
    return _COACH
