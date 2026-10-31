"""Golden evaluation corpus: the behavioural assertions a policy must keep satisfying.

Every human judgement collected by the studio - a seeded validation row, a locked interactive
step, a plain-English critique, a rejected frontier action - becomes an executable case. A
retrained policy is measured against the whole accumulated set, so a reward edit that fixes the
critiqued behaviour while breaking an earlier one is visible instead of silent.

Scope note (deliberate, not an oversight): an assertion names the observation the policy sees at
the moment of the judgement. For this product's adapters that is sufficient, because agent state
(cooldown, rolling alert rate, queue load) is part of the observation vector. For an environment
that keeps hidden state outside the observation the corpus is a lower bound on real behaviour,
which is why the recorded observation prefix is kept for audit rather than treated as a full
trajectory replay.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, Field

CORPUS_FILENAME = "eval_corpus.json"
CORPUS_VERSION = 1

# One critique can name an action in dozens of steps; the corpus stays a test suite, not a dump.
MAX_CASES_PER_CRITIQUE = 20


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _vector_key(vector: Sequence[float]) -> tuple[int, ...]:
    """Rounds to 6 decimals so float noise cannot create duplicate cases."""
    return tuple(int(round(float(x) * 1_000_000)) for x in vector)


class GoldenEvalCase(BaseModel):
    """One executable assertion about the policy's behaviour in a recorded situation."""

    case_id: str = Field(default_factory=lambda: f"case_{uuid.uuid4().hex[:12]}")
    source: str = "user_critique"  # val_seed | user_locked_step | user_critique | passive_log
    obs_prefix: list[list[float]] = Field(default_factory=list)
    expected_actions: list[int] = Field(default_factory=list)
    forbidden_actions: list[int] = Field(default_factory=list)
    action_labels: list[str] = Field(default_factory=list)
    rationale: str = ""
    state_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    # A later user judgement can supersede an older assertion; waived cases are reported, not scored.
    waived: bool = False
    waived_reason: str = ""

    def obs(self) -> list[float]:
        return self.obs_prefix[-1] if self.obs_prefix else []

    def fingerprint(self) -> str:
        payload = {
            "obs": _vector_key(self.obs()),
            "expected": sorted(self.expected_actions),
            "forbidden": sorted(self.forbidden_actions),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]

    def is_assertion(self) -> bool:
        return bool(self.expected_actions or self.forbidden_actions)


class EvalCorpusReport(BaseModel):
    total_cases: int = 0
    scored_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    waived_cases: int = 0
    pass_rate_pct: float = 0.0
    by_source: dict[str, dict[str, int]] = Field(default_factory=dict)
    regressed_case_ids: list[str] = Field(default_factory=list)
    failing_case_ids: list[str] = Field(default_factory=list)
    failed_details: list[dict[str, Any]] = Field(default_factory=list)


class EvalCorpusStore:
    """Persists and evaluates the per-session behavioural regression suite."""

    def __init__(self, base_dir: Path | str = "workspaces"):
        self.base_dir = Path(base_dir)

    # -- persistence -------------------------------------------------------
    def corpus_path(self, session_id: str) -> Path:
        return self.base_dir / session_id / CORPUS_FILENAME

    def load(self, session_id: str) -> list[GoldenEvalCase]:
        path = self.corpus_path(session_id)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        items = raw.get("cases", []) if isinstance(raw, dict) else raw
        cases: list[GoldenEvalCase] = []
        for item in items:
            try:
                cases.append(GoldenEvalCase.model_validate(item))
            except Exception:
                continue  # a single corrupt row must not destroy the suite
        return cases

    def save(self, session_id: str, cases: Iterable[GoldenEvalCase]) -> Path:
        path = self.corpus_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CORPUS_VERSION,
            "updated_at": _now(),
            "cases": [c.model_dump() for c in cases],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    # -- growth channels ---------------------------------------------------
    def add_case(
        self,
        session_id: str,
        obs_vector: Sequence[float],
        *,
        expected_actions: Sequence[int] | None = None,
        forbidden_actions: Sequence[int] | None = None,
        source: str = "user_critique",
        rationale: str = "",
        state_summary: dict[str, Any] | None = None,
        action_labels: Sequence[str] | None = None,
        obs_prefix: Sequence[Sequence[float]] | None = None,
    ) -> GoldenEvalCase | None:
        """Appends one assertion; returns None when it is a no-op or a duplicate."""
        if obs_vector is None or len(obs_vector) == 0:
            return None
        case = GoldenEvalCase(
            source=source,
            obs_prefix=[list(map(float, p)) for p in (obs_prefix or [obs_vector])],
            expected_actions=[int(a) for a in (expected_actions or [])],
            forbidden_actions=[int(a) for a in (forbidden_actions or [])],
            action_labels=list(action_labels or []),
            rationale=rationale,
            state_summary=state_summary or {},
        )
        if not case.is_assertion():
            return None
        cases = self.load(session_id)
        fingerprint = case.fingerprint()
        if any(existing.fingerprint() == fingerprint for existing in cases):
            return None
        cases.append(case)
        self.save(session_id, cases)
        return case

    def waive_case(self, session_id: str, case_id: str, reason: str = "") -> bool:
        """Marks an assertion as superseded by a later user judgement."""
        cases = self.load(session_id)
        for case in cases:
            if case.case_id == case_id:
                case.waived = True
                case.waived_reason = reason or "superseded by a later user judgement"
                self.save(session_id, cases)
                return True
        return False

    def seed_from_validation_split(
        self, session_id: str, env: Any, *, max_cases: int = 30
    ) -> list[GoldenEvalCase]:
        """Seeds held-out cases the policy never trains on.

        Validation rows are disjoint from training rows by the content-hash split, so these
        assertions are genuine generalisation checks rather than memorised examples.
        """
        getter = getattr(env, "get_demonstrations", None)
        if getter is None:
            return []
        try:
            features, targets = getter(num_samples=max_cases, split="val")
        except (TypeError, ValueError):
            return []
        except Exception:
            return []
        if features is None or targets is None or len(features) == 0:
            return []

        labels = list(getattr(env, "action_labels", []) or [])
        added: list[GoldenEvalCase] = []
        for obs, target in zip(features, targets):
            case = self.add_case(
                session_id,
                [float(x) for x in obs],
                expected_actions=[int(target)],
                source="val_seed",
                rationale="Held-out validation row: the label the policy should reproduce.",
                action_labels=labels,
            )
            if case is not None:
                added.append(case)
        return added

    def harvest_from_critique(
        self,
        session_id: str,
        critique_text: str,
        episodes: Iterable[dict[str, Any]],
        action_labels: Sequence[str] | None = None,
    ) -> list[GoldenEvalCase]:
        """Turns "never do X when Y" into forbidden-action cases from the recent rollouts."""
        text = " ".join(str(critique_text or "").lower().split()).replace("_", " ")
        labels = list(action_labels or [])
        if not text or not labels:
            return []

        criticised = {
            idx
            for idx, label in enumerate(labels)
            if label and str(label).lower().replace("_", " ") in text
        }
        if not criticised:
            return []

        added: list[GoldenEvalCase] = []
        for episode in episodes:
            if len(added) >= MAX_CASES_PER_CRITIQUE:
                break
            prefix: list[list[float]] = []
            for step in episode.get("steps", []) or []:
                obs = step.get("obs")
                if obs:
                    prefix.append([float(x) for x in obs])
                action = step.get("action")
                if action is None or not prefix:
                    continue
                if int(action) in criticised:
                    case = self.add_case(
                        session_id,
                        prefix[-1],
                        forbidden_actions=[int(action)],
                        source="user_critique",
                        rationale=f"User critiqued this behaviour: {critique_text[:200]}",
                        state_summary={
                            "episode_id": episode.get("episode_id"),
                            "step": step.get("step"),
                        },
                        action_labels=labels,
                        obs_prefix=prefix,
                    )
                    if case is not None:
                        added.append(case)
                    if len(added) >= MAX_CASES_PER_CRITIQUE:
                        break
        return added

    # -- evaluation --------------------------------------------------------
    def evaluate_policy(
        self,
        session_id: str,
        predict_fn: Callable[[list[float]], int],
        *,
        baseline_passing_ids: Iterable[str] | None = None,
    ) -> EvalCorpusReport:
        """Scores a policy against every non-waived assertion in the suite."""
        cases = self.load(session_id)
        baseline = set(baseline_passing_ids or [])
        report = EvalCorpusReport(total_cases=len(cases))
        passing: set[str] = set()

        for case in cases:
            bucket = report.by_source.setdefault(
                case.source, {"passed": 0, "failed": 0, "waived": 0}
            )
            if not case.is_assertion() or case.waived:
                report.waived_cases += 1
                bucket["waived"] += 1
                continue

            report.scored_cases += 1
            try:
                predicted = int(predict_fn(case.obs()))
            except Exception as exc:  # a broken predictor fails the case, never passes silently
                report.failed_cases += 1
                bucket["failed"] += 1
                report.failing_case_ids.append(case.case_id)
                report.failed_details.append(
                    {"case_id": case.case_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue

            ok = True
            if case.expected_actions and predicted not in case.expected_actions:
                ok = False
            if case.forbidden_actions and predicted in case.forbidden_actions:
                ok = False

            if ok:
                report.passed_cases += 1
                bucket["passed"] += 1
                passing.add(case.case_id)
            else:
                report.failed_cases += 1
                bucket["failed"] += 1
                report.failing_case_ids.append(case.case_id)
                report.failed_details.append(
                    {
                        "case_id": case.case_id,
                        "source": case.source,
                        "predicted": predicted,
                        "expected": case.expected_actions,
                        "forbidden": case.forbidden_actions,
                        "rationale": case.rationale,
                    }
                )

        report.pass_rate_pct = (
            round(100.0 * report.passed_cases / report.scored_cases, 2)
            if report.scored_cases
            else 0.0
        )
        report.regressed_case_ids = sorted(baseline - passing)
        return report

    def passing_case_ids(self, session_id: str, report: EvalCorpusReport) -> set[str]:
        """Case ids that passed, so a later round can detect regressions against them."""
        failing = set(report.failing_case_ids)
        return {
            case.case_id
            for case in self.load(session_id)
            if case.is_assertion() and not case.waived and case.case_id not in failing
        }


def greedy_predictor(model: Any) -> Callable[[list[float]], int]:
    """Wraps an SB3 policy as ``obs -> argmax action``, evaluated exactly as it exports."""
    import numpy as np
    import torch

    def predict(obs: Sequence[float]) -> int:
        tensor = torch.tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            features = model.policy.extract_features(tensor)
            latent_pi, _ = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)
            return int(torch.argmax(logits, dim=-1).item())

    return predict
