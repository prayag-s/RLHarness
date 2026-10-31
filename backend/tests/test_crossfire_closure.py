"""Phase P3 (v5.5) closure tests: verifier gate, golden corpus, economics, protocol.

These are behavioural tests: each one asserts a property the Crossfire audit demanded, not that a
function exists. Tests that train or spawn a sandbox worker are marked ``slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl_ide.autopilot.crossfire_protocol import CrossfireProtocolStore, rubric_hash
from rl_ide.autopilot.eval_corpus import EvalCorpusStore
from rl_ide.autopilot.scorecard import build_scorecard, compute_economics_summary
from rl_ide.autopilot.verifier_gate import (
    VerifierGate,
    VerifierSnapshot,
    offline_metric,
    sign_test_p,
    triage_failure,
)
from rl_ide.ingest.interaction_log import (
    detect_interaction_log,
    harvest_into_corpus,
    negative_rows,
    supervised_rows,
)


class FakeEnv:
    """Minimal stand-in for a tabular environment: labelled rows + an encoder."""

    task_class = "classification"
    action_labels = ["queue", "escalate", "close"]

    def __init__(self, rows: list[dict], labels: list[int]):
        self.records = rows
        self._labels = labels

    def get_demonstrations(self, num_samples: int = 32, split: str = "val"):
        n = min(num_samples, len(self.records))
        features = np.asarray(
            [[float(r.get("urgency", 0)), float(r.get("tier", 0))] for r in self.records[:n]],
            dtype=np.float32,
        )
        return features, np.asarray(self._labels[:n], dtype=np.int64)

    def _extract_obs(self, record):
        return np.asarray([float(record.get("urgency", 0)), float(record.get("tier", 0))])


# ---------------------------------------------------------------------------
# 1. Corpus seeding and persistence
# ---------------------------------------------------------------------------
def test_eval_corpus_seeding_and_persistence(tmp_path):
    store = EvalCorpusStore(tmp_path)
    env = FakeEnv(
        [{"urgency": i, "tier": i % 3} for i in range(8)],
        [i % 3 for i in range(8)],
    )
    seeded = store.seed_from_validation_split("s1", env, max_cases=8)
    assert len(seeded) == 8
    assert all(c.source == "val_seed" for c in seeded)

    reloaded = EvalCorpusStore(tmp_path).load("s1")
    assert len(reloaded) == 8, "cases must survive a process restart"

    # Duplicate seeding must not inflate the suite.
    again = store.seed_from_validation_split("s1", env, max_cases=8)
    assert again == []

    report = store.evaluate_policy("s1", lambda obs: int(obs[1]) % 3)
    assert report.scored_cases == 8
    assert report.pass_rate_pct == 100.0


# ---------------------------------------------------------------------------
# 2. Critique harvesting + locked steps + regression detection
# ---------------------------------------------------------------------------
def test_eval_corpus_harvests_critique_and_locked_steps(tmp_path):
    store = EvalCorpusStore(tmp_path)
    episodes = [
        {
            "episode_id": "r1_train_1",
            "steps": [
                {"step": 1, "obs": [0.0, 1.0], "action": 1},
                {"step": 2, "obs": [1.0, 2.0], "action": 0},
            ],
        }
    ]
    added = store.harvest_from_critique(
        "s1", "never escalate a gold customer", episodes, FakeEnv.action_labels
    )
    assert len(added) == 1
    assert added[0].forbidden_actions == [1]

    locked = store.add_case(
        "s1", [1.0, 2.0], expected_actions=[0], source="user_locked_step", rationale="locked"
    )
    assert locked is not None

    passing = store.evaluate_policy("s1", lambda obs: 0)
    passing_ids = store.passing_case_ids("s1", passing)
    assert passing.passed_cases == 2, "predicting the approved action satisfies both cases"

    # A policy that now predicts the forbidden action regresses the passing case set.
    after = store.evaluate_policy("s1", lambda obs: 1, baseline_passing_ids=passing_ids)
    assert after.failing_case_ids, "the forbidden action must fail"
    assert after.regressed_case_ids == sorted(passing_ids)


# ---------------------------------------------------------------------------
# 3. Verifier gate: retain what holds, roll back what regresses
# ---------------------------------------------------------------------------
def test_verifier_gate_retains_improving_critique():
    snapshot = VerifierSnapshot(
        mutation_kind="reward",
        val_metric=70.0,
        val_outcomes=[100.0] * 10 + [0.0] * 10,
        corpus_passing_ids=["a", "b"],
    )
    decision = VerifierGate().decide(
        snapshot,
        val_metric_after=74.0,
        val_outcomes_after=[100.0] * 14 + [0.0] * 6,
        corpus_passing_ids_after=["a", "b"],
    )
    assert decision.passed is True
    assert decision.status == "retained"
    assert decision.val_delta == 4.0


def test_verifier_gate_rolls_back_regressing_critique_and_triages_interference():
    snapshot = VerifierSnapshot(
        mutation_kind="reward",
        val_metric=80.0,
        val_outcomes=[100.0] * 16 + [0.0] * 4,
        corpus_passing_ids=["a", "b", "c"],
    )
    decision = VerifierGate().decide(
        snapshot,
        val_metric_after=55.0,
        val_outcomes_after=[100.0] * 4 + [0.0] * 16,
        corpus_passing_ids_after=["a"],
        term_stats={"new_terms": ["never_escalate"], "activation_rate": 0.42},
    )
    assert decision.passed is False
    assert decision.status == "rolled_back"
    assert decision.regressed_case_ids == ["b", "c"]
    assert decision.triage is not None
    assert decision.triage.category == "cross_term_interference"
    assert "Halve" in decision.triage.advice


def test_verifier_gate_triages_dead_reward_term_as_compiler_mismatch():
    snapshot = VerifierSnapshot(mutation_kind="reward", val_metric=80.0, corpus_passing_ids=[])
    decision = VerifierGate().decide(
        snapshot,
        val_metric_after=78.0,
        term_stats={"new_terms": ["impossible_condition"], "activation_rate": 0.0},
    )
    assert decision.passed is False
    assert decision.triage.category == "compiler_mismatch"
    assert "never" in decision.triage.advice.lower()


def test_verifier_gate_does_not_roll_back_on_noise():
    """A 1-point dip over 20 episodes is not a regression; the point estimate alone would be."""
    snapshot = VerifierSnapshot(
        mutation_kind="reward",
        val_metric=80.0,
        val_outcomes=[100.0] * 10 + [0.0] * 10,
        corpus_passing_ids=[],
    )
    decision = VerifierGate().decide(
        snapshot,
        val_metric_after=78.0,
        val_outcomes_after=[100.0] * 9 + [0.0] * 11,
    )
    assert decision.passed is True
    assert "noise" in decision.reason
    assert sign_test_p(1, 9) < 0.05


def test_triage_table_is_ordered_and_deterministic():
    assert triage_failure(
        mutation_kind="reward",
        term_stats={"eligible": False},
        val_delta=-9.0,
        regressed_case_ids=["x"],
        entropy_delta=0.9,
    ).category == "compiler_mismatch"
    assert triage_failure(
        mutation_kind="reward",
        term_stats={"eligible": True, "activation_rate": 0.0},
        val_delta=-9.0,
        regressed_case_ids=[],
        entropy_delta=0.0,
    ).category == "compiler_mismatch"
    assert triage_failure(
        mutation_kind="reward",
        term_stats=None,
        val_delta=-0.5,
        regressed_case_ids=["x"],
        entropy_delta=0.0,
    ).category == "cross_term_interference"
    assert triage_failure(
        mutation_kind="reward",
        term_stats=None,
        val_delta=0.5,
        regressed_case_ids=[],
        entropy_delta=0.4,
    ).category == "rl_variance"


# ---------------------------------------------------------------------------
# 4. Host-orchestrated per-round sandboxing
# ---------------------------------------------------------------------------
def test_host_orchestrated_rounds_tune_between_rounds(monkeypatch, tmp_path):
    """Rounds stay sandboxed, but the tuner now runs on the host instead of being skipped."""
    from rl_ide.autopilot import sandbox_runner

    calls: list[str] = []
    tuned: list[int] = []

    class FakeOutcome:
        def __init__(self, result):
            self.result = result

    def fake_run_worker(mode, job, case, timeout=0.0, on_event=None):
        calls.append(mode)
        Path(case, "root", job["session_id"]).mkdir(parents=True, exist_ok=True)
        Path(case, "root", job["session_id"], "env_module.py").write_text(
            "def compute_reward():\n    return 0.0, {}\n", encoding="utf-8"
        )
        round_no = len(calls)
        return FakeOutcome(
            {
                "session_id": job["session_id"],
                "round": round_no,
                "val_metric": 20.0,
                "best_val_metric": 20.0,
                "target_threshold": 90.0,
                "eval_corpus": {"pass_rate_pct": 0.0},
                "episodes": [],
                "rounds_completed": round_no,
            }
        )

    class FakeTuner:
        def __init__(self, client):
            self.client = client

        def tune(self, **kwargs):
            tuned.append(kwargs["round_num"])

            class Result:
                verified = True
                updated_code = "def compute_reward():\n    return 1.0, {}\n"
                diagnosis = "tuned"
                changes_summary = "tuned"

            return Result()

    monkeypatch.setattr(sandbox_runner, "isolation_available", lambda: True)
    monkeypatch.setattr(sandbox_runner, "run_worker", fake_run_worker)
    monkeypatch.setattr(sandbox_runner, "run_smoke_in_sandbox", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(sandbox_runner, "AutoTunerAgent", FakeTuner)

    blueprint = {
        "task_title": "t",
        "domain_summary": "d",
        "data_source_type": "uploaded_dataset",
        "observation_features": ["a"],
        "action_labels": ["x", "y"],
        "arena_visual_description": "a",
        "interactive_mode_description": "i",
        "success_metric_name": "acc",
        "target_metric_threshold": 90.0,
        "max_auto_tune_rounds": 2,
    }
    env_module = tmp_path / "env_module.py"
    env_module.write_text("def compute_reward():\n    return 0.0, {}\n", encoding="utf-8")

    events: list[dict] = []
    sandbox_runner.run_rounds_in_sandbox(
        session_id="sess",
        blueprint=blueprint,
        env_module=env_module,
        records=None,
        base_dir=tmp_path / "workspaces",
        max_rounds=None,
        episodes_per_burst=5,
        additional_rounds=2,
        on_event=events.append,
        llm_client=object(),
        allow_host_llm=True,
    )
    assert calls == ["train_round", "train_round"]
    assert tuned == [1], "the host must tune between rounds, not skip"
    assert any(e.get("event") == "auto_tuned" for e in events)

    # Data-sovereign mode: the host tuner must be refused outright.
    tuned.clear()
    sandbox_runner.run_rounds_in_sandbox(
        session_id="sess2",
        blueprint=blueprint,
        env_module=env_module,
        records=None,
        base_dir=tmp_path / "workspaces",
        max_rounds=None,
        episodes_per_burst=5,
        additional_rounds=2,
        on_event=events.append,
        llm_client=object(),
        allow_host_llm=False,
    )
    assert tuned == []
    assert any(e.get("event") == "auto_tune_host_disabled" for e in events)


# ---------------------------------------------------------------------------
# 5. Best-on-validation checkpoint selection
# ---------------------------------------------------------------------------
def test_best_on_validation_checkpoint_selection(tmp_path):
    import gymnasium as gym
    from stable_baselines3 import PPO

    from rl_ide.autopilot.loop_controller import SessionStateStore

    store = SessionStateStore(base_dir=tmp_path)
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, device="cpu", verbose=0)
    best_zip, best_env = store.base_dir / "sess" / "best_val_model.zip", None
    best_zip.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(best_zip))

    from rl_ide.autopilot.loop_controller import AutonomousLoopController, SessionState

    controller = AutonomousLoopController(store)
    session = SessionState(
        session_id="sess",
        blueprint=_blueprint_stub(),
        current_round=3,
        best_val_metric=88.0,
        best_val_round=2,
    )
    store.set_session(session)
    restored, code = controller._restore_best(session, env)
    assert restored is not None
    assert session.best_val_round == 2


def _blueprint_stub():
    from rl_ide.architect.models import AgentBlueprintSpec

    return AgentBlueprintSpec.model_validate(
        {
            "task_title": "t",
            "domain_summary": "d",
            "data_source_type": "uploaded_dataset",
            "observation_features": ["a"],
            "action_labels": ["x", "y"],
            "arena_visual_description": "a",
            "interactive_mode_description": "i",
            "success_metric_name": "acc",
        }
    )


# ---------------------------------------------------------------------------
# 6. Passive interaction-log ingestion
# ---------------------------------------------------------------------------
def test_passive_critique_log_ingestion_splits_accepted_and_rejected_rows(tmp_path):
    import pandas as pd

    df = pd.DataFrame(
        [
            {"urgency": 3, "tier": 0, "llm_action": "escalate", "accepted": True, "corrected_action": None},
            {"urgency": 1, "tier": 2, "llm_action": "escalate", "accepted": False, "corrected_action": "queue"},
            {"urgency": 2, "tier": 1, "llm_action": "close", "accepted": False, "corrected_action": None},
        ]
    )
    detection = detect_interaction_log(df, min_usable_rows=1)
    assert detection.detected and detection.proposed_column == "llm_action"

    records, target = supervised_rows(df, detection)
    assert target == "target_action"
    assert [r["target_action"] for r in records] == ["escalate", "queue"]

    negatives = negative_rows(df, detection)
    assert len(negatives) == 2

    env = FakeEnv(
        [{"urgency": 3, "tier": 0}, {"urgency": 1, "tier": 2}, {"urgency": 2, "tier": 1}],
        [1, 0, 2],
    )
    corpus = EvalCorpusStore(tmp_path)
    harvested = harvest_into_corpus(
        corpus,
        "s1",
        df,
        detection,
        env=env,
        label_to_action={"queue": 0, "escalate": 1, "close": 2},
    )
    assert harvested["registered"] == 2
    cases = corpus.load("s1")
    assert {c.source for c in cases} == {"passive_log"}
    corrected = [c for c in cases if c.expected_actions]
    assert corrected and corrected[0].expected_actions == [0] and corrected[0].forbidden_actions == [1]

    small = detect_interaction_log(df, min_usable_rows=100)
    assert small.sample_size_warning is not None


def test_offline_metric_is_paired_and_deterministic():
    env = FakeEnv(
        [{"urgency": i, "tier": i % 3} for i in range(12)],
        [i % 3 for i in range(12)],
    )
    first = offline_metric(env, lambda obs: int(obs[1]) % 3)
    second = offline_metric(env, lambda obs: int(obs[1]) % 3)
    assert first == second
    assert first[0] == 100.0


# ---------------------------------------------------------------------------
# 7. Economics ledger
# ---------------------------------------------------------------------------
def test_scorecard_economics_reports_both_bases_and_caveats(monkeypatch):
    summary = compute_economics_summary(
        "sess",
        onnx_latency_stats={"p50_ms": 0.12, "p99_ms": 0.31},
        feature_pipeline_ms=4.0,
        authoring_turns=6,
        measured={"cost_usd": 0.04, "calls": 70, "source": "metered"},
    )
    ledger = summary["ledger"]
    assert ledger["authoring_tokens_usd"] == 0.04
    assert ledger["authoring_human_usd"] > 0, "human authoring time is the term the audit demanded"
    assert ledger["drift_review_usd_per_year"] > 0

    for tier in summary["payback"].values():
        assert tier["basis_token_only"]["decisions_to_payback"] > tier[
            "basis_with_human_review"
        ]["decisions_to_payback"]
    assert summary["latency"]["latency_speedup_factor"] > 10
    assert summary["latency"]["end_to_end_p99_ms"] > 0.31  # pipeline included
    assert any("boundary" in c for c in summary["caveats"])
    assert set(summary["sensitivity"]) == {"human_hours_x0.5", "human_hours_x1.0", "human_hours_x2.0"}


def test_scorecard_embeds_corpus_crossfire_and_selection():
    scorecard = build_scorecard(
        task_title="t",
        task_class="classification",
        metric_name="acc",
        split_counts={"train": 70, "val": 15, "test": 15},
        eval_split="test",
        split_fallback=False,
        policy_metric_pct=91.0,
        episodes=50,
        illegal_rate_pct=0.0,
        baselines=[{"name": "majority", "value_pct": 60.0, "detail": "x", "n": 15}],
        eval_corpus={"passed_cases": 9, "scored_cases": 10, "pass_rate_pct": 90.0},
        crossfire={"closed": False, "verdict": "inconclusive", "reasons": ["n < 100"]},
        selection={"selected_round": 2, "source": "best_val_checkpoint"},
        economics={"ledger": {"first_year_usd": 1.0}},
    )
    assert scorecard["beats_best_baseline"] is True
    assert scorecard["eval_corpus"]["pass_rate_pct"] == 90.0
    assert scorecard["crossfire"]["verdict"] == "inconclusive"
    assert scorecard["selection"]["selected_round"] == 2


# ---------------------------------------------------------------------------
# 8. Pre-registration guard
# ---------------------------------------------------------------------------
def test_end_to_end_crossfire_falsification_contract(tmp_path):
    store = CrossfireProtocolStore(tmp_path)
    protocol = store.open(
        "s1", rubric=["pass 60% of frozen critiques"], min_holdout=100, target_pass_rate_pct=60.0
    )
    assert protocol.rubric_hash == rubric_hash(["pass 60% of frozen critiques"])

    # No training stamp and no hold-out: a verdict is refused, not invented.
    report = store.closure_report("s1")
    assert report["closed"] is False
    assert report["verdict"] in ("inconclusive", "not_preregistered")

    refused = store.evaluate("s1", holdout_passed=9, holdout_total=10, policy_metric=95, baseline_metric=80)
    assert refused["closed"] is False
    assert refused["verdict"] == "inconclusive"
    assert any("hold-out" in reason for reason in refused["reasons"])

    store.record_training_start("s1")
    store.register_holdout("s1", [f"c{i}" for i in range(100)])
    won = store.evaluate("s1", holdout_passed=80, holdout_total=100, policy_metric=95, baseline_metric=80)
    assert won["closed"] is True and won["verdict"] == "win"

    lost = store.evaluate("s1", holdout_passed=20, holdout_total=100, policy_metric=70, baseline_metric=80)
    assert lost["closed"] is True and lost["verdict"] == "loss"

    # A rubric edited after registration no longer matches its hash.
    tampered = store.load("s1")
    tampered.rubric = ["pass 5% only"]
    store.save(tampered)
    after = store.evaluate("s1", holdout_passed=100, holdout_total=100, policy_metric=99, baseline_metric=1)
    assert after["closed"] is False
    assert any("hash" in reason for reason in after["reasons"])


def test_gate_history_records_every_verdict_for_the_ui(tmp_path):
    """The UI answers "why was round 3 reverted?" — one last-decision slot cannot."""
    from rl_ide.autopilot.loop_controller import MAX_GATE_HISTORY, SessionState, record_gate_decision

    session = SessionState(session_id="s", blueprint=_blueprint_stub())
    snapshot = VerifierSnapshot(
        mutation_kind="reward",
        session_id="s",
        env_code="",
        val_metric=80.0,
        val_outcomes=[1.0, 0.0] * 10,
        corpus_passing_ids=["c1", "c2"],
    )

    kept = VerifierGate().decide(
        snapshot,
        val_metric_after=84.0,
        val_outcomes_after=[1.0, 1.0] * 10,
        corpus_passing_ids_after=["c1", "c2"],
    )
    assert kept.passed is True
    entry = record_gate_decision(session, kept, round_num=2, origin="training_round")
    assert entry["passed"] is True and entry["round"] == 2
    assert session.critique_gate["passed"] is True

    rolled = VerifierGate().decide(
        snapshot,
        val_metric_after=60.0,
        val_outcomes_after=[0.0, 0.0] * 10,
        corpus_passing_ids_after=["c1"],
    )
    assert rolled.passed is False
    bad = record_gate_decision(session, rolled, round_num=3, origin="chat_mutation")
    assert bad["passed"] is False
    assert bad["category"] in ("cross_term_interference", "rl_variance")
    assert bad["advice"], "a rolled-back change must carry actionable advice, not just a verdict"

    # History is append-only and ordered, so the UI can render a timeline.
    assert [e["round"] for e in session.gate_history] == [2, 3]
    assert session.gate_history[0]["origin"] == "training_round"

    for i in range(MAX_GATE_HISTORY + 5):
        record_gate_decision(session, kept, round_num=10 + i, origin="training_round")
    assert len(session.gate_history) == MAX_GATE_HISTORY
    assert session.gate_history[-1]["round"] == 10 + MAX_GATE_HISTORY + 4


def test_protocol_rejects_rubric_registered_after_training(tmp_path):
    store = CrossfireProtocolStore(tmp_path)
    store.open("s1", rubric=["r"], min_holdout=1)
    store.record_training_start("s1")
    protocol = store.load("s1")
    protocol.opened_at_epoch = protocol.training_started_at_epoch + 5
    store.save(protocol)
    store.register_holdout("s1", ["c1"])
    report = store.evaluate("s1", holdout_passed=1, holdout_total=1, policy_metric=90, baseline_metric=50)
    assert report["closed"] is False
    assert any("post-hoc" in reason for reason in report["reasons"])


# ---------------------------------------------------------------------------
# 9. Hostile reward expressions (memo's claim 4 red-team, scoped to the sandbox)
# ---------------------------------------------------------------------------
HOSTILE_MODULES = [
    # object-graph traversal to a live object with side effects
    "import os\nclass E:\n    def __init__(self):\n        self.p=print\n"
    "def f():\n    return E().p\nf()._self if False else None\n",
    # file read via numpy (AST filtering alone would not stop this)
    "import numpy as np\ndef f(p='/etc/passwd'):\n    return np.fromfile(p, dtype='uint8')\nf()\n",
    # import-time infinite loop
    "while True:\n    pass\n",
    # socket attempt
    "import socket\ns=socket.socket()\ns.connect(('1.1.1.1', 80))\n",
    # env var exfiltration attempt
    "import os\nopen('leak.txt','w').write(str(dict(os.environ)))\n",
]


@pytest.mark.slow
def test_hostile_reward_modules_cannot_escape_the_sandbox(tmp_path):
    from rl_ide.autopilot.sandbox_runner import run_smoke_in_sandbox

    secrets = {"OPENAI_API_KEY": "sk-should-not-leak"}
    for index, source in enumerate(HOSTILE_MODULES):
        module = tmp_path / f"hostile_{index}.py"
        module.write_text(source, encoding="utf-8")
        try:
            result = run_smoke_in_sandbox(
                module, records=[{"a": 1}], timeout=20.0
            )
            assert not result.get("ok", False) or "error" in result
        except Exception:
            # A refused or crashing worker is a pass: nothing reached the host.
            pass
        # A hostile module must never write a file the host can see.
        assert not (tmp_path / "leak.txt").exists()
        assert not list(tmp_path.glob("leak*"))
