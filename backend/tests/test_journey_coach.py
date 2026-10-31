"""Journey Coach: deterministic findings, hard blocking, acknowledgement and judge safety.

These tests pin the properties the design promised:

* rules are free and run without a provider,
* a structural blocker stops execution in hard mode and can never be acknowledged away,
* a warning survives a restart and stays acknowledged until its evidence changes,
* the LLM can only restate facts, never invent a number, and failure degrades to rules.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rl_ide.coach import (
    CoachBlocked,
    CoachStage,
    JourneyCoach,
    Severity,
)
from rl_ide.coach import judge as judge_module
from rl_ide.coach.facts import (
    dataset_facts,
    objective_facts,
    parameter_facts,
    preview_facts,
    result_facts,
)
from rl_ide.coach.rules import (
    dataset_items,
    objective_items,
    parameter_items,
    preview_items,
    result_items,
)
from rl_ide.coach.store import CoachStore


class Dataset:
    """Minimal stand-in for ``IngestedDataset``: a frame plus an optional profile."""

    def __init__(self, df: pd.DataFrame, target: str | None = None, name: str = "data"):
        self.df = df
        self.name = name
        self.metadata: dict = {"label_column": target} if target else {}
        self.profile = None
        self.target = target


def blueprint(**overrides) -> dict:
    base = {
        "task_title": "Invoice triage",
        "domain_summary": "Decide whether an invoice is approved, reviewed or blocked.",
        "data_source_type": "pure_simulation",
        "observation_features": ["amount", "vendor_risk_score"],
        "arena_visual_description": "A dark invoice row with the decision and top reasons.",
        "interactive_mode_description": "Inspect one invoice and override the decision.",
        "success_metric_name": "Audit Accuracy (%)",
        "action_labels": ["approve", "review", "block"],
        "max_auto_tune_rounds": 3,
        "episodes_per_round": 200,
        "target_metric_threshold": 85.0,
        "initial_reward_components": [
            {"name": "correct_decision", "weight": 1.0, "description": "reward a correct call"}
        ],
    }
    base.update(overrides)
    return base


def make_store(tmp_path) -> CoachStore:
    return CoachStore(tmp_path / "coach.db")


class FakeClient:
    def __init__(self, payload=None, fail: bool = False):
        self.payload = payload or {"items": []}
        self.fail = fail
        self.calls = 0

    def generate_json(self, prompt, system_prompt=None, timeout=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return self.payload


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def test_a_vague_objective_is_flagged():
    facts = objective_facts("Build a fraud detector.", blueprint())
    ids = {item.rule_id for item in objective_items(facts)}
    assert "objective.too_vague" in ids
    assert "objective.no_success_metric" in ids


def test_an_encoded_prohibition_is_not_flagged():
    prompt = (
        "Given one invoice row, decide approve, review, or block. Success is at least 90% "
        "agreement with the recorded outcome. Never approve when vendor_risk_score >= 85."
    )
    facts = objective_facts(prompt, blueprint())
    assert facts.hard_constraint_count >= 1
    ids = {item.rule_id for item in objective_items(facts)}
    assert "objective.unencoded_prohibition" not in ids


def test_an_unencoded_prohibition_is_flagged_with_a_fix():
    prompt = (
        "Given one invoice row, decide approve, review, or block. Success is at least 90% "
        "agreement. Never approve a risky invoice."
    )
    facts = objective_facts(prompt, blueprint())
    item = next(
        i for i in objective_items(facts) if i.rule_id == "objective.unencoded_prohibition"
    )
    assert item.severity == Severity.WARNING
    assert any(action.kind == "send_message" for action in item.actions)


def test_no_action_space_is_a_blocker():
    facts = objective_facts("Do something useful with this table.", {})
    item = next(i for i in objective_items(facts) if i.rule_id == "objective.no_action_space")
    assert item.severity == Severity.BLOCKER


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def test_single_class_target_blocks(tmp_path):
    df = pd.DataFrame({"amount": range(10), "label": ["ok"] * 10})
    facts = dataset_facts(Dataset(df, target="label"))
    item = next(i for i in dataset_items(facts) if i.rule_id == "data.single_class_target")
    assert item.severity == Severity.BLOCKER


def test_imbalance_and_identifier_leakage_warn():
    df = pd.DataFrame(
        {
            "invoice_id": [f"INV-{i}" for i in range(200)],
            "amount": range(200),
            "label": ["ok"] * 194 + ["bad"] * 6,
        }
    )
    facts = dataset_facts(Dataset(df, target="label"))
    ids = {i.rule_id for i in dataset_items(facts)}
    assert "data.class_imbalance" in ids
    assert "data.id_leakage" in ids


def test_missing_label_blocks_a_supervised_objective():
    df = pd.DataFrame({"amount": range(50), "vendor": ["a", "b"] * 25})
    facts = dataset_facts(Dataset(df))
    item = next(
        i
        for i in dataset_items(facts, supervised_objective=True)
        if i.rule_id == "data.missing_label_for_supervised_objective"
    )
    assert item.severity == Severity.BLOCKER
    assert any(action.kind == "send_message" for action in item.actions)


def test_empty_dataset_blocks():
    df = pd.DataFrame({"amount": []})
    facts = dataset_facts(Dataset(df))
    assert any(i.rule_id == "data.empty" for i in dataset_items(facts))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def test_target_100_and_few_episodes_are_explained_with_numbers():
    facts = parameter_facts(
        blueprint(target_metric_threshold=100.0, episodes_per_round=10), {}
    )
    items = {i.rule_id: i for i in parameter_items(facts)}
    assert "params.target_100" in items
    assert "params.episodes_too_few" in items
    assert "±" in items["params.target_100"].why
    apply_action = next(
        a for a in items["params.target_100"].actions if a.kind == "apply_params"
    )
    assert apply_action.payload["target_metric_threshold"] <= 90.0


def test_zero_budget_blocks():
    facts = parameter_facts(
        blueprint(max_auto_tune_rounds=0, episodes_per_round=0), {}
    )
    item = next(i for i in parameter_items(facts) if i.rule_id == "params.nonpositive_budget")
    assert item.severity == Severity.BLOCKER


def test_an_out_of_range_target_warns_but_does_not_block():
    facts = parameter_facts(blueprint(target_metric_threshold=105.0), {})
    item = next(
        i for i in parameter_items(facts) if i.rule_id == "params.target_out_of_range"
    )
    assert item.severity == Severity.WARNING
    assert "exhaust its rounds" in item.why


def test_target_below_baseline_is_flagged():
    df = pd.DataFrame({"label": ["ok"] * 90 + ["bad"] * 10, "x": range(100)})
    facts = parameter_facts(
        blueprint(target_metric_threshold=88.0), {}, Dataset(df, target="label")
    )
    assert any(i.rule_id == "params.target_below_baseline" for i in parameter_items(facts))


# ---------------------------------------------------------------------------
# Preview and results
# ---------------------------------------------------------------------------

def test_preview_without_a_metric_is_flagged():
    facts = preview_facts(
        blueprint(success_metric_name="Audit Accuracy (%)"),
        "<html><body><canvas></canvas><script>setInterval(()=>{}, 1000)</script></body></html>",
    )
    assert any(i.rule_id == "preview.no_metric" for i in preview_items(facts))


def test_static_preview_is_flagged():
    facts = preview_facts(blueprint(), "<html><body>Accuracy 90%</body></html>")
    assert any(i.rule_id == "preview.static" for i in preview_items(facts))


def test_wide_interval_and_baseline_are_explained():
    scorecard = {
        "held_out": True,
        "task_class": "classification",
        "metric_name": "Audit Accuracy (%)",
        "policy": {
            "value_pct": 80.0,
            "episodes": 50,
            "ci95_pct": [70.0, 90.0],
            "illegal_actions_pct": 0.0,
        },
        "baselines": [{"name": "majority", "value_pct": 79.0}],
        "uplift_vs_best_baseline_pct": 1.0,
        "notes": [],
    }
    facts = result_facts(scorecard)
    ids = {i.rule_id for i in result_items(facts)}
    assert "results.wide_ci" in ids
    assert "results.worse_than_baseline" in ids


# ---------------------------------------------------------------------------
# Enforcement and persistence
# ---------------------------------------------------------------------------

def test_hard_mode_raises_on_a_blocker(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_IDE_COACH_BLOCKERS", "hard")
    coach = JourneyCoach(store=make_store(tmp_path), use_llm=False)
    report = coach.evaluate(
        CoachStage.PARAMETERS,
        conversation_id="c1",
        blueprint=blueprint(max_auto_tune_rounds=0, episodes_per_round=0),
    )
    assert report.status == "blocked"
    with pytest.raises(CoachBlocked):
        coach.enforce(report)


def test_advisory_mode_reports_but_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_IDE_COACH_BLOCKERS", "advisory")
    coach = JourneyCoach(store=make_store(tmp_path), use_llm=False)
    report = coach.evaluate(
        CoachStage.PARAMETERS,
        conversation_id="c1",
        blueprint=blueprint(max_auto_tune_rounds=0, episodes_per_round=0),
    )
    assert report.status == "blocked"
    coach.enforce(report)  # must not raise


def test_a_blocker_cannot_be_acknowledged_away(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_IDE_COACH_BLOCKERS", "hard")
    coach = JourneyCoach(store=make_store(tmp_path), use_llm=False)
    coach.evaluate(
        CoachStage.PARAMETERS,
        conversation_id="c1",
        blueprint=blueprint(max_auto_tune_rounds=0, episodes_per_round=0),
    )
    with pytest.raises(PermissionError):
        coach.acknowledge("c1", "params.nonpositive_budget")


def test_a_warning_stays_acknowledged_until_evidence_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_IDE_COACH_BLOCKERS", "hard")
    store = make_store(tmp_path)
    coach = JourneyCoach(store=store, use_llm=False)
    params = {"target_metric_threshold": 100.0}

    first = coach.evaluate(
        CoachStage.PARAMETERS, conversation_id="c1", blueprint=blueprint(), params=params
    )
    assert any(i.rule_id == "params.target_100" and i.status == "open" for i in first.items)

    coach.acknowledge("c1", "params.target_100")
    again = coach.evaluate(
        CoachStage.PARAMETERS, conversation_id="c1", blueprint=blueprint(), params=params
    )
    assert any(
        i.rule_id == "params.target_100" and i.status == "acknowledged" for i in again.items
    )

    fixed = coach.evaluate(
        CoachStage.PARAMETERS,
        conversation_id="c1",
        blueprint=blueprint(),
        params={"target_metric_threshold": 90.0},
    )
    assert not any(i.rule_id == "params.target_100" for i in fixed.items)
    assert not any(
        i["rule_id"] == "params.target_100" for i in store.items("c1")
    ), "a fixed finding must be resolved, not left open forever"


def test_findings_survive_a_restart(tmp_path):
    kwargs = dict(
        conversation_id="c1",
        blueprint=blueprint(),
        params={"target_metric_threshold": 100.0},
    )
    JourneyCoach(store=make_store(tmp_path), use_llm=False).evaluate(
        CoachStage.PARAMETERS, **kwargs
    )
    coach = JourneyCoach(store=make_store(tmp_path), use_llm=False)
    report = coach.evaluate(CoachStage.PARAMETERS, **kwargs)
    assert any(i.rule_id == "params.target_100" for i in report.items)


# ---------------------------------------------------------------------------
# LLM judge safety
# ---------------------------------------------------------------------------

def _llm_item(title: str, why: str, kind: str = "send_message") -> dict:
    return {
        "rule_id": "llm.rewrite",
        "severity": "warning",
        "title": title,
        "why": why,
        "action": {"kind": kind, "label": "Fix it", "payload": {"message": "do it"}},
    }


def test_judge_rejects_an_invented_number():
    facts = objective_facts("Build a fraud detector.", blueprint()).to_dict()
    client = FakeClient(
        {"items": [_llm_item("The score will be 97.5%", "You will reach 97.5% accuracy easily.")]}
    )
    assert judge_module.judge(CoachStage.OBJECTIVE, facts, client) == []


def test_judge_accepts_a_finding_grounded_in_facts():
    facts = objective_facts(
        "Build a fraud detector with 90% accuracy.", blueprint()
    ).to_dict()
    client = FakeClient(
        {
            "items": [
                _llm_item(
                    "The success metric is not visible in the arena",
                    "The stated metric exists, but nothing in the preview exposes it to a human.",
                )
            ]
        }
    )
    items = judge_module.judge(CoachStage.OBJECTIVE, facts, client)
    assert len(items) == 1
    assert items[0].source == "llm"


def test_judge_can_never_return_a_blocker():
    facts = objective_facts("Build a fraud detector.", blueprint()).to_dict()
    payload = {
        "items": [_llm_item("This will block the run", "The setup cannot work at all.")]
    }
    payload["items"][0]["severity"] = "blocker"
    items = judge_module.judge(CoachStage.OBJECTIVE, facts, FakeClient(payload))
    assert items[0].severity == Severity.WARNING


def test_judge_failure_fails_open(tmp_path):
    coach = JourneyCoach(client=FakeClient(fail=True), store=make_store(tmp_path))
    report = coach.evaluate(
        CoachStage.OBJECTIVE,
        conversation_id="c1",
        prompt="Build a fraud detector.",
        blueprint=blueprint(),
    )
    assert report.llm_used is False
    assert "RuntimeError" in (report.llm_skipped_reason or "")
    assert report.items, "rules must still produce findings when the provider is down"


def test_judge_is_cached_by_content_hash(tmp_path):
    client = FakeClient({"items": []})
    coach = JourneyCoach(client=client, store=make_store(tmp_path))
    kwargs = dict(
        conversation_id="c1",
        prompt="Build a fraud detector.",
        blueprint=blueprint(),
    )
    coach.evaluate(CoachStage.OBJECTIVE, **kwargs)
    coach.evaluate(CoachStage.OBJECTIVE, **kwargs)
    assert client.calls == 1, "re-opening a panel must not re-bill the provider"


def test_rules_win_over_an_llm_restatement(tmp_path):
    payload = {
        "items": [
            _llm_item(
                "The objective is too short to constrain the build",
                "A short objective leaves every important choice to a default.",
            )
        ]
    }
    coach = JourneyCoach(client=FakeClient(payload), store=make_store(tmp_path))
    report = coach.evaluate(
        CoachStage.OBJECTIVE,
        conversation_id="c1",
        prompt="Build a fraud detector.",
        blueprint=blueprint(),
    )
    titles = [i.title for i in report.items]
    assert titles.count("The objective is too short to constrain the build") == 1


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

@pytest.fixture()
def guidance_api(tmp_path, monkeypatch):
    from rl_ide.api import server
    from rl_ide.state.conversation_store import ConversationStore

    conv_store = ConversationStore(
        db_path=tmp_path / "conversations.db", workspaces_dir=tmp_path / "ws"
    )
    conv_store.save_or_update(
        conversation_id="c1",
        title="Invoice triage",
        status="ready_to_build",
        blueprint_spec=blueprint(target_metric_threshold=100.0),
        training_params={
            "max_auto_tune_rounds": 3,
            "episodes_per_round": 10,
            "target_metric_threshold": 100.0,
        },
        chat_history=[
            {"role": "user", "content": "Build an invoice triage agent.", "timestamp": "now"}
        ],
    )
    coach = JourneyCoach(store=CoachStore(tmp_path / "coach.db"), use_llm=False)

    monkeypatch.setattr(server, "get_conversation_store", lambda: conv_store)
    monkeypatch.setattr(server, "get_coach", lambda: coach)
    monkeypatch.setenv("RL_IDE_COACH_BLOCKERS", "hard")
    return TestClient(server.app), coach


def test_guidance_endpoints_expose_evaluate_and_ack(guidance_api):
    client, _ = guidance_api

    listed = client.get("/api/v4/conversations/c1/guidance")
    assert listed.status_code == 200
    assert listed.json()["blocker_mode"] == "hard"

    evaluated = client.post(
        "/api/v4/conversations/c1/guidance/evaluate",
        json={"stage": "parameters", "use_llm": False},
    )
    assert evaluated.status_code == 200
    body = evaluated.json()
    ids = {item["rule_id"] for item in body["items"]}
    assert "params.target_100" in ids

    acked = client.post(
        "/api/v4/conversations/c1/guidance/params.target_100/ack", json={}
    )
    assert acked.status_code == 200
    statuses = {
        item["rule_id"]: item["status"] for item in acked.json()["items"]
    }
    assert statuses["params.target_100"] == "acknowledged"


def test_guidance_ack_refuses_a_blocker(guidance_api, monkeypatch):
    client, coach = guidance_api
    from rl_ide.api import server

    # Make the budget zero so the parameters stage produces a blocker.
    server.get_conversation_store().update_params(
        "c1", {"max_auto_tune_rounds": 0, "episodes_per_round": 0}
    )
    evaluated = client.post(
        "/api/v4/conversations/c1/guidance/evaluate",
        json={"stage": "parameters", "use_llm": False},
    )
    assert evaluated.status_code == 200
    blocked = [
        item for item in evaluated.json()["items"] if item["severity"] == "blocker"
    ]
    assert blocked

    refused = client.post(
        f"/api/v4/conversations/c1/guidance/{blocked[0]['rule_id']}/ack", json={}
    )
    assert refused.status_code == 409


def test_session_start_returns_409_with_the_finding(guidance_api):
    client, _ = guidance_api
    from rl_ide.api import server
    import shutil

    sid = "sess_coach_block"
    try:
        response = client.post(
            "/api/v4/sessions/start",
            json={
                "blueprint": blueprint(
                    max_auto_tune_rounds=0, episodes_per_round=0
                ),
                "session_id": sid,
                "run_sync": False,
            },
        )
        assert response.status_code == 409
        body = response.json()
        assert body["guidance"]["status"] == "blocked"
        assert any(
            item["rule_id"] == "params.nonpositive_budget"
            for item in body["guidance"]["items"]
        )
    finally:
        shutil.rmtree(server.global_store.base_dir / sid, ignore_errors=True)
