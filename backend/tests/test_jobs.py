"""G5/G6: a studio turn is a durable job, not a coroutine nobody can see.

These tests cover the properties a paying customer actually depends on: a fast 202 with a job id,
progress that can be polled, a cancel that is honoured at a stage boundary, and a restart that turns
interrupted work into an explicit retry hint instead of a spinner that never resolves.
"""

from __future__ import annotations

import time
import types
import uuid

import pytest
from fastapi.testclient import TestClient

from rl_ide.api import server
from rl_ide.state.job_store import (
    CANCELLED,
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    JobCancelled,
    JobStore,
    get_job_store,
)


@pytest.fixture
def job_store(tmp_path, monkeypatch) -> JobStore:
    store = JobStore(db_path=tmp_path / "jobs.db")
    monkeypatch.setattr("rl_ide.state.job_store._global_job_store", store)
    monkeypatch.setattr(server, "jobs", store)
    return store


def test_a_job_is_claimed_by_exactly_one_worker(job_store):
    job_store.create(kind="training", conversation_id="c1")
    first = job_store.claim_next("training")
    second = job_store.claim_next("training")

    assert first is not None and first["status"] == RUNNING
    assert second is None, "a queued job must never be executed twice"


def test_restart_turns_running_work_into_a_retry_hint(job_store):
    running = job_store.create(kind="chat", conversation_id="c1")
    job_store.set_stage(running["job_id"], "env_synthesis", "writing code")
    queued = job_store.create(kind="training", conversation_id="c2")

    assert job_store.reconcile_interrupted() == 2

    dead = job_store.get(running["job_id"])
    assert dead["status"] == FAILED
    assert "restarted" in dead["error"] and "retry" in dead["error"].lower()
    assert dead["retryable"] is True
    assert job_store.get(queued["job_id"])["status"] == FAILED


def test_cancel_is_cooperative_and_idempotent(job_store):
    job = job_store.create(kind="chat", conversation_id="c1")
    assert job_store.request_cancel(job["job_id"]) is True
    assert job_store.is_cancel_requested(job["job_id"]) is True

    job_store.finish(job["job_id"], status=SUCCEEDED, result={"ok": True})
    assert job_store.request_cancel(job["job_id"]) is False, "a finished job cannot be cancelled"


def test_runner_stops_at_a_stage_boundary_when_cancelled(job_store, monkeypatch):
    job = job_store.create(kind="chat", conversation_id="cancelled_conv", owner_id="acme")

    def fake_process_message(**kwargs):
        kwargs["progress_callback"]("intake", "planning")
        job_store.request_cancel(job["job_id"])
        # A real turn would call should_cancel() at the next boundary and be interrupted there.
        assert kwargs["should_cancel"]() is True
        raise JobCancelled("cancelled during intake")

    monkeypatch.setattr(server.global_omni_agent, "process_message", fake_process_message)

    server._run_chat_job(job)

    finished = job_store.get(job["job_id"])
    assert finished["status"] == CANCELLED
    assert "Cancel" in finished["error"]


@pytest.mark.slow  # runs a real (fallback-LLM) synthesis
def test_async_chat_returns_a_job_quickly_and_reports_stages(job_store):
    client = TestClient(server.app)
    conv_id = f"job_async_{uuid.uuid4().hex[:8]}"

    started = time.perf_counter()
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train an agent to triage support tickets"},
        headers={"Prefer": "respond-async"},
    )
    elapsed = time.perf_counter() - started

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["job_id"].startswith("job_")
    assert body["status"] in (QUEUED, RUNNING)
    assert elapsed < 2.0, f"the studio must acknowledge within a second, took {elapsed:.2f}s"

    stage_seen: list[str] = []
    deadline = time.time() + 300
    finished = None
    while time.time() < deadline:
        job = client.get(f"/api/v4/jobs/{body['job_id']}").json()
        if job["stage"] not in stage_seen:
            stage_seen.append(job["stage"])
        if job["status"] in (SUCCEEDED, FAILED, CANCELLED):
            finished = job
            break
        time.sleep(0.25)

    assert finished is not None, f"job never finished; stages seen: {stage_seen}"
    assert finished["status"] == SUCCEEDED, finished
    assert stage_seen[0] == "intake", stage_seen
    assert finished["result"]["conversation_id"] == conv_id
    assert finished["result"]["arena_html"]


@pytest.mark.slow  # runs a real (fallback-LLM) synthesis
def test_stage_events_stream_over_the_websocket(job_store):
    """The UI must be able to show progress live, not only by polling."""
    client = TestClient(server.app)
    conv_id = f"job_ws_{uuid.uuid4().hex[:8]}"

    with client.websocket_connect(f"/ws/v4/{conv_id}") as ws:
        resp = client.post(
            f"/api/v4/conversations/{conv_id}/chat",
            json={"user_message": "Train an agent to score insurance claims"},
            headers={"Prefer": "respond-async"},
        )
        assert resp.status_code == 202, resp.text

        events: list[dict] = []
        for _ in range(40):
            message = ws.receive_json()
            events.append(message)
            if message.get("event") in ("job_completed", "job_failed", "job_cancelled"):
                break

    assert events[-1]["event"] == "job_completed", events[-3:]
    stages = [e["stage"] for e in events if e.get("event") == "job_stage"]
    assert stages and stages[0] == "intake", stages
    job_events = [e for e in events if "job_id" in e]
    assert job_events and all(e["job_id"] == resp.json()["job_id"] for e in job_events), (
        "every job event must identify its job"
    )


def test_sync_chat_still_returns_the_full_response(job_store):
    client = TestClient(server.app)
    conv_id = f"job_sync_{uuid.uuid4().hex[:8]}"

    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train an agent to route invoices by vendor risk"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["conversation_id"] == conv_id

    listed = client.get("/api/v4/jobs", params={"conversation_id": conv_id}).json()
    assert listed and listed[0]["kind"] == "chat" and listed[0]["status"] == SUCCEEDED


def test_jobs_are_scoped_to_their_tenant(job_store):
    client = TestClient(server.app)
    job = job_store.create(kind="chat", conversation_id="c1", owner_id="acme")

    assert client.get(f"/api/v4/jobs/{job['job_id']}", headers={"x-owner-id": "globex"}).status_code == 404
    assert client.get(f"/api/v4/jobs/{job['job_id']}", headers={"x-owner-id": "acme"}).status_code == 200


def test_training_queue_runs_the_job_and_records_its_result(job_store, monkeypatch):
    conv = server.get_conversation_store().create_conversation(
        title="queue test", conversation_id=f"job_train_{uuid.uuid4().hex[:8]}"
    )
    conv_id = conv["conversation_id"]
    server.get_conversation_store().save_or_update(
        conversation_id=conv_id,
        blueprint_spec=server.AgentBlueprintSpec(
            task_title="Queue", domain_summary="d", data_source_type="pure_simulation",
            observation_features=["f"], action_labels=["a"], arena_visual_description="g",
            interactive_mode_description="p", success_metric_name="r",
            target_metric_threshold=80.0, max_auto_tune_rounds=2, initial_reward_components=[],
        ).model_dump(),
    )
    (server.global_store.base_dir / conv_id).mkdir(parents=True, exist_ok=True)
    (server.global_store.base_dir / conv_id / "env_module.py").write_text("class X: pass", encoding="utf-8")

    calls: list[dict] = []

    def fake_training(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            status="completed",
            current_round=2,
            current_metric=88.5,
            rounds_history=[],
            episodes=[],
            latest_model_zip_path=None,
        )

    monkeypatch.setattr(server, "run_training_task", fake_training)
    job = job_store.create(
        kind="training",
        conversation_id=conv_id,
        owner_id="local",
        payload={"max_rounds": 2, "additional_rounds": 2, "episodes_per_burst": 50},
    )

    server._drain_training_jobs()

    finished = job_store.get(job["job_id"])
    assert finished["status"] == SUCCEEDED, finished
    assert finished["result"]["best_metric"] == 88.5
    assert calls and calls[0]["max_rounds"] == 2 and calls[0]["episodes_per_burst"] == 50
    assert get_job_store() is job_store


# -----------------------------------------------------------------------------
# Resuming an interrupted training run (G6 residual)
# -----------------------------------------------------------------------------
def test_resume_route_continues_from_saved_rounds(tmp_path, monkeypatch):
    """An interrupted run must continue from its saved rounds, not start over."""
    from rl_ide.api.server import app
    from rl_ide.state.conversation_store import ConversationStore

    store = ConversationStore(db_path=tmp_path / "c.db", workspaces_dir=tmp_path / "ws")
    conv_id = "conv_resume"
    store.create_conversation(conversation_id=conv_id, owner_id="local")
    bp = {
        "task_title": "Router",
        "domain_summary": "route",
        "data_source_type": "pure_simulation",
        "observation_features": ["a", "b"],
        "action_labels": ["x", "y"],
        "arena_visual_description": "board",
        "interactive_mode_description": "click",
        "success_metric_name": "accuracy",
        "target_metric_threshold": 80.0,
        "max_auto_tune_rounds": 5,
        "initial_reward_components": [],
    }
    store.save_or_update(
        conv_id,
        blueprint_spec=bp,
        rounds_history=[{"round_num": 1, "metric": 60.0}, {"round_num": 2, "metric": 70.0}],
        latest_model_zip_path=str(tmp_path / "model.zip"),
    )
    (tmp_path / "ws" / conv_id).mkdir(parents=True, exist_ok=True)
    (tmp_path / "ws" / conv_id / "env_module.py").write_text("# env\n", encoding="utf-8")

    import rl_ide.api.server as server

    job_store = JobStore(db_path=tmp_path / "jobs.db")
    monkeypatch.setattr(server, "get_conversation_store", lambda: store)
    monkeypatch.setattr(server, "jobs", job_store)
    monkeypatch.setattr(server, "global_store", type("S", (), {"base_dir": tmp_path / "ws"})())
    monkeypatch.setattr(server, "_ensure_training_consumer", lambda: None)
    monkeypatch.setattr(server, "_broadcast_threadsafe", lambda *a, **k: None)

    client = TestClient(app)
    resp = client.post(f"/api/v4/conversations/{conv_id}/training/resume", json={})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["completed_rounds"] == 2
    assert body["additional_rounds"] == 3, "only the rounds that are still missing"
    assert body["warm_start"] is True

    queued = job_store.get(body["job_id"])
    assert queued["kind"] == "training"
    assert queued["payload"]["additional_rounds"] == 3
    assert queued["payload"]["resume_of_rounds"] == 2


def test_resume_is_refused_when_nothing_can_be_resumed(tmp_path, monkeypatch):
    from rl_ide.api.server import app
    from rl_ide.state.conversation_store import ConversationStore

    store = ConversationStore(db_path=tmp_path / "c2.db", workspaces_dir=tmp_path / "ws2")
    import rl_ide.api.server as server

    monkeypatch.setattr(server, "get_conversation_store", lambda: store)
    monkeypatch.setattr(server, "global_store", type("S", (), {"base_dir": tmp_path / "ws2"})())
    client = TestClient(app)

    assert client.post("/api/v4/conversations/nope/training/resume", json={}).status_code == 404

    store.create_conversation(conversation_id="conv_untrained", owner_id="local")
    resp = client.post("/api/v4/conversations/conv_untrained/training/resume", json={})
    assert resp.status_code == 409
    assert "blueprint" in resp.json()["detail"]


def test_a_failed_training_job_tells_the_client_how_to_resume(tmp_path, monkeypatch):
    import rl_ide.api.server as server
    from rl_ide.state.conversation_store import ConversationStore

    store = ConversationStore(db_path=tmp_path / "c3.db", workspaces_dir=tmp_path / "ws3")
    store.create_conversation(conversation_id="conv_hint", owner_id="local")
    store.save_or_update("conv_hint", rounds_history=[{"round_num": 1, "metric": 55.0}])
    monkeypatch.setattr(server, "get_conversation_store", lambda: store)

    hint = server._resume_hint(
        {"status": "failed", "kind": "training", "conversation_id": "conv_hint"}
    )
    assert hint and "training/resume" in hint and "1 round" in hint

    none_saved = server._resume_hint(
        {"status": "failed", "kind": "training", "conversation_id": "conv_missing"}
    )
    assert none_saved is None, "no conversation means no advice to give"

    chat = server._resume_hint({"status": "failed", "kind": "chat", "conversation_id": "conv_hint"})
    assert chat and "Re-send" in chat
    assert server._resume_hint({"status": "succeeded", "kind": "chat"}) is None
