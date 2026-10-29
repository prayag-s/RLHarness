"""Hardening tests for transport, isolation, verification gating, durability and honesty."""

import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rl_ide.architect.intake import IntakeArchitectAgent
from rl_ide.architect.omni_agent import StudioOmniAgent
from rl_ide.autopilot.loop_controller import AutonomousLoopController, SessionStateStore
from rl_ide.api.server import app, global_controller, global_store
from rl_ide.export.exporter import PolicyExporter
from rl_ide.llm.client import LLMConfig, create_llm_client
from rl_ide.state.conversation_store import ConversationStore
from rl_ide.synthesizer.env_builder import (
    EnvironmentSynthesisError,
    JITEnvironmentSynthesizer,
    synthesize_universal_env,
)
from rl_ide.synthesizer.verifier import SandboxSmokeTester


@pytest.fixture
def client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# Transport: provider clients must send real bearer credentials
# -----------------------------------------------------------------------------
def test_llm_clients_send_bearer_authorization(monkeypatch):
    captured: list[dict] = []

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "pong"}}], "content": [{"text": "pong"}]}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            captured.append({"url": url, "headers": headers or {}})
            return _FakeResponse()

    import rl_ide.llm.client as client_mod

    monkeypatch.setattr(client_mod.httpx, "Client", _FakeClient)

    for provider, key in [("deepseek", "sk-test-deepseek"), ("openai", "sk-test-openai"), ("ollama", "")]:
        captured.clear()
        client = create_llm_client(
            LLMConfig(provider=provider, api_key=key or "ollama", model_name="test-model")
        )
        client.generate("hello")
        assert captured, f"{provider} made no HTTP call"
        auth = captured[-1]["headers"].get("Authorization", "")
        assert auth.startswith("Bearer "), f"{provider} must send a bearer token, got {auth!r}"
        if key:
            assert auth == f"Bearer {key}"


# -----------------------------------------------------------------------------
# Isolation: live inference is refused while training mutates the session
# -----------------------------------------------------------------------------
def test_interact_rejected_while_training(client: TestClient):
    session_id = f"test_busy_{uuid.uuid4().hex[:6]}"
    lock = global_controller.session_lock(session_id)
    assert lock.acquire(blocking=False)
    try:
        resp = client.post(f"/api/v4/sessions/{session_id}/interact", json={"user_action": 1})
        assert resp.status_code == 409
        assert "training" in resp.json()["detail"].lower()
    finally:
        lock.release()


# -----------------------------------------------------------------------------
# Export: a bundle cannot be "successful" while missing required artefacts
# -----------------------------------------------------------------------------
def test_export_requires_artefacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        PolicyExporter.package_runbundle(
            session_id="missing",
            blueprint=None,
            model=None,
            env_module_path=tmp_path / "nope_env.py",
            arena_html_path=tmp_path / "nope_arena.html",
            output_dir=tmp_path / "bundles",
        )


# -----------------------------------------------------------------------------
# Cleanup: clear_all only removes conversation-owned directories
# -----------------------------------------------------------------------------
def test_clear_all_is_scoped(tmp_path):
    store = ConversationStore(db_path=tmp_path / "db.sqlite", workspaces_dir=tmp_path / "workspaces")
    conv = store.create_conversation(title="owned")
    owned = tmp_path / "workspaces" / conv["conversation_id"]
    owned.mkdir(parents=True, exist_ok=True)
    unrelated = tmp_path / "workspaces" / "not_a_conversation"
    unrelated.mkdir(parents=True, exist_ok=True)

    deleted = store.clear_all()
    assert deleted == 1
    assert not owned.exists()
    assert unrelated.exists(), "clear_all must not delete directories it does not own"


# -----------------------------------------------------------------------------
# Durability: dataset records survive for resumed sessions
# -----------------------------------------------------------------------------
def test_dataset_records_persisted_for_resume(tmp_path):
    store = ConversationStore(db_path=tmp_path / "db.sqlite", workspaces_dir=tmp_path / "workspaces")
    agent = StudioOmniAgent(conv_store=store)
    conv_id = f"test_records_{uuid.uuid4().hex[:6]}"

    from rl_ide.ingest import load_bytes

    dataset = load_bytes(
        filename="audit.csv",
        content=b"amount,risk,label\n10,0.1,0\n20,0.9,1\n30,0.2,0\n40,0.8,1\n",
    )
    resp = agent.process_message(conv_id, "Train an auditor on this table", dataset=dataset)
    assert resp.status in ("arena_preview", "synthesis_failed")

    records_file = store.workspaces_dir / conv_id / "dataset_records.json"
    assert records_file.exists()
    from rl_ide.state.records_store import load_records

    records = load_records(store.workspaces_dir / conv_id)
    assert records and records[0]["label"] in (0, 1)
    # Label leakage guard: the target column must not be an observation.
    assert "label" not in resp.blueprint_spec.observation_features


# -----------------------------------------------------------------------------
# Intake answers apply mid-conversation (not only on the first turn)
# -----------------------------------------------------------------------------
def test_intake_answers_applied_mid_session(tmp_path):
    store = ConversationStore(db_path=tmp_path / "db.sqlite", workspaces_dir=tmp_path / "workspaces")
    agent = StudioOmniAgent(conv_store=store)
    conv_id = f"test_answers_{uuid.uuid4().hex[:6]}"
    assert agent.process_message(conv_id, "Train an agent on a 3x3 board").status == "arena_preview"

    response = agent.process_message(
        conv_id,
        "Use these preferences",
        selected_options={
            "q_arena_ui": "Minimalist dark-mode layout focused on the main metric",
            "q_expectations": "95% mastery (5 rounds)",
        },
    )
    assert response.blueprint_spec is not None
    assert response.blueprint_spec.arena_visual_description.startswith("Minimalist")
    assert response.blueprint_spec.target_metric_threshold == 95.0
    assert response.blueprint_spec.max_auto_tune_rounds == 5

    saved = store.get_conversation(conv_id)
    assert saved["blueprint_spec"]["target_metric_threshold"] == 95.0


# -----------------------------------------------------------------------------
# Verification gating: unverified code is never written or imported
# -----------------------------------------------------------------------------
def test_unverified_environment_is_not_written(tmp_path, monkeypatch):
    blueprint = IntakeArchitectAgent().interview(
        "Train an agent on a 3x3 board", force_smart_defaults=True
    ).blueprint_spec

    def _never_verifies(self, *args, **kwargs):
        return False, "broken", ["synthetic failure"]

    monkeypatch.setattr(SandboxSmokeTester, "verify_and_heal", _never_verifies)
    synthesizer = JITEnvironmentSynthesizer()
    target = tmp_path / "session"
    with pytest.raises(EnvironmentSynthesisError):
        synthesizer.synthesize(blueprint=blueprint, workspace_dir=target)
    assert not (target / "env_module.py").exists()


# -----------------------------------------------------------------------------
# Auto-tuner: unverified output is never persisted or loaded
# -----------------------------------------------------------------------------
def test_auto_tune_failure_keeps_previous_code(tmp_path):
    from rl_ide.autopilot.auto_tuner import TuningResult

    store = SessionStateStore(base_dir=tmp_path / "workspaces")
    controller = AutonomousLoopController(store=store)
    blueprint = IntakeArchitectAgent().interview(
        "Train an agent on a 3x3 board", force_smart_defaults=True
    ).blueprint_spec

    _, env_path, env_cls = JITEnvironmentSynthesizer().synthesize(
        blueprint=blueprint, workspace_dir=tmp_path / "workspaces" / "sess_tune"
    )
    original_code = env_path.read_text(encoding="utf-8")

    def _bad_tune(**kwargs):
        return TuningResult(
            round=kwargs["round_num"],
            current_metric=0.0,
            target_threshold=blueprint.target_metric_threshold,
            diagnosis="synthetic",
            changes_summary="synthetic",
            updated_code=original_code + "\nthis is not python",
            verified=False,
            applied=True,
        )

    controller.auto_tuner.__dict__  # noqa: B018 - ensure attribute exists
    controller._auto_tuner = type("T", (), {"tune": staticmethod(_bad_tune)})()
    controller.auto_tuner  # noqa: B018

    events: list[dict] = []
    session = controller.run_loop(
        session_id="sess_tune",
        blueprint=blueprint,
        env_cls=env_cls,
        max_rounds=3,
        episodes_per_burst=2,
        on_event_callback=events.append,
    )
    assert any(e.get("event") == "auto_tune_failed" for e in events)
    assert session.env_module_code == original_code
    assert env_path.read_text(encoding="utf-8") == original_code


# -----------------------------------------------------------------------------
# Target parsing must not hijack ordinary percentages
# -----------------------------------------------------------------------------
def test_percentage_parsing_only_for_explicit_targets():
    from rl_ide.autopilot.loop_controller import parse_nudge_meta_command

    assert parse_nudge_meta_command("make the discount 20% lower") is None
    assert parse_nudge_meta_command("target 90% mastery")["target"] == 90.0
    assert parse_nudge_meta_command("reach 95%")["target"] == 95.0
    assert parse_nudge_meta_command("train for 3 more rounds")["rounds"] == 3


# -----------------------------------------------------------------------------
# Reward shaping must preserve the underlying training signal (never zero it)
# -----------------------------------------------------------------------------
def test_dataset_reward_shaping_preserves_base_signal(tmp_path):
    from rl_ide.ingest import load_bytes
    from rl_ide.synthesizer.env_builder import instantiate_env, load_synthesized_env_class

    store = ConversationStore(db_path=tmp_path / "db.sqlite", workspaces_dir=tmp_path / "workspaces")
    agent = StudioOmniAgent(conv_store=store)
    conv_id = f"test_reward_{uuid.uuid4().hex[:6]}"
    dataset = load_bytes(
        filename="audit.csv",
        content=b"amount,risk,label\n10,0.1,0\n20,0.9,1\n30,0.2,0\n40,0.8,1\n",
    )
    assert agent.process_message(conv_id, "Train an auditor on this table", dataset=dataset).status == "arena_preview"

    response = agent.process_message(conv_id, "Penalize mistakes more heavily")
    assert "updated" in response.assistant_message.lower()

    env_path = store.workspaces_dir / conv_id / "env_module.py"
    code = env_path.read_text(encoding="utf-8")
    assert "mistake_penalty" in code and "accuracy_reward" in code

    env = instantiate_env(load_synthesized_env_class(env_path), dataset_records=dataset.df.to_dict(orient="records"))
    env.reset()
    _, reward, _, _, info = env.step(0)
    # Base correctness signal (±1) must still be present alongside the shaped term.
    assert "accuracy_reward" in info["reward_breakdown"]
    assert reward != 0.0, "Reward shaping must not zero out the original training signal"
