"""Regression tests for Phase P0-B.1: no data leaves, tenants are isolated, disk is bounded.
Covers gap G3 (bundles ship without customer rows unless asked), G4 (owner-scoped conversations and
datasets that survive a restart) and G9 (verification workspaces are temporary, deletes reclaim
pages, deleted conversations take their bundles with them).
"""
from __future__ import annotations
import json
import sqlite3
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from stable_baselines3 import PPO
from rl_ide.api.server import app
from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.autopilot.loop_controller import SessionStateStore
from rl_ide.export.exporter import PolicyExporter
from rl_ide.ingest import load_bytes
from rl_ide.state.conversation_store import ConversationStore
from rl_ide.state.dataset_store import DatasetStore
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer, instantiate_env, load_synthesized_env_class
from rl_ide.synthesizer.verifier import SandboxSmokeTester

# Trains, exports or spawns sandbox workers: part of the full suite, not the push gate.
pytestmark = pytest.mark.slow
FIXTURE = Path(__file__).parent / "fixtures" / "real_world" / "support_tickets.csv"
LABELS = [
    "Route to Tier 1 Support",
    "Route to Senior Support",
    "Route to Retention Specialist",
    "Route to Engineering Escalation",
]
CUSTOMER_EMAIL = "customer-4711@example.com"
def _blueprint() -> AgentBlueprintSpec:
    return AgentBlueprintSpec(
        task_title="Support Triage",
        domain_summary="Route tickets.",
        data_source_type="uploaded_dataset",
        observation_features=["placeholder"],
        action_labels=list(LABELS),
        arena_visual_description="desk",
        interactive_mode_description="ticket",
        success_metric_name="Routing Accuracy (%)",
        target_metric_threshold=90.0,
        label_column="resolution_queue",
    )
def _frame(rows: int = 120) -> pd.DataFrame:
    base = pd.read_csv(FIXTURE).head(rows).copy()
    # A high-cardinality identifier is skipped by feature derivation, so this value can only ever
    # appear in the embedded records - exactly what G3 is about.
    base["ticket_id"] = [f"TCK-{i:05d}" for i in range(len(base))]
    base.loc[0, "ticket_id"] = CUSTOMER_EMAIL
    return base
# -----------------------------------------------------------------------------
# G4 — datasets survive a restart and are owner-scoped
# -----------------------------------------------------------------------------
def test_dataset_is_durable_across_store_instances(tmp_path):
    dataset = load_bytes(filename="tickets.csv", content=_frame().to_csv(index=False).encode())
    store = DatasetStore(db_path=tmp_path / "d.db", files_dir=tmp_path / "files")
    store.save(dataset, owner_id="acme")
    reopened = DatasetStore(db_path=tmp_path / "d.db", files_dir=tmp_path / "files")
    loaded = reopened.get(dataset.dataset_id, owner_id="acme")
    assert loaded is not None, "uploads must survive a restart, not live in a process dict"
    assert len(loaded.df) == len(dataset.df)
    assert list(loaded.df.columns) == list(dataset.df.columns)
    assert loaded.df.iloc[0]["ticket_id"] == CUSTOMER_EMAIL
def test_dataset_is_scoped_to_its_owner(tmp_path):
    dataset = load_bytes(filename="tickets.csv", content=_frame().to_csv(index=False).encode())
    store = DatasetStore(db_path=tmp_path / "d.db", files_dir=tmp_path / "files")
    store.save(dataset, owner_id="acme")
    assert store.get(dataset.dataset_id, owner_id="globex") is None
    assert store.list(owner_id="globex") == []
    assert store.delete(dataset.dataset_id, owner_id="globex") is False
    assert store.get(dataset.dataset_id, owner_id="acme") is not None
    assert store.delete(dataset.dataset_id, owner_id="acme") is True
    assert store.get(dataset.dataset_id, owner_id="acme") is None
    assert not (tmp_path / "files" / f"{dataset.dataset_id}.json.gz").exists()
# -----------------------------------------------------------------------------
# G4 — conversations are owner-scoped
# -----------------------------------------------------------------------------
def test_conversations_are_scoped_and_cannot_be_read_across_owners(tmp_path):
    store = ConversationStore(db_path=tmp_path / "c.db", workspaces_dir=tmp_path / "ws")
    a = store.create_conversation(title="Acme routing", owner_id="acme")
    b = store.create_conversation(title="Globex routing", owner_id="globex")
    assert [c["conversation_id"] for c in store.list_conversations(owner_id="acme")] == [a["conversation_id"]]
    assert [c["conversation_id"] for c in store.list_conversations(owner_id="globex")] == [b["conversation_id"]]
    assert store.get_conversation(a["conversation_id"], owner_id="globex") is None
    assert store.delete(a["conversation_id"], owner_id="globex") is False
    assert store.get_conversation(a["conversation_id"], owner_id="acme") is not None
def test_delete_removes_workspace_bundles_and_reclaims_db_pages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ConversationStore(db_path=tmp_path / "c.db", workspaces_dir=tmp_path / "ws")
    conv = store.create_conversation(title="Doomed", owner_id="acme")
    cid = conv["conversation_id"]
    # Inflate the row the way a real run does (episodes are stored as one big JSON blob).
    store.save_or_update(cid, episodes=[{"blob": "x" * 5000}] * 200)
    (tmp_path / "ws" / cid / "model.zip").write_bytes(b"0" * 1024)
    bundle = tmp_path / "exported_bundles" / f"{cid}.runbundle"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_bytes(b"0" * 64)
    assert store.delete(cid, owner_id="acme") is True
    assert not (tmp_path / "ws" / cid).exists()
    assert not bundle.exists(), "deleting a conversation must take its exported bundle with it"
    with sqlite3.connect(tmp_path / "c.db") as conn:
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0, "delete must VACUUM"
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
def test_api_hides_other_owners_conversations():
    client = TestClient(app)
    created = client.post(
        "/api/v4/conversations",
        json={"title": "acme private"},
        headers={"x-owner-id": "acme"},
    ).json()
    cid = created["conversation_id"]
    try:
        assert client.get("/api/v4/conversations", headers={"x-owner-id": "acme"}).json()
        assert client.get("/api/v4/conversations", headers={"x-owner-id": "globex"}).json() == []
        assert client.get(f"/api/v4/conversations/{cid}", headers={"x-owner-id": "globex"}).status_code == 404
        assert (
            client.delete(f"/api/v4/conversations/{cid}", headers={"x-owner-id": "globex"}).status_code == 404
        )
    finally:
        client.delete(f"/api/v4/conversations/{cid}", headers={"x-owner-id": "acme"})
# -----------------------------------------------------------------------------
# G3 — bundles do not ship customer rows unless explicitly asked
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    frame = _frame()
    dataset = load_bytes(filename="tickets.csv", content=frame.to_csv(index=False).encode())
    builder = JITEnvironmentSynthesizer.__new__(JITEnvironmentSynthesizer)
    builder._llm_client = None
    code, _ = builder._build_dataset_code(_blueprint(), dataset)
    work = tmp_path_factory.mktemp("bundle_src")
    env_path = work / "env_module.py"
    env_path.write_text(code, encoding="utf-8")
    arena_path = work / "arena.html"
    arena_path.write_text("<html><body>arena</body></html>", encoding="utf-8")
    assert CUSTOMER_EMAIL in env_path.read_text(encoding="utf-8")
    env = instantiate_env(load_synthesized_env_class(env_path), dataset_records=frame.to_dict(orient="records"))
    model = PPO("MlpPolicy", env, device="cpu", verbose=0)
    out = tmp_path_factory.mktemp("bundles")
    plain, _, _ = PolicyExporter.package_runbundle(
        session_id="plain",
        blueprint=_blueprint(),
        model=model,
        env_module_path=env_path,
        arena_html_path=arena_path,
        output_dir=out,
        scorecard={"held_out": True, "policy": {"value_pct": 90.0}},
    )
    with_data, _, _ = PolicyExporter.package_runbundle(
        session_id="withdata",
        blueprint=_blueprint(),
        model=model,
        env_module_path=env_path,
        arena_html_path=arena_path,
        output_dir=out,
        scorecard={"held_out": True, "policy": {"value_pct": 90.0}},
        include_data=True,
    )
    return plain, with_data
def test_default_bundle_contains_no_customer_rows(packaged):
    plain, _ = packaged
    shipped = (plain / "env_module.py").read_text(encoding="utf-8")
    assert CUSTOMER_EMAIL not in shipped
    assert "_records = []" in shipped
    assert (plain / "data" / "README.md").exists()
    scorecard = json.loads((plain / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["deployment"]["contains_training_data"] is False
def test_include_data_bundle_is_self_contained(packaged):
    _, with_data = packaged
    shipped = (with_data / "env_module.py").read_text(encoding="utf-8")
    assert CUSTOMER_EMAIL in shipped
    scorecard = json.loads((with_data / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard["deployment"]["contains_training_data"] is True
# -----------------------------------------------------------------------------
# G9 — verification workspaces are temporary
# -----------------------------------------------------------------------------
def test_verification_leaves_no_workspace_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = load_bytes(filename="tickets.csv", content=_frame().to_csv(index=False).encode())
    builder = JITEnvironmentSynthesizer.__new__(JITEnvironmentSynthesizer)
    builder._llm_client = None
    code, _ = builder._build_dataset_code(_blueprint(), dataset)
    # The pipeline rewrites the blueprint with the derived features before verifying; an empty
    # declaration here means "do not assert a specific observation width".
    blueprint = _blueprint()
    blueprint.observation_features = []
    before = len(list(Path(tempfile.gettempdir()).glob("rl_verify_*")))
    ok, _verified_code, repair_logs = SandboxSmokeTester().verify_and_heal(code, blueprint, dataset)
    after = len(list(Path(tempfile.gettempdir()).glob("rl_verify_*")))
    assert ok is True, str(repair_logs)[-800:]
    assert after <= before, "verification must not leak a temporary workspace"
    assert not list((tmp_path / "workspaces").glob("temp_verify_*"))

def test_hosted_chat_owns_the_conversation_it_creates():
    """A turn from tenant 'acme' must create acme's conversation, not the local user's.

    Found live: the omni agent created every conversation with the default owner, so a hosted
    tenant's own conversation 404'd for itself while remaining readable by `local`.
    """
    import uuid

    client = TestClient(app)
    conv_id = f"tenant_chat_{uuid.uuid4().hex[:8]}"

    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Train an agent to triage support tickets"},
        headers={"x-owner-id": "acme"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["conversation_id"] == conv_id

    assert client.get(f"/api/v4/conversations/{conv_id}", headers={"x-owner-id": "acme"}).status_code == 200
    assert client.get(f"/api/v4/conversations/{conv_id}", headers={"x-owner-id": "globex"}).status_code == 404
    assert client.get(f"/api/v4/conversations/{conv_id}").status_code == 404, "the local user must not see it"

    usage = client.get(
        "/api/v4/usage", params={"conversation_id": conv_id}, headers={"x-owner-id": "acme"}
    )
    assert usage.status_code == 200
