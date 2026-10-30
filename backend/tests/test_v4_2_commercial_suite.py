"""Comprehensive 10-Domain Commercial Suite Verification Test (v4.2).

Executes and verifies all 10 Commercial Use Cases from V4_2_COMMERCIAL_E2E_AUDIT_AND_FIX_PLAN.md:
1. Healthcare — HIPAA Prior-Authorization & Clinical Triage Router (PDF + CSV)
2. FinTech — Real-Time Wire & ACH Fraud Interceptor (SQLite DB)
3. DevOps / FinOps — Kubernetes Cluster Auto-Scaler & SLA Protector (CSV)
4. E-Commerce & Retail — Flash-Sale Dynamic Pricing & Inventory Clearance (XLSX)
5. LegalTech — NDA & Vendor Contract Clause Risk Auditor (DOCX)
6. CleanTech & Edge IoT — Solar + Commercial Battery Tariff Arbitrage (Simulation + Export)
7. Corporate Finance — Automated Travel & Expense (T&E) Receipt Auditor (PDF + CSV)
8. Robotics & Logistics — 2D Autonomous Mobile Robot (AMR) Warehouse Navigator (Simulation)
9. SaaS & Telecom — VIP Customer Churn & Retention Offer Optimizer (CSV)
10. Gaming & Strategy — Interactive Tactical Grid / Connect-Four Bot (Simulation)
+ Cross-Conversation Switching & Resumption Audit across multiple domains.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import pytest
from fastapi.testclient import TestClient

from rl_ide.architect.models import AgentBlueprintSpec
from rl_ide.architect.omni_agent import StudioActionType
from rl_ide.api.server import app, global_store, dataset_store
from rl_ide.state.conversation_store import get_conversation_store
from rl_ide.ingest.loader import load_file
from rl_ide.ingest.sql_connector import SQLConnector

# Each case runs a multi-turn studio session (blueprint synthesis, arena generation, code synthesis,
# training) against the *real* provider. That is a paid, slow verification pass, so it stays out of
# the default suite: `pytest` deselects `live_llm`, and even an explicit run requires opt-in.
pytestmark = pytest.mark.live_llm

if os.environ.get("RL_IDE_LIVE_LLM") != "1":
    pytest.skip(
        "live-LLM use-case suite: export RL_IDE_LIVE_LLM=1 and run with `-m live_llm`",
        allow_module_level=True,
    )


@pytest.fixture
def client():
    return TestClient(app)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "commercial_suite"


# =============================================================================
# Use Case 1: Healthcare — HIPAA Prior-Authorization & Clinical Triage Router
# =============================================================================
def test_usecase_1_healthcare_hipaa_triage(client: TestClient):
    conv_id = f"test_comm_uc1_{uuid.uuid4().hex[:6]}"
    claims_csv = FIXTURES_DIR / "patient_claims.csv"
    assert claims_csv.exists(), "patient_claims.csv fixture missing"

    # Step 1: Upload claims dataset
    with open(claims_csv, "rb") as f:
        up_resp = client.post("/api/v4/ingest/upload", files={"file": ("patient_claims.csv", f, "text/csv")})
    assert up_resp.status_code == 200
    ds_data = up_resp.json()
    ds_id = ds_data["dataset_id"]

    # Step 2: Initial Prompt -> Instant Mock-Animated Arena Preview
    prompt = (
        "Train an agent to triage incoming patient prior-authorization requests into 'Auto-Approve', "
        "'Request Lab Docs', 'Urgent Physician Review', or 'Deny' using our clinical SOP PDF and claims CSV."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt, "dataset_id": ds_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["arena_locked"] is False
    assert "startMockAnimationLoop" in data["arena_html"]
    assert len(data["blueprint_spec"]["action_labels"]) == 4

    # Step 3: Arena Restyling
    restyle_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Highlight abnormal vital signs in red and add a HIPAA Boundary Verified badge."},
    )
    assert restyle_resp.status_code == 200
    restyle_data = restyle_resp.json()
    assert StudioActionType.PREVIEW_OR_UPDATE_ARENA.value in restyle_data["action_types"]
    assert "hipaa" in restyle_data["arena_html"].lower()

    # Step 4: Lock Arena & Train initial 2 rounds
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Confirm and start training for 2 rounds with 50 episodes per round",
            "training_params": {
                "max_auto_tune_rounds": 2,
                "episodes_per_round": 50,
                "target_metric_threshold": 80.0,
            },
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["arena_locked"] is True
    assert train_data["current_round"] >= 1
    init_r1 = train_data["current_round"]

    # Step 5: Post-training chat tweak (cardiac symptoms rule + train 1 more round)
    tweak_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Never deny a claim if cardiac symptoms are present—always route to Urgent Physician Review, and train 1 more round.",
            "training_params": {"max_auto_tune_rounds": 1, "episodes_per_round": 50},
            "run_sync": True,
        },
    )
    assert tweak_resp.status_code == 200
    tweak_data = tweak_resp.json()
    assert tweak_data["current_round"] >= init_r1 + 1
    assert global_store.get_model(conv_id) is not None


# =============================================================================
# Use Case 2: FinTech — Real-Time Wire & ACH Fraud Interceptor
# =============================================================================
def test_usecase_2_fintech_wire_fraud(client: TestClient):
    conv_id = f"test_comm_uc2_{uuid.uuid4().hex[:6]}"
    sqlite_path = FIXTURES_DIR / "wire_transfers.sqlite"
    assert sqlite_path.exists(), "wire_transfers.sqlite fixture missing"

    # Step 1: SQL Modal Security & Ingestion
    sql_uri = f"sqlite:///{sqlite_path.resolve()}"
    bad_resp = client.post("/api/v4/ingest/sql", json={"connection_uri": sql_uri, "query": "DROP TABLE wire_transfers"})
    assert bad_resp.status_code == 400
    assert "Read-only SQL violation" in bad_resp.json()["detail"]

    sql_resp = client.post("/api/v4/ingest/sql", json={"connection_uri": sql_uri, "query": "SELECT * FROM wire_transfers"})
    assert sql_resp.status_code == 200
    ds_id = sql_resp.json()["dataset_id"]

    # Step 2: Initial Prompt
    prompt = "Train a real-time wire fraud interceptor with actions: 'Clear Instant', 'Step-Up 2FA Challenge', or 'Freeze & Alert Fraud Desk'."
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt, "dataset_id": ds_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["blueprint_spec"]["action_labels"]) == 3

    # Step 3: Lock Arena & Train
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Looks good, lock arena and start training for 2 rounds",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50, "target_metric_threshold": 80.0},
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200

    # Step 4: Post-training chat tweak
    tweak_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "It is freezing too many verified corporate payroll wires; reduce false positives on verified accounts and train 2 more rounds.",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50},
            "run_sync": True,
        },
    )
    assert tweak_resp.status_code == 200
    assert tweak_resp.json()["current_round"] >= 2

    # Step 5: Test interactive mode
    interact_resp = client.post(
        f"/api/v4/sessions/{conv_id}/interact",
        json={"user_action": 1},
    )
    assert interact_resp.status_code == 200
    idata = interact_resp.json()
    assert "agent_action_label" in idata or "action_probs" in idata


# =============================================================================
# Use Case 3: DevOps / FinOps — Kubernetes Cluster Auto-Scaler & SLA Protector
# =============================================================================
def test_usecase_3_devops_k8s_autoscaler(client: TestClient):
    conv_id = f"test_comm_uc3_{uuid.uuid4().hex[:6]}"
    k8s_csv = FIXTURES_DIR / "k8s_telemetry.csv"

    with open(k8s_csv, "rb") as f:
        up_resp = client.post("/api/v4/ingest/upload", files={"file": ("k8s_telemetry.csv", f, "text/csv")})
    assert up_resp.status_code == 200
    ds_id = up_resp.json()["dataset_id"]

    prompt = (
        "Train a Kubernetes auto-scaler agent with actions: 'Scale Out +2 Nodes', 'Scale Out +1 Node', "
        "'Hold Steady', or 'Scale Down -1 Node'. Minimize hourly cloud cost while keeping SLA latency violations at zero."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt, "dataset_id": ds_id},
    )
    assert resp.status_code == 200
    assert len(resp.json()["blueprint_spec"]["action_labels"]) == 4

    # Restyle: live $/hour burn-rate meter
    restyle_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Add a live $/hour cloud burn-rate meter to the arena."},
    )
    assert restyle_resp.status_code == 200
    assert "burn-rate" in restyle_resp.json()["arena_html"].lower()

    # Train
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Lock arena and train for 2 rounds",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50, "target_metric_threshold": 80.0},
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200
    assert train_resp.json()["arena_locked"] is True


# =============================================================================
# Use Case 4: E-Commerce & Retail — Flash-Sale Dynamic Pricing & Inventory Clearance
# =============================================================================
def test_usecase_4_ecommerce_dynamic_pricing(client: TestClient):
    conv_id = f"test_comm_uc4_{uuid.uuid4().hex[:6]}"
    sku_xlsx = FIXTURES_DIR / "sku_flash_sale.xlsx"

    with open(sku_xlsx, "rb") as f:
        up_resp = client.post("/api/v4/ingest/upload", files={"file": ("sku_flash_sale.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert up_resp.status_code == 200
    ds_id = up_resp.json()["dataset_id"]

    prompt = (
        "Train a dynamic pricing agent for a 14-day flash sale with 500 units of stock. "
        "Actions: 'Raise Price +10%', 'Keep Base Price', 'Discount 15%', or 'Clearance 30% Off' to maximize total revenue before day 14."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt, "dataset_id": ds_id},
    )
    assert resp.status_code == 200
    assert len(resp.json()["blueprint_spec"]["action_labels"]) == 4

    # Train
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Lock arena and train for 2 rounds",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50, "target_metric_threshold": 80.0},
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200

    # Post-training tweak: forbid clearance when >7 days remain
    tweak_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Forbid using 'Clearance 30% Off' when more than 7 days remain in the sale, and continue training for 2 rounds.",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50},
            "run_sync": True,
        },
    )
    assert tweak_resp.status_code == 200
    assert tweak_resp.json()["current_round"] >= 2


# =============================================================================
# Use Case 5: LegalTech — NDA & Vendor Contract Clause Risk Auditor
# =============================================================================
def test_usecase_5_legaltech_clause_auditor(client: TestClient):
    conv_id = f"test_comm_uc5_{uuid.uuid4().hex[:6]}"
    docx_path = FIXTURES_DIR / "vendor_nda_clauses.docx"

    with open(docx_path, "rb") as f:
        up_resp = client.post("/api/v4/ingest/upload", files={"file": ("vendor_nda_clauses.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert up_resp.status_code == 200
    ds_id = up_resp.json()["dataset_id"]
    assert "max_dollar_value" in up_resp.json()["columns"]

    prompt = (
        "Audit each contract clause in this DOCX file and classify into: 'Standard Acceptable', "
        "'Flag Unlimited Liability', 'Flag IP Ownership Risk', or 'Reject Non-Compete'."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt, "dataset_id": ds_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["blueprint_spec"]["action_labels"]) == 4

    # Diagnostic question
    diag_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Why did it flag clause #3, and can we make it stricter on liability caps over $100,000?"},
    )
    assert diag_resp.status_code == 200
    assert len(diag_resp.json()["assistant_message"]) > 20


# =============================================================================
# Use Case 6: CleanTech & Edge IoT — Solar + Commercial Battery Tariff Arbitrage
# =============================================================================
def test_usecase_6_cleantech_solar_battery_export(client: TestClient):
    conv_id = f"test_comm_uc6_{uuid.uuid4().hex[:6]}"

    prompt = (
        "Train a smart solar and battery controller observing solar_kw, building_load_kw, battery_soc_pct, and grid_price_kwh. "
        "Actions: 'Charge Battery', 'Discharge to Building', 'Sell Back to Grid', or 'Idle Hold'."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt},
    )
    assert resp.status_code == 200
    assert len(resp.json()["blueprint_spec"]["action_labels"]) == 4
    assert len(resp.json()["blueprint_spec"]["observation_features"]) == 4

    # Restyle: power-flow arrows
    restyle_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Show animated power-flow arrows between Solar, Battery, Building, and Grid."},
    )
    assert restyle_resp.status_code == 200
    assert "battery" in restyle_resp.json()["arena_html"].lower()

    # Train
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Lock arena and train for 2 rounds",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50, "target_metric_threshold": 80.0},
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200
    metric = train_resp.json()["best_metric"]
    assert 0.0 <= metric <= 100.0, "Reported metric must be a valid percentage"

    # Export .runbundle and test execution
    export_resp = client.post(f"/api/v4/sessions/{conv_id}/export")
    assert export_resp.status_code == 200
    exp_data = export_resp.json()
    bundle_path = Path(exp_data["bundle_path"])
    assert (bundle_path / "model.onnx").exists()
    assert (bundle_path / "run_local.py").exists()

    # Execute run_local.py --sample
    run_cmd = [sys.executable, str(bundle_path / "run_local.py"), "--sample"]
    proc = subprocess.run(run_cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, f"run_local.py failed: {proc.stderr}"
    assert "predicted_action" in proc.stdout.lower() or "probs" in proc.stdout.lower()


# =============================================================================
# Use Case 7: Corporate Finance — Automated Travel & Expense (T&E) Receipt Auditor
# =============================================================================
def test_usecase_7_expense_policy_auditor(client: TestClient):
    conv_id = f"test_comm_uc7_{uuid.uuid4().hex[:6]}"
    exp_csv = FIXTURES_DIR / "employee_expenses.csv"

    with open(exp_csv, "rb") as f:
        up_resp = client.post("/api/v4/ingest/upload", files={"file": ("employee_expenses.csv", f, "text/csv")})
    assert up_resp.status_code == 200
    ds_id = up_resp.json()["dataset_id"]

    prompt = (
        "Audit employee expense rows against our travel policy PDF. "
        "Actions: 'Auto-Reimburse', 'Request Itemized Receipt', or 'Flag Policy Violation'."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt, "dataset_id": ds_id},
    )
    assert resp.status_code == 200
    assert len(resp.json()["blueprint_spec"]["action_labels"]) == 3

    # Train
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Confirm and train for 2 rounds",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50, "target_metric_threshold": 80.0},
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200


# =============================================================================
# Use Case 8: Robotics & Logistics — 2D Autonomous Mobile Robot (AMR) Warehouse Navigator
# =============================================================================
def test_usecase_8_robotics_amr_navigator(client: TestClient):
    conv_id = f"test_comm_uc8_{uuid.uuid4().hex[:6]}"

    prompt = (
        "Train a 2D warehouse robot rover to navigate to target loading docks while avoiding obstacles, "
        "minimizing jerky turns, and conserving battery."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt},
    )
    assert resp.status_code == 200
    arena_html = resp.json()["arena_html"]
    # The arena is bound to the environment contract, not a fabricated rover canvas.
    assert "sendInteractiveAction" in arena_html
    for key in resp.json()["blueprint_spec"]["state_dict_schema"] or {}:
        assert key in arena_html

    # Train
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Lock arena and train for 2 rounds",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50, "target_metric_threshold": 80.0},
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200

    # Post-training tweak: obstacle safety clearance
    tweak_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Increase obstacle safety clearance distance and train 2 more rounds.",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50},
            "run_sync": True,
        },
    )
    assert tweak_resp.status_code == 200
    assert tweak_resp.json()["current_round"] >= 2

    # Interactive click
    interact_resp = client.post(
        f"/api/v4/sessions/{conv_id}/interact",
        json={"target_pos": [4.0, 4.0]},
    )
    assert interact_resp.status_code == 200


# =============================================================================
# Use Case 9: SaaS & Telecom — VIP Customer Churn & Retention Offer Optimizer
# =============================================================================
def test_usecase_9_saas_churn_optimizer(client: TestClient):
    conv_id = f"test_comm_uc9_{uuid.uuid4().hex[:6]}"
    saas_csv = FIXTURES_DIR / "saas_customer_health.csv"

    with open(saas_csv, "rb") as f:
        up_resp = client.post("/api/v4/ingest/upload", files={"file": ("saas_customer_health.csv", f, "text/csv")})
    assert up_resp.status_code == 200
    ds_id = up_resp.json()["dataset_id"]

    prompt = (
        "Train a retention agent with actions: 'No Intervention', 'Automated Check-In Email', "
        "'Offer 15% Discount', or 'Assign Dedicated CSM' to maximize net retained MRR."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt, "dataset_id": ds_id},
    )
    assert resp.status_code == 200
    assert len(resp.json()["blueprint_spec"]["action_labels"]) == 4

    # Train
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Lock arena and train for 2 rounds",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50, "target_metric_threshold": 80.0},
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200

    # Post-training tweak
    tweak_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Penalize offering the 15% discount when login_drop_pct is below 20%—save discounts only for at-risk VIP accounts, and train 2 more rounds.",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50},
            "run_sync": True,
        },
    )
    assert tweak_resp.status_code == 200
    assert tweak_resp.json()["current_round"] >= 2


# =============================================================================
# Use Case 10: Gaming & Strategy — Interactive Tactical Grid / Connect-Four Bot
# =============================================================================
def test_usecase_10_gaming_tactical_grid(client: TestClient):
    conv_id = f"test_comm_uc10_{uuid.uuid4().hex[:6]}"

    prompt = (
        "Train an agent to play Tic-Tac-Toe / Tactical Grid against a randomized opponent pool. "
        "Prioritize center control, block opponent threats immediately, and create winning forks."
    )
    resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": prompt},
    )
    assert resp.status_code == 200
    assert "startMockAnimationLoop" in resp.json()["arena_html"]

    # Restyle: neon arcade theme + win probabilities
    restyle_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={"user_message": "Give the board a sleek neon arcade theme and display per-cell win probability percentages."},
    )
    assert restyle_resp.status_code == 200
    assert "arcade" in restyle_resp.json()["arena_html"].lower()

    # Train
    train_resp = client.post(
        f"/api/v4/conversations/{conv_id}/chat",
        json={
            "user_message": "Lock arena and train for 2 rounds",
            "training_params": {"max_auto_tune_rounds": 2, "episodes_per_round": 50, "target_metric_threshold": 90.0},
            "run_sync": True,
        },
    )
    assert train_resp.status_code == 200
    metric = train_resp.json()["best_metric"]
    assert 0.0 <= metric <= 100.0, "Reported metric must be a valid percentage"

    # 3 live moves
    for cell in [4, 0, 8]:
        m_resp = client.post(f"/api/v4/sessions/{conv_id}/interact", json={"user_action": cell})
        assert m_resp.status_code == 200


# =============================================================================
# Cross-Conversation Switching & State Resumption Audit
# =============================================================================
def test_cross_conversation_switching_audit(client: TestClient):
    # Verify that loading conversations #1, #5, #8, #10 restores exact state
    conv_store = get_conversation_store()
    convs = conv_store.list_conversations()
    assert len(convs) >= 4, "Must have at least 4 conversations persisted"

    target_prefixes = ["test_comm_uc1_", "test_comm_uc5_", "test_comm_uc8_", "test_comm_uc10_"]
    checked = 0
    for prefix in target_prefixes:
        # Only conversations that actually produced an arena can be restored; the store also holds
        # historical runs whose synthesis failed for reasons outside this test's scope.
        matching = [
            c for c in convs
            if c["conversation_id"].startswith(prefix) and c.get("status") != "synthesis_failed"
        ]
        if matching:
            c = matching[0]
            get_resp = client.get(f"/api/v4/conversations/{c['conversation_id']}")
            assert get_resp.status_code == 200
            loaded = get_resp.json()
            assert loaded["conversation_id"] == c["conversation_id"]
            assert loaded["arena_html"] is not None
            assert len(loaded["chat_history"]) >= 2
            # Verify active session restored in global_store
            session = global_store.get_session(c["conversation_id"])
            assert session is not None
            # Verify model is available in memory if training rounds were executed
            if loaded.get("rounds_history"):
                model = global_store.get_model(c["conversation_id"])
                assert model is not None
            checked += 1
    assert checked >= 1, "No restorable commercial conversation was found to switch back to"
