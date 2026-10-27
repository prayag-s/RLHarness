"""Verification Suite for Phase 3C: Vibe Mode Autopilot Orchestrator & FastAPI Endpoints."""

from __future__ import annotations
import io
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import docx

from rl_ide.api.server import create_app
from rl_ide.vibe.translator import dsl_to_plain_rules, plain_rule_to_dsl_diff
from rl_ide.vibe.orchestrator import compute_skill_summary


@pytest.fixture
def test_client():
    app = create_app()
    return TestClient(app)


def test_1_universal_ingest_upload_and_sql_endpoints(tmp_path: Path, test_client: TestClient):
    """Test 1: Multipart file upload (/api/ingest/upload) for CSV, PDF, DOCX and SQL database query (/api/ingest/sql)."""
    # 1. Test CSV file upload
    csv_content = "id,revenue,cost,margin,flagged\n"
    for i in range(50):
        csv_content += f"{i},{100 + i * 5},{40 + i * 2},{(60 + i * 3) / 100.0},{1 if i % 10 == 0 else 0}\n"

    csv_file = io.BytesIO(csv_content.encode("utf-8"))
    res_csv = test_client.post(
        "/api/ingest/upload",
        files={"file": ("financials.csv", csv_file, "text/csv")},
        data={"role": "tabular"},
    )
    assert res_csv.status_code == 200
    data_csv = res_csv.json()
    assert data_csv["dataset_id"].startswith("ds_")
    assert data_csv["format"] == "csv"
    assert data_csv["total_rows"] == 50
    assert "revenue" in data_csv["columns"]
    assert len(data_csv["sample_records"]) > 0

    # 2. Test PDF policy upload with reportlab
    pdf_path = tmp_path / "procurement_policy.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, "Procurement Compliance Policy")
    c.setFont("Helvetica", 10)
    c.drawString(72, 720, "Invoices exceeding $5,000 must include an approved purchase order.")
    c.drawString(72, 700, "Transactions without valid PO will be rejected by audit.")
    c.drawString(72, 680, "Escalate high risk transactions immediately.")
    c.showPage()
    c.save()

    with open(pdf_path, "rb") as f:
        res_pdf = test_client.post(
            "/api/ingest/upload",
            files={"file": ("procurement_policy.pdf", f, "application/pdf")},
            data={"role": "document_records"},
        )
    assert res_pdf.status_code == 200
    data_pdf = res_pdf.json()
    assert data_pdf["format"] == "pdf"
    assert data_pdf["role"] == "document_records"
    assert len(data_pdf["extracted_rules"]) > 0
    rule_descs = [r["description"].lower() for r in data_pdf["extracted_rules"]]
    assert any("po" in desc or "5,000" in desc or "invoice" in desc for desc in rule_descs)

    # 3. Test DOCX upload with python-docx
    docx_path = tmp_path / "audit_manual.docx"
    doc = docx.Document()
    doc.add_heading("Internal Audit Manual", 0)
    doc.add_paragraph("Mandatory: Invoices above $5,000 require senior authorization.")
    doc.add_paragraph("Strictly reject any invoice lacking a valid PO.")
    doc.save(str(docx_path))

    with open(docx_path, "rb") as f:
        res_docx = test_client.post(
            "/api/ingest/upload",
            files={"file": ("audit_manual.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            data={"role": "document_records"},
        )
    assert res_docx.status_code == 200
    data_docx = res_docx.json()
    assert data_docx["format"] == "docx"
    assert len(data_docx["extracted_rules"]) > 0

    # 4. Test Live SQL endpoint with read-only validation
    db_file = tmp_path / "test_sql.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, total_amount REAL, customer_status TEXT)")
    for i in range(40):
        cur.execute("INSERT INTO orders VALUES (?, ?, ?)", (i, float(150 + i * 10), "VIP" if i % 5 == 0 else "REGULAR"))
    conn.commit()
    conn.close()

    uri = f"sqlite:///{db_file.resolve()}"
    res_sql = test_client.post(
        "/api/ingest/sql",
        json={
            "connection_uri": uri,
            "query_or_table": "SELECT * FROM orders WHERE total_amount > 200",
            "limit": 1000,
        },
    )
    assert res_sql.status_code == 200
    data_sql = res_sql.json()
    assert data_sql["format"] == "sql"
    assert data_sql["total_rows"] > 0
    assert "total_amount" in data_sql["columns"]

    # Test read-only guardrail strictly blocks write queries
    res_blocked = test_client.post(
        "/api/ingest/sql",
        json={
            "connection_uri": uri,
            "query_or_table": "DROP TABLE orders",
        },
    )
    assert res_blocked.status_code in (400, 422, 500) or "violation" in res_blocked.text.lower()


def test_2_vibe_lifecycle_tictactoe(test_client: TestClient):
    """Test 2: Tic-Tac-Toe Vibe create, interactive sparring, and conversational coaching."""
    # 1. POST /api/vibe/create
    create_payload = {
        "prompt": "Train an agent to play tic-tac-toe, take center when open, block opponent wins, and win.",
        "burst_episodes": 10,
    }
    res_create = test_client.post("/api/vibe/create", json=create_payload)
    assert res_create.status_code == 200, res_create.text
    vibe_data = res_create.json()

    session_id = vibe_data["session_id"]
    assert session_id.startswith("vibe_")
    assert vibe_data["ui_mode"] == "board_grid"
    assert vibe_data["mastery_score"] >= 80
    assert vibe_data["status_badge"] in ("Mastered", "Proficient")
    assert vibe_data["win_rate"] >= 0.80

    # Verify plain-English rule cards
    rules = vibe_data["rules"]
    assert len(rules) >= 2
    rule_titles = [r["title"].lower() for r in rules]
    assert any("center" in t for t in rule_titles)
    assert any("block" in t or "win" in t for t in rule_titles)

    # 2. POST /api/sim/interact (Human vs Agent live play)
    # Human plays top-left corner (cell 0)
    interact_payload = {
        "session_id": session_id,
        "board_state": [0] * 9,
        "human_cell_idx": 0,
    }
    res_interact = test_client.post("/api/sim/interact", json=interact_payload)
    assert res_interact.status_code == 200
    interact_data = res_interact.json()
    assert interact_data["ui_mode"] == "board_grid"
    board = interact_data["board"]
    assert board[0] == -1  # Human move
    agent_cell = interact_data["agent_cell"]
    assert agent_cell != -1
    assert board[agent_cell] == +1  # Agent counter-move
    assert len(interact_data["action_probabilities"]) == 9

    # 3. POST /api/vibe/coach: Thumbs up golden lock
    res_lock = test_client.post(
        "/api/vibe/coach",
        json={
            "session_id": session_id,
            "feedback_type": "thumbs_up",
            "target_episode_id": 1,
        },
    )
    assert res_lock.status_code == 200
    lock_data = res_lock.json()
    assert "locked" in lock_data["coach_reply"].lower()

    # 4. POST /api/vibe/coach: 1-click rule card priority adjustment
    center_rule = next(r for r in rules if "center" in r["title"].lower())
    res_boost = test_client.post(
        "/api/vibe/coach",
        json={
            "session_id": session_id,
            "term_name": center_rule["id"],
            "new_importance": "Must Do",
            "burst_episodes": 5,
        },
    )
    assert res_boost.status_code == 200
    boost_data = res_boost.json()
    updated_center = next(r for r in boost_data["rules"] if r["id"] == center_rule["id"])
    assert updated_center["importance"] == "Must Do"

    # 5. POST /api/vibe/coach: Conversational critique
    res_critique = test_client.post(
        "/api/vibe/coach",
        json={
            "session_id": session_id,
            "critique_text": "Block opponent threats aggressively and seize winning moves immediately.",
            "burst_episodes": 5,
        },
    )
    assert res_critique.status_code == 200
    critique_data = res_critique.json()
    assert len(critique_data["applied_diff"]) >= 1
    assert critique_data["after_mastery"] >= 80


def test_3_vibe_lifecycle_robot_rover(test_client: TestClient):
    """Test 3: 2D Robot Rover Vibe create, interactive obstacle dragging, and obstacle avoidance coaching."""
    # 1. POST /api/vibe/create
    create_payload = {
        "prompt": "Train a robot rover to reach the target waypoint quickly while avoiding obstacles and minimizing jerk.",
        "burst_episodes": 5,
    }
    res_create = test_client.post("/api/vibe/create", json=create_payload)
    assert res_create.status_code == 200, res_create.text
    vibe_data = res_create.json()

    session_id = vibe_data["session_id"]
    assert session_id.startswith("vibe_")
    assert vibe_data["ui_mode"] == "robot_canvas"
    assert len(vibe_data["rules"]) >= 2
    rule_titles = [r["title"].lower() for r in vibe_data["rules"]]
    assert any("beacon" in t or "reach" in t or "target" in t for t in rule_titles)
    assert any("obstacle" in t or "clearance" in t for t in rule_titles)

    # 2. POST /api/sim/interact (Dragging Target & Placing Obstacles)
    interact_payload = {
        "session_id": session_id,
        "robot_state": {"x": 15.0, "y": 15.0, "vx": 0.0, "vy": 0.0, "battery": 100.0},
        "target_xy": [45.0, 55.0],
        "obstacles_list": [
            {"x": 28.0, "y": 28.0, "radius": 7.0},
            {"x": 60.0, "y": 60.0, "radius": 9.0},
        ],
        "n_frames": 4,
    }
    res_interact = test_client.post("/api/sim/interact", json=interact_payload)
    assert res_interact.status_code == 200
    interact_data = res_interact.json()
    assert interact_data["ui_mode"] == "robot_canvas"
    assert len(interact_data["trajectory"]) == 5  # start + 4 frames
    assert interact_data["target"]["x"] == 45.0
    assert interact_data["target"]["y"] == 55.0
    assert "robot" in interact_data

    # 3. POST /api/vibe/coach: Coaching to increase obstacle standoff distance
    res_coach = test_client.post(
        "/api/vibe/coach",
        json={
            "session_id": session_id,
            "critique_text": "Stay further away from obstacles and avoid collisions",
            "burst_episodes": 5,
        },
    )
    assert res_coach.status_code == 200
    coach_data = res_coach.json()
    assert len(coach_data["applied_diff"]) >= 1
    diff_names = [d["term_name"] for d in coach_data["applied_diff"]]
    assert any("obstacle" in name or "collision" in name for name in diff_names)
    assert "obstacles" in coach_data["coach_reply"].lower()
