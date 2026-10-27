"""Verification Suite for Phase 3A: Universal File & Live SQL Ingestor + AutoDataAdapter."""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rl_ide.contracts.adapter import DomainAdapter
from rl_ide.adapters.auto_data_adapter import AutoDataAdapter
from rl_ide.ingest.sql_connector import (
    SQLConnector,
    validate_read_only_sql,
    test_sql_connection as check_sql_conn,
    list_sql_tables,
    load_sql_dataset,
)
from rl_ide.ingest.profiler import (
    SchemaProfiler,
    DocumentRuleExtractor,
    AutoFeatureVectorizer,
)
from rl_ide.ingest.loader import (
    load_file,
    load_bytes,
    IngestedDataset,
)
from rl_ide.compiler.nl_to_config import NLToConfigCompiler
from rl_ide.trainer.runner import TrainingRunner
from rl_ide.export.exporter import PolicyExporter


def test_1_ingest_csv_xlsx_and_train_export(tmp_path: Path):
    """Test 1: Ingest CSV and XLSX into AutoDataAdapter, compile DSL, run warm-start + burst, export ONNX."""
    n_rows = 120
    rng = np.random.default_rng(42)

    amounts = rng.uniform(50.0, 15000.0, size=n_rows)
    risk_scores = rng.uniform(0.0, 1.0, size=n_rows)
    has_pos = rng.choice([True, False], size=n_rows, p=[0.7, 0.3])
    departments = rng.choice(["sales", "engineering", "finance", "hr"], size=n_rows)
    # Ground truth: flag if amount > 5000 or risk > 0.8
    targets = [
        "flag" if (amt > 5000.0 or risk > 0.8) else "pass"
        for amt, risk in zip(amounts, risk_scores)
    ]

    sample_dict = {
        "transaction_id": [f"TX_{i:04d}" for i in range(n_rows)],
        "amount": amounts,
        "risk_score": risk_scores,
        "has_po": has_pos,
        "department": departments,
        "target": targets,
    }
    df = pd.DataFrame(sample_dict)

    # 1. Save as CSV and XLSX
    csv_file = tmp_path / "transactions.csv"
    xlsx_file = tmp_path / "transactions.xlsx"
    df.to_csv(csv_file, index=False)
    df.to_excel(xlsx_file, index=False, engine="openpyxl")

    # Ingest CSV
    csv_dataset = load_file(csv_file, target_col="target")
    assert isinstance(csv_dataset, IngestedDataset)
    assert csv_dataset.format == "csv"
    assert len(csv_dataset.df) == n_rows
    assert csv_dataset.profile is not None
    assert "amount" in csv_dataset.profile.feature_columns

    # Ingest XLSX
    xlsx_dataset = load_file(xlsx_file, target_col="target")
    assert xlsx_dataset.format == "xlsx"
    assert len(xlsx_dataset.df) == n_rows

    # 2. Synthesize AutoDataAdapter
    adapter = AutoDataAdapter(
        dataset=csv_dataset,
        action_labels=["flag", "pass"],
        episode_length=20,
    )
    assert isinstance(adapter, DomainAdapter)

    obs_spec = adapter.observations()
    action_spec = adapter.actions()
    schema = adapter.event_schema()

    assert obs_spec.vector_dim > 0
    assert action_spec.labels == ["flag", "pass"]
    assert "obs.amount" in schema
    assert "obs.risk_score" in schema
    assert "obs.has_po" in schema
    assert "action.label" in schema
    assert "truth.has_label" in schema

    # 3. Compile natural language prompt into TaskConfig
    compiler = NLToConfigCompiler()
    task_config = compiler.compile(
        user_prompt="Audit transactions: flag high value amounts and penalize unapproved transactions.",
        adapter_name="AutoDataAdapter",
        adapter=adapter,
        task_id="csv_audit_task",
    )
    assert task_config.adapter_name == "AutoDataAdapter"
    assert len(task_config.reward_dsl.reward.terms) >= 1

    # 4. Train with TrainingRunner
    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=task_config.reward_dsl,
        task_id="csv_audit_task",
    )
    # Warm-start
    runner.warmstart_policy(epochs=5, n_samples=50)
    # Burst
    burst_logs = runner.run_burst(n_episodes=5)
    assert len(burst_logs) >= 5
    assert len(runner.buffer) >= 5

    # Check telemetry frame
    step_res = adapter.step(0)
    frame = adapter.render_frame(
        episode_id=1,
        step_result=step_res,
        total_reward=10.0,
        breakdown={"accurate_classification_match": 5.0},
    )
    assert frame["ui_mode"] == "tabular_inspector"
    assert "record" in frame
    assert "obs" in frame

    # 5. Export to ONNX with numerical parity check (< 1e-5)
    onnx_file = tmp_path / "model.onnx"
    obs_dim = adapter.observations().vector_dim
    PolicyExporter.export_onnx(model=runner.model, output_path=onnx_file, obs_dim=obs_dim)
    assert onnx_file.exists()

    max_diff = PolicyExporter.verify_parity(
        model=runner.model,
        onnx_path=onnx_file,
        obs_dim=obs_dim,
        n_samples=32,
    )
    assert max_diff < 1e-5

    # Package RunBundle
    bundle_dir, zip_path, bundle_diff = PolicyExporter.create_runbundle(
        task_config=task_config,
        reward_dsl=task_config.reward_dsl,
        model=runner.model,
        episodes=burst_logs,
        output_dir=tmp_path / "bundles",
    )
    assert Path(bundle_dir).exists()
    assert Path(zip_path).exists()
    assert (Path(bundle_dir) / "run_local.py").exists()


def test_2_sqlite_sqlalchemy_readonly_guardrails_and_query_burst(tmp_path: Path):
    """Test 2: SQLite database connection, read-only SQL validation guardrail, and TrainingRunner burst."""
    db_file = tmp_path / "finance.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            balance REAL,
            credit_score INTEGER,
            is_delinquent INTEGER,
            status TEXT
        )
    """)

    rows = []
    for i in range(100):
        bal = float(500 + i * 150)
        score = int(450 + (i % 40) * 10)
        delinq = 1 if (score < 580 or bal > 12000) else 0
        stat = "delinquent" if delinq else "good_standing"
        rows.append((i + 1, bal, score, delinq, stat))

    cursor.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()

    sqlite_uri = f"sqlite:///{db_file.resolve()}"

    # 1. Test connection and table listing
    conn_info = check_sql_conn(sqlite_uri)
    assert conn_info["connected"] is True
    assert conn_info["dialect"] == "sqlite"
    assert "accounts" in conn_info["tables"]

    tables = list_sql_tables(sqlite_uri)
    assert "accounts" in tables

    # 2. Strict read-only SQL validation guardrails
    # Permitted queries
    validate_read_only_sql("SELECT * FROM accounts WHERE credit_score > 600")
    validate_read_only_sql("WITH high_bal AS (SELECT * FROM accounts WHERE balance > 5000) SELECT * FROM high_bal")

    # Strictly forbidden operations MUST raise ValueError
    with pytest.raises(ValueError, match="Read-only SQL violation"):
        validate_read_only_sql("DROP TABLE accounts")

    with pytest.raises(ValueError, match="Read-only SQL violation"):
        validate_read_only_sql("DELETE FROM accounts WHERE balance < 1000")

    with pytest.raises(ValueError, match="Read-only SQL violation"):
        validate_read_only_sql("UPDATE accounts SET balance = 0")

    with pytest.raises(ValueError, match="Read-only SQL violation"):
        validate_read_only_sql("INSERT INTO accounts VALUES (999, 100.0, 700, 0, 'new')")

    with pytest.raises(ValueError, match="Read-only SQL violation"):
        validate_read_only_sql("ALTER TABLE accounts ADD COLUMN extra TEXT")

    with pytest.raises(ValueError, match="Read-only SQL violation"):
        validate_read_only_sql("SELECT * FROM accounts; DROP TABLE accounts")

    with pytest.raises(ValueError, match="Read-only SQL violation"):
        validate_read_only_sql("-- comment only\n")

    # 3. Load dataset via read-only SQL query
    query = "SELECT account_id, balance, credit_score, is_delinquent, status FROM accounts WHERE credit_score >= 500"
    df = load_sql_dataset(sqlite_uri, query, limit=50)
    assert len(df) == 50
    assert "balance" in df.columns
    assert "credit_score" in df.columns

    # 4. Synthesize AutoDataAdapter and execute training burst
    adapter = AutoDataAdapter(
        dataset=df,
        action_labels=["approve", "reject"],
        episode_length=15,
        target_col="status",
    )
    assert isinstance(adapter, DomainAdapter)
    assert "obs.balance" in adapter.event_schema()
    assert "obs.credit_score" in adapter.event_schema()

    compiler = NLToConfigCompiler()
    task_config = compiler.compile(
        user_prompt="Approve accounts with high credit score and low delinquency.",
        adapter_name="AutoDataAdapter",
        adapter=adapter,
        task_id="sql_account_task",
    )

    runner = TrainingRunner(
        adapter=adapter,
        reward_dsl=task_config.reward_dsl,
        task_id="sql_account_task",
    )
    burst_logs = runner.run_burst(n_episodes=4)
    assert len(burst_logs) == 4
    for log in burst_logs:
        assert len(log.steps) > 0
        assert isinstance(log.total_reward, float)


def test_3_document_pdf_docx_rule_extraction_and_chunk_stepping(tmp_path: Path):
    """Test 3: Document intelligence with PDF and DOCX rule extraction and chunk episode stepping."""
    # 1. Generate multi-page PDF with reportlab
    pdf_path = tmp_path / "procurement_policy.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Page 1: SOP rules
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, "Corporate Procurement Standard Operating Procedures")
    c.setFont("Helvetica", 10)
    c.drawString(72, 720, "All invoices over $5,000 must require PO verification before payment.")
    c.drawString(72, 700, "Approving transactions without authorized PO will be strictly penalized.")
    c.drawString(72, 680, "Escalate any high risk or fraudulent transaction immediately to compliance.")
    c.showPage()

    # Page 2: Section 2 details
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 750, "Invoice Batch 2026-Q1 Summary")
    c.setFont("Helvetica", 10)
    c.drawString(72, 720, "Invoice #9012 for $12,500.00 dated 2026-03-15 pending review.")
    c.drawString(72, 700, "Routine office supply order $450.00 approved.")
    c.showPage()
    c.save()

    # 2. Generate DOCX with python-docx
    docx_path = tmp_path / "audit_guidelines.docx"
    doc = Document()
    doc.add_heading("Financial Auditing Guidelines", 0)
    doc.add_paragraph("Mandatory Rule: Invoices over $5,000 require VP sign-off.")
    doc.add_paragraph("Strictly reject any invoice with missing PO or unauthorized vendor.")
    doc.add_paragraph("Transactions under $500 are eligible for automated clearance.")
    doc.save(str(docx_path))

    # 3. Test DocumentRuleExtractor directly
    sample_text = (
        "Invoices over $5,000 require VP approval.\n"
        "Strictly penalize approving requests without PO.\n"
        "Escalate high risk transactions promptly.\n"
    )
    rules = DocumentRuleExtractor.extract_rules(sample_text, known_columns=["amount", "has_po"])
    assert len(rules) >= 2
    rule_names = [r.name for r in rules]
    assert any("5000" in name for name in rule_names)
    assert any("po" in name for name in rule_names)

    # 4. Ingest PDF and verify extracted rules and chunk records
    pdf_dataset = load_file(pdf_path, role="document_records")
    assert isinstance(pdf_dataset, IngestedDataset)
    assert pdf_dataset.format == "pdf"
    assert len(pdf_dataset.df) == 2  # 2 pages
    assert len(pdf_dataset.extracted_rules) >= 1

    # Check extracted document NLP features
    df_pdf = pdf_dataset.df
    assert "page_number" in df_pdf.columns
    assert "word_count" in df_pdf.columns
    assert "has_currency_amount" in df_pdf.columns
    assert "max_dollar_value" in df_pdf.columns
    assert "risk_keyword_count" in df_pdf.columns
    assert bool(df_pdf.iloc[0]["has_currency_amount"]) is True
    assert float(df_pdf.iloc[0]["max_dollar_value"]) >= 5000.0

    # 5. Ingest DOCX and verify rules
    docx_dataset = load_file(docx_path)
    assert docx_dataset.format == "docx"
    assert len(docx_dataset.extracted_rules) >= 1
    assert len(docx_dataset.df) >= 3

    # 6. Wrap PDF document dataset into AutoDataAdapter
    adapter = AutoDataAdapter(
        dataset=pdf_dataset,
        action_labels=["flag", "approve", "escalate"],
        episode_length=2,
    )
    assert isinstance(adapter, DomainAdapter)
    assert adapter.ui_mode == "document_reader"

    schema = adapter.event_schema()
    assert "obs.page_number" in schema
    assert "obs.word_count" in schema
    assert "obs.has_currency_amount" in schema
    assert "obs.max_dollar_value" in schema
    assert "obs.risk_keyword_count" in schema

    # Step through episode
    obs, info = adapter.reset()
    assert info["ui_mode"] == "document_reader"
    assert len(obs) == adapter.observations().vector_dim

    step_res = adapter.step(action_idx=0)
    assert step_res.events["obs"]["page_number"] == 1
    assert bool(step_res.events["obs"]["has_currency_amount"]) is True
    assert float(step_res.events["obs"]["max_dollar_value"]) >= 5000.0

    frame = adapter.render_frame(
        episode_id=1,
        step_result=step_res,
        total_reward=8.0,
        breakdown={"flag_high_value": 8.0},
    )
    assert frame["ui_mode"] == "document_reader"
    assert "obs" in frame
    assert "record" in frame

    # In-memory bytes loading test
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    mem_dataset = load_bytes("in_memory.pdf", pdf_bytes)
    assert mem_dataset.format == "pdf"
    assert len(mem_dataset.df) == 2
