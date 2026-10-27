"""Universal File Loader for Tabular, Structured, and Unstructured Document Datasets."""

from __future__ import annotations
import io
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from pydantic import BaseModel, Field
import docx
from pypdf import PdfReader

from rl_ide.ingest.profiler import (
    SchemaProfiler,
    DatasetProfile,
    DocumentRuleExtractor,
    CandidateRule,
)
from rl_ide.ingest.sql_connector import SQLConnector


class IngestedDataset(BaseModel):
    """Normalized ingested dataset container ready for AutoDataAdapter synthesis."""
    model_config = {"arbitrary_types_allowed": True}

    dataset_id: str = Field(default_factory=lambda: f"ds_{uuid.uuid4().hex[:10]}")
    name: str
    source_type: str  # "file" | "sql" | "document"
    format: str  # "csv" | "tsv" | "xlsx" | "json" | "jsonl" | "sqlite" | "pdf" | "docx" | "txt" | "md"
    df: pd.DataFrame
    raw_text: str = ""
    extracted_rules: list[CandidateRule] = Field(default_factory=list)
    profile: DatasetProfile | None = None
    role: str = "tabular"  # "tabular" | "document_records" | "knowledge_rules"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _extract_document_features(text: str, page_number: int = 1) -> dict[str, Any]:
    """Extracts NLP and auditing features from a document text section or page."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # Currency extraction
    currency_matches = re.findall(r"(?:\$|USD\s*|\bamount\b\s*:?\s*\$?)[\s]*([0-9,]+(?:\.[0-9]{1,2})?)", text, re.IGNORECASE)
    dollar_values = []
    for m in currency_matches:
        try:
            val = float(m.replace(",", ""))
            dollar_values.append(val)
        except ValueError:
            pass

    has_currency_amount = len(dollar_values) > 0
    max_dollar_value = max(dollar_values) if dollar_values else 0.0

    # Date detection
    has_date = bool(re.search(
        r"(\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b)",
        text,
        re.IGNORECASE,
    ))

    # Risk keywords
    risk_words = [
        "urgent", "delay", "overdue", "penalty", "violation", "risk",
        "fraud", "breach", "audit", "unauthorized", "missing", "escalate"
    ]
    risk_count = 0
    text_lower = text.lower()
    for rw in risk_words:
        risk_count += len(re.findall(rf"\b{rw}\b", text_lower))

    # Sentiment score [-1.0, 1.0]
    pos_words = ["approved", "compliant", "verified", "success", "clear", "authorized", "valid"]
    pos_count = sum(len(re.findall(rf"\b{pw}\b", text_lower)) for pw in pos_words)
    if (pos_count + risk_count) > 0:
        sentiment_score = (pos_count - risk_count) / float(pos_count + risk_count)
    else:
        sentiment_score = 0.0

    return {
        "page_number": int(page_number),
        "word_count": int(word_count),
        "char_count": int(char_count),
        "has_currency_amount": bool(has_currency_amount),
        "max_dollar_value": float(max_dollar_value),
        "has_date": bool(has_date),
        "risk_keyword_count": int(risk_count),
        "sentiment_score": float(sentiment_score),
    }


def load_file(
    path_or_buf: str | Path | io.BytesIO,
    filename: str | None = None,
    role: str = "tabular",
    name: str | None = None,
    target_col: str | None = None,
) -> IngestedDataset:
    """Loads any supported file format into an IngestedDataset."""
    if isinstance(path_or_buf, (str, Path)):
        file_path = Path(path_or_buf)
        fname = filename or file_path.name
        ext = file_path.suffix.lower()
        dataset_name = name or file_path.stem
        is_path = True
    else:
        fname = filename or "dataset.csv"
        ext = os.path.splitext(fname)[1].lower()
        dataset_name = name or os.path.splitext(fname)[0]
        is_path = False

    raw_text = ""
    extracted_rules: list[CandidateRule] = []

    # 1. Tabular Structured Formats
    if ext in (".csv", ".tsv", ".txt") and role == "tabular":
        sep = "\t" if ext == ".tsv" else ","
        if is_path:
            df = pd.read_csv(file_path, sep=sep)
        else:
            df = pd.read_csv(path_or_buf, sep=sep)
        fmt = "tsv" if ext == ".tsv" else "csv"

    elif ext in (".xlsx", ".xls"):
        if is_path:
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            df = pd.read_excel(path_or_buf, engine="openpyxl")
        fmt = "xlsx"

    elif ext in (".json", ".jsonl"):
        if is_path:
            try:
                df = pd.read_json(file_path)
            except ValueError:
                df = pd.read_json(file_path, lines=True)
        else:
            try:
                df = pd.read_json(path_or_buf)
            except ValueError:
                path_or_buf.seek(0)
                df = pd.read_json(path_or_buf, lines=True)
        fmt = "jsonl" if ext == ".jsonl" else "json"

    elif ext in (".sqlite", ".db"):
        fmt = "sqlite"
        if not is_path:
            raise ValueError("SQLite datasets must be provided as a filesystem path.")
        conn = SQLConnector(f"sqlite:///{file_path.resolve()}")
        tables = conn.list_tables()
        if not tables:
            raise ValueError(f"SQLite database at {file_path} contains no tables.")
        target_table = tables[0]
        df = conn.load_dataset(target_table)
        conn.close()

    # 2. PDF Document Format
    elif ext == ".pdf":
        fmt = "pdf"
        reader = PdfReader(file_path if is_path else path_or_buf)
        pages_text: list[str] = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            pages_text.append(txt)

        raw_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

        # Build records per page
        records = []
        for i, p_txt in enumerate(pages_text):
            features = _extract_document_features(p_txt, page_number=i + 1)
            features["text_preview"] = p_txt[:200]
            records.append(features)

        df = pd.DataFrame(records)
        extracted_rules = DocumentRuleExtractor.extract_rules(raw_text, known_columns=list(df.columns))
        if role == "tabular":
            role = "document_records"

    # 3. Word Document Format (.docx)
    elif ext == ".docx":
        fmt = "docx"
        doc = docx.Document(file_path if is_path else path_or_buf)
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        raw_text = "\n\n".join(paras)

        records = []
        for i, para in enumerate(paras):
            features = _extract_document_features(para, page_number=i + 1)
            features["paragraph_preview"] = para[:200]
            records.append(features)

        if not records:
            # Fallback for empty doc
            records.append(_extract_document_features("", page_number=1))

        df = pd.DataFrame(records)
        extracted_rules = DocumentRuleExtractor.extract_rules(raw_text, known_columns=list(df.columns))
        if role == "tabular":
            role = "document_records"

    # 4. Plain Text or Markdown
    elif ext in (".txt", ".md"):
        fmt = "md" if ext == ".md" else "txt"
        if is_path:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        else:
            raw_text = path_or_buf.read().decode("utf-8", errors="replace")

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        records = []
        for i, line in enumerate(lines):
            features = _extract_document_features(line, page_number=i + 1)
            features["line_preview"] = line[:150]
            records.append(features)

        if not records:
            records.append(_extract_document_features("", page_number=1))

        df = pd.DataFrame(records)
        extracted_rules = DocumentRuleExtractor.extract_rules(raw_text, known_columns=list(df.columns))
        if role == "tabular":
            role = "document_records"

    else:
        raise ValueError(f"Unsupported file format '{ext}'.")

    # Profile the resulting DataFrame
    profiler = SchemaProfiler()
    profile = profiler.profile(df, target_col=target_col)

    return IngestedDataset(
        name=dataset_name,
        source_type="document" if ext in (".pdf", ".docx", ".txt", ".md") else "file",
        format=fmt,
        df=df,
        raw_text=raw_text,
        extracted_rules=extracted_rules,
        profile=profile,
        role=role,
        metadata={"filename": fname, "format": fmt, "num_rows": len(df)},
    )


def load_bytes(
    filename: str,
    content: bytes,
    role: str = "tabular",
    name: str | None = None,
    target_col: str | None = None,
) -> IngestedDataset:
    """Loads dataset from raw in-memory bytes."""
    buf = io.BytesIO(content)
    return load_file(
        buf,
        filename=filename,
        role=role,
        name=name,
        target_col=target_col,
    )
