"""Live SQL Database Connector with read-only query guardrails and schema inspection."""

from __future__ import annotations
import re
from typing import Any
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, inspect, text


FORBIDDEN_SQL_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "CREATE", "REPLACE", "EXEC", "EXECUTE",
    "CALL", "VACUUM", "MERGE", "UPSERT", "INTO", "ATTACH", "DETACH"
}


def validate_read_only_sql(query: str) -> None:
    """Strictly validates that a SQL query is read-only (SELECT / WITH).
    
    Raises ValueError if write or destructive statements, multiple statements,
    or forbidden SQL keywords are detected.
    """
    if not query or not query.strip():
        raise ValueError("Read-only SQL violation: Query cannot be empty.")

    # 1. Strip comments to prevent comment-based evasions
    # Single-line comments (-- comment)
    clean = re.sub(r"--.*?$", "", query, flags=re.MULTILINE)
    # Multi-line comments (/* comment */)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
    clean = clean.strip()

    if not clean:
        raise ValueError("Read-only SQL violation: Query contains only comments.")

    # 2. Block multiple statements separated by semicolons
    # Allow an optional trailing semicolon, but no semicolon within the statement body
    trimmed_trailing = clean.rstrip(";").strip()
    if ";" in trimmed_trailing:
        raise ValueError("Read-only SQL violation: Multiple SQL statements are not permitted.")

    # 3. Verify starting command is SELECT or WITH
    first_token = trimmed_trailing.split()[0].upper()
    if first_token not in ("SELECT", "WITH"):
        raise ValueError(
            f"Read-only SQL violation: Only SELECT or WITH queries are permitted. Found '{first_token}'."
        )

    # 4. Check for forbidden keywords as standalone word tokens
    for kw in FORBIDDEN_SQL_KEYWORDS:
        # Match as whole word case-insensitively
        if re.search(rf"\b{kw}\b", trimmed_trailing, flags=re.IGNORECASE):
            raise ValueError(f"Read-only SQL violation: Disallowed keyword '{kw}' detected.")


class SQLConnector:
    """Connects to live SQL databases via SQLAlchemy with strict read-only safety."""

    def __init__(self, uri: str):
        self.uri = uri.strip()
        # Create engine with pool_pre_ping for connection health
        self._engine = create_engine(self.uri, pool_pre_ping=True)

    def test_connection(self) -> dict[str, Any]:
        """Tests connectivity and retrieves dialect and table names."""
        try:
            with self._engine.connect() as conn:
                inspector = inspect(self._engine)
                table_names = inspector.get_table_names()
                dialect_name = self._engine.dialect.name
                return {
                    "connected": True,
                    "dialect": dialect_name,
                    "tables": table_names,
                    "error": None,
                }
        except Exception as e:
            return {
                "connected": False,
                "dialect": "unknown",
                "tables": [],
                "error": str(e),
            }

    def list_tables(self) -> list[str]:
        """Returns list of accessible table names."""
        inspector = inspect(self._engine)
        return inspector.get_table_names()

    def load_dataset(self, query_or_table: str, limit: int = 2000) -> pd.DataFrame:
        """Executes a read-only query or reads an entire table up to `limit` rows."""
        target = query_or_table.strip()
        tables = self.list_tables()

        if target in tables or re.fullmatch(r"^[a-zA-Z0-9_]+$", target):
            # Table name reference
            sanitized_table = re.sub(r"[^a-zA-Z0-9_]", "", target)
            query = f"SELECT * FROM {sanitized_table} LIMIT {int(limit)}"
        else:
            # Query string
            validate_read_only_sql(target)
            query = target.rstrip(";").strip()
            # If no LIMIT clause, apply pagination limit
            if not re.search(r"\bLIMIT\s+\d+\b", query, flags=re.IGNORECASE):
                query = f"SELECT * FROM ({query}) AS _rl_dataset_subq LIMIT {int(limit)}"

        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        return df

    def close(self) -> None:
        """Disposes the database engine pool."""
        self._engine.dispose()


def test_sql_connection(uri: str) -> dict[str, Any]:
    """Helper to test a SQL connection URI."""
    connector = SQLConnector(uri)
    res = connector.test_connection()
    connector.close()
    return res


test_sql_connection.__test__ = False


def list_sql_tables(uri: str) -> list[str]:
    """Helper to list tables from a SQL database URI."""
    connector = SQLConnector(uri)
    try:
        return connector.list_tables()
    finally:
        connector.close()


def load_sql_dataset(
    uri: str | None = None,
    query_or_table: str | None = None,
    limit: int = 2000,
    *,
    connection_uri: str | None = None,
    query: str | None = None,
    name: str = "sql_dataset",
) -> Any:
    """Helper to execute a read-only query and return an IngestedDataset."""
    resolved_uri = connection_uri or uri
    resolved_query = query or query_or_table
    if not resolved_uri:
        raise ValueError("Database connection URI must be provided.")
    if not resolved_query:
        raise ValueError("SQL query or table name must be provided.")

    connector = SQLConnector(resolved_uri)
    try:
        df = connector.load_dataset(resolved_query, limit=limit)
    finally:
        connector.close()

    from rl_ide.ingest.profiler import SchemaProfiler
    from rl_ide.ingest.loader import IngestedDataset

    profiler = SchemaProfiler()
    profile = profiler.profile(df)

    return IngestedDataset(
        name=name,
        source_type="sql",
        format="sql",
        df=df,
        profile=profile,
        role="tabular",
        metadata={
            "connection_uri": resolved_uri,
            "query": resolved_query,
            "num_rows": len(df),
        },
    )
