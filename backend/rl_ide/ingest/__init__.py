"""Universal Data, Live SQL, and Document Ingestion Engine."""

from rl_ide.ingest.sql_connector import (
    SQLConnector,
    validate_read_only_sql,
    test_sql_connection,
    list_sql_tables,
    load_sql_dataset,
)
from rl_ide.ingest.profiler import (
    SchemaProfiler,
    DatasetProfile,
    ColumnProfile,
    AutoFeatureVectorizer,
    DocumentRuleExtractor,
    CandidateRule,
    sanitize_column_name,
)
from rl_ide.ingest.loader import (
    IngestedDataset,
    load_file,
    load_bytes,
)

__all__ = [
    "SQLConnector",
    "validate_read_only_sql",
    "test_sql_connection",
    "list_sql_tables",
    "load_sql_dataset",
    "SchemaProfiler",
    "DatasetProfile",
    "ColumnProfile",
    "AutoFeatureVectorizer",
    "DocumentRuleExtractor",
    "CandidateRule",
    "sanitize_column_name",
    "IngestedDataset",
    "load_file",
    "load_bytes",
]
