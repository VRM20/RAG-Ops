"""Test configuration: everything runs offline against a throwaway embedded Qdrant.

Environment variables are set *before* ragops is imported so that real credentials in
a developer's .env are never used by unit tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_STATE = Path(tempfile.mkdtemp(prefix="ragops-test-"))
os.environ.update(
    {
        "RAGOPS_OFFLINE": "true",
        "RAGOPS_ENV": "ci",
        "STATE_DIR": str(_STATE),
        "QDRANT_PATH": str(_STATE / "qdrant"),
        "QDRANT_URL": "",
        "TASK_QUEUE": "inline",
        "BEDROCK_GUARDRAIL_ID": "",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "",
        "ADMIN_API_KEY": "",
        "GUARDRAIL_ALLOWED_EMAIL_DOMAINS": "lumenledger.example",
        "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    }
)

import asyncio  # noqa: E402

import pytest  # noqa: E402

from ragops.config import reset_config_caches  # noqa: E402

reset_config_caches()


@pytest.fixture(scope="session")
def state_dir() -> Path:
    return _STATE


@pytest.fixture(scope="session")
def indexed_corpus() -> list:
    """Ingest the full demo corpus once per test session (offline embeddings, pypdf)."""
    from ragops.ingestion.pipeline import IngestionService

    records = asyncio.run(IngestionService().ingest_corpus())
    failed = [r for r in records if r.status != "indexed"]
    assert not failed, f"ingestion failed: {[(r.doc_id, r.error) for r in failed]}"
    return records
