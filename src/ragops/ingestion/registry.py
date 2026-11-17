"""Document registry (SQLite): ingestion status and metadata per document and index.

SQLite is safe to share between the API process and Celery workers on one host. For a
multi-host deployment swap this for Postgres; the interface stays the same.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from ragops.config import get_settings

Status = Literal["queued", "parsing", "chunking", "embedding", "indexed", "failed"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    index_name  TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    title       TEXT NOT NULL,
    filename    TEXT NOT NULL,
    source_path TEXT NOT NULL,
    acl_groups  TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    parser      TEXT,
    pages       INTEGER,
    num_chunks  INTEGER DEFAULT 0,
    size_bytes  INTEGER DEFAULT 0,
    sha256      TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (index_name, doc_id)
);
"""


class DocumentRecord(BaseModel):
    index_name: str
    doc_id: str
    title: str
    filename: str
    source_path: str
    acl_groups: list[str] = Field(default_factory=list)
    status: Status = "queued"
    error: str | None = None
    parser: str | None = None
    pages: int | None = None
    num_chunks: int = 0
    size_bytes: int = 0
    sha256: str | None = None
    created_at: str = ""
    updated_at: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Registry:
    _lock = threading.Lock()

    def __init__(self, path=None):
        self.path = path or (get_settings().state_dir / "ragops.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                yield conn
                conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> DocumentRecord:
        data = dict(row)
        data["acl_groups"] = json.loads(data["acl_groups"])
        return DocumentRecord.model_validate(data)

    def upsert(self, rec: DocumentRecord) -> DocumentRecord:
        now = _now()
        rec.created_at = rec.created_at or now
        rec.updated_at = now
        data = rec.model_dump()
        data["acl_groups"] = json.dumps(rec.acl_groups)
        cols = ", ".join(data)
        marks = ", ".join(f":{k}" for k in data)
        updates = ", ".join(f"{k}=excluded.{k}" for k in data if k not in {"index_name", "doc_id", "created_at"})
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO documents ({cols}) VALUES ({marks}) "
                f"ON CONFLICT(index_name, doc_id) DO UPDATE SET {updates}",
                data,
            )
        return rec

    def set_status(self, index_name: str, doc_id: str, status: Status, **fields) -> None:
        fields.update(status=status, updated_at=_now())
        sets = ", ".join(f"{k}=:{k}" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE documents SET {sets} WHERE index_name=:index_name AND doc_id=:doc_id",
                {**fields, "index_name": index_name, "doc_id": doc_id},
            )

    def get(self, index_name: str, doc_id: str) -> DocumentRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE index_name=? AND doc_id=?", (index_name, doc_id)
            ).fetchone()
        return self._row(row) if row else None

    def list(self, index_name: str) -> list[DocumentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE index_name=? ORDER BY created_at, doc_id", (index_name,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def delete(self, index_name: str, doc_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE index_name=? AND doc_id=?", (index_name, doc_id))
