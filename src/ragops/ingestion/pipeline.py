"""Ingestion pipeline: file -> parse -> chunk -> embed (dense + BM25) -> index.

Idempotent by design: point ids are derived from chunk ids and a document's old chunks
are deleted before the new ones are written, so re-ingesting a changed document never
leaves stale chunks behind.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel

from ragops.config import DATA_DIR, RagConfig, get_rag_config, get_settings
from ragops.ingestion.chunking import Chunker
from ragops.ingestion.parsers import parse_document, sha256_bytes
from ragops.ingestion.registry import DocumentRecord, Registry
from ragops.observability.tracing import span
from ragops.providers.embeddings import get_embedder
from ragops.retrieval.sparse import BM25Encoder
from ragops.retrieval.store import collection_name, get_store
from ragops.types import Chunk

log = logging.getLogger(__name__)

CORPUS_DIR = DATA_DIR / "corpus"


class CorpusDocument(BaseModel):
    doc_id: str
    title: str
    source: str
    format: str = "md"
    acl_groups: list[str]
    organization: str = "Internal document"  # running footer of rendered PDFs
    domain: str = "general"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80] or "document"


def uploads_dir() -> Path:
    return get_settings().state_dir / "uploads"


def load_manifest() -> list[CorpusDocument]:
    data = yaml.safe_load((CORPUS_DIR / "manifest.yaml").read_text(encoding="utf-8"))
    return [CorpusDocument.model_validate(d) for d in data["documents"]]


def corpus_file(doc: CorpusDocument) -> Path:
    """Path of the file to ingest for a manifest entry (renders the PDF if needed)."""
    source = CORPUS_DIR / doc.source
    if doc.format != "pdf":
        return source
    pdf = CORPUS_DIR / "pdf" / f"{doc.doc_id}.pdf"
    if not pdf.exists() or pdf.stat().st_mtime < source.stat().st_mtime:
        from ragops.ingestion.pdf_builder import markdown_to_pdf

        markdown_to_pdf(source.read_text(encoding="utf-8"), pdf, footer_text=doc.organization)
    return pdf


class IngestionService:
    def __init__(self, cfg: RagConfig | None = None, registry: Registry | None = None):
        self.cfg = cfg or get_rag_config()
        self.registry = registry or Registry()
        self.index_name = collection_name(self.cfg)
        self.store = get_store(self.cfg)
        self.embedder = get_embedder()
        self.bm25 = BM25Encoder()

    def register(self, *, filename: str, data: bytes, acl_groups: list[str], title: str | None = None, doc_id: str | None = None) -> DocumentRecord:
        """Persist the upload and create a `queued` registry record."""
        doc_id = slugify(doc_id or Path(filename).stem)
        target = uploads_dir() / doc_id / Path(filename).name
        if target.parent.exists():
            shutil.rmtree(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rec = DocumentRecord(
            index_name=self.index_name,
            doc_id=doc_id,
            title=title or Path(filename).stem.replace("-", " ").title(),
            filename=Path(filename).name,
            source_path=str(target),
            acl_groups=sorted(set(acl_groups)) or ["public"],
            status="queued",
            size_bytes=len(data),
            sha256=sha256_bytes(data),
        )
        return self.registry.upsert(rec)

    async def process(self, doc_id: str) -> DocumentRecord:
        rec = self.registry.get(self.index_name, doc_id)
        if rec is None:
            raise KeyError(doc_id)
        try:
            with span("ingest.document", "CHAIN", input_value=rec.filename, **{"doc.id": doc_id}):
                await self.store.ensure_collection()
                data = await asyncio.to_thread(Path(rec.source_path).read_bytes)

                self.registry.set_status(self.index_name, doc_id, "parsing")
                parsed = await parse_document(rec.filename, data, self.cfg.ingestion)
                for warning in parsed.warnings:
                    log.warning("%s: %s", doc_id, warning)

                self.registry.set_status(self.index_name, doc_id, "chunking", parser=parsed.parser, pages=parsed.pages)
                chunker = Chunker(self.cfg.ingestion.chunking, embed_fn=self.embedder.embed_for_similarity)
                chunks = await chunker.chunk(
                    doc_id=doc_id,
                    markdown=parsed.markdown,
                    acl_groups=rec.acl_groups,
                    title=rec.title,
                    metadata={"filename": rec.filename, "parser": parsed.parser, "sha256": parsed.sha256},
                )
                if not chunks:
                    raise ValueError("document produced no chunks (empty or unreadable)")

                self.registry.set_status(self.index_name, doc_id, "embedding", num_chunks=len(chunks))
                await self._index(doc_id, chunks)

            self.registry.set_status(self.index_name, doc_id, "indexed", error=None, num_chunks=len(chunks))
            log.info("Indexed %s: %d chunks (%s)", doc_id, len(chunks), parsed.parser)
        except Exception as exc:
            log.exception("Ingestion failed for %s", doc_id)
            self.registry.set_status(self.index_name, doc_id, "failed", error=f"{type(exc).__name__}: {exc}"[:500])
        return self.registry.get(self.index_name, doc_id)  # type: ignore[return-value]

    async def _index(self, doc_id: str, chunks: list[Chunk]) -> None:
        texts = [c.embed_text for c in chunks]
        dense = await self.embedder.embed_documents(texts)
        sparse = [self.bm25.encode_document(t) for t in texts]
        await self.store.delete_document(doc_id)
        await self.store.upsert(chunks, dense, sparse)

    async def delete(self, doc_id: str) -> None:
        await self.store.delete_document(doc_id)
        self.registry.delete(self.index_name, doc_id)
        shutil.rmtree(uploads_dir() / doc_id, ignore_errors=True)

    def register_corpus(self) -> list[str]:
        ids = []
        for doc in load_manifest():
            path = corpus_file(doc)
            self.register(filename=path.name, data=path.read_bytes(), acl_groups=doc.acl_groups, title=doc.title, doc_id=doc.doc_id)
            ids.append(doc.doc_id)
        return ids

    async def ingest_corpus(self) -> list[DocumentRecord]:
        return [await self.process(doc_id) for doc_id in self.register_corpus()]
