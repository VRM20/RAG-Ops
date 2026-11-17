"""Qdrant vector store with hybrid (dense + BM25 sparse) vectors and ACL filtering.

Collection layout
-----------------
* named dense vector ``dense``  - cosine, ``embedding.dimensions`` wide
* named sparse vector ``bm25``  - with ``Modifier.IDF`` (Qdrant computes IDF)
* payload: chunk fields + ``acl_groups`` (keyword index) + ``doc_id`` (keyword index)

Index versioning
----------------
The collection name embeds a fingerprint of everything that changes the vectors
(embedding model, dimensions, chunking config, PDF parser). Changing any of them
writes a *new* collection instead of silently mixing incompatible vectors: old and new
indexes coexist (blue/green), and an eval run always records which index it used.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from functools import lru_cache

from qdrant_client import AsyncQdrantClient, models

from ragops.config import RagConfig, get_rag_config, get_settings
from ragops.retrieval.sparse import SparseVec
from ragops.types import Chunk

DENSE = "dense"
SPARSE = "bm25"
_NAMESPACE = uuid.UUID("6f1c2a8e-3b7d-4c55-9a43-2f5e0d8b9c11")


def effective_embedding_model(cfg: RagConfig) -> str:
    return "offline/hash-embedding" if get_settings().ragops_offline else cfg.embedding.model


def index_fingerprint(cfg: RagConfig) -> str:
    payload = {
        "embedding_model": effective_embedding_model(cfg),
        "dimensions": cfg.embedding.dimensions,
        "chunking": cfg.ingestion.chunking.model_dump(),
        "pdf_parser": "pypdf" if get_settings().ragops_offline else cfg.ingestion.pdf_parser,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def collection_name(cfg: RagConfig | None = None) -> str:
    cfg = cfg or get_rag_config()
    return f"{get_settings().qdrant_collection_prefix}_{index_fingerprint(cfg)}"


def point_id(chunk_id: str) -> str:
    # Deterministic ids make re-ingestion idempotent (upsert overwrites).
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def acl_filter(groups: list[str], doc_ids: list[str] | None = None) -> models.Filter:
    must: list[models.Condition] = [models.FieldCondition(key="acl_groups", match=models.MatchAny(any=groups))]
    if doc_ids:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
    return models.Filter(must=must)


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def chunk_from_payload(payload: dict) -> Chunk:
    return Chunk.model_validate(payload)


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, dimensions: int):
        self.client = client
        self.collection = collection
        self.dimensions = dimensions

    async def ensure_collection(self) -> None:
        if await self.client.collection_exists(self.collection):
            return
        await self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in ("acl_groups", "doc_id"):
            await self.client.create_payload_index(
                self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )

    async def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVec]) -> None:
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector={DENSE: d, SPARSE: models.SparseVector(indices=s.indices, values=s.values)},
                payload=c.model_dump(),
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for i in range(0, len(points), 128):
            await self.client.upsert(self.collection, points=points[i : i + 128], wait=True)

    async def delete_document(self, doc_id: str) -> None:
        if await self.client.collection_exists(self.collection):
            await self.client.delete(self.collection, points_selector=models.FilterSelector(filter=_doc_filter(doc_id)), wait=True)

    async def dense_search(self, vector: list[float], flt: models.Filter, limit: int) -> list[tuple[Chunk, float]]:
        res = await self.client.query_points(
            self.collection, query=vector, using=DENSE, query_filter=flt, limit=limit, with_payload=True
        )
        return [(chunk_from_payload(p.payload), float(p.score)) for p in res.points]

    async def sparse_search(self, vector: SparseVec, flt: models.Filter, limit: int) -> list[tuple[Chunk, float]]:
        if not vector.indices:
            return []
        res = await self.client.query_points(
            self.collection,
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using=SPARSE,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return [(chunk_from_payload(p.payload), float(p.score)) for p in res.points]

    async def document_chunks(self, doc_id: str, limit: int = 1000) -> list[Chunk]:
        if not await self.client.collection_exists(self.collection):
            return []
        points, _ = await self.client.scroll(
            self.collection, scroll_filter=_doc_filter(doc_id), limit=limit, with_payload=True, with_vectors=False
        )
        return sorted((chunk_from_payload(p.payload) for p in points), key=lambda c: c.chunk_index)

    async def count(self) -> int:
        if not await self.client.collection_exists(self.collection):
            return 0
        return (await self.client.count(self.collection, exact=True)).count


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None)
    s.qdrant_path.mkdir(parents=True, exist_ok=True)
    return AsyncQdrantClient(path=str(s.qdrant_path))


def get_store(cfg: RagConfig | None = None) -> VectorStore:
    cfg = cfg or get_rag_config()
    return VectorStore(get_qdrant_client(), collection_name(cfg), cfg.embedding.dimensions)
