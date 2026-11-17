"""Hybrid retriever: ACL-filtered dense + BM25 search -> RRF fusion -> rerank.

    query ──► embed (RETRIEVAL_QUERY) ──► Qdrant dense  (filter: acl_groups ∩ user.groups) ─┐
          └─► BM25 query vector ───────► Qdrant sparse (same filter) ──────────────────────┤
                                                                                           ▼
                                                      RRF fusion (k=60) → top fused_top_k
                                                                                           ▼
                                                      cross-encoder rerank → top_n contexts

Dense search captures paraphrases ("time off" ~ "leave"); BM25 nails exact tokens that
embeddings blur (grade "M4", "SEV1", "FOIR", "INR 25 lakh"). The ACL filter is applied
inside both searches, so unauthorised chunks are never candidates.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from ragops.config import RagConfig, get_rag_config
from ragops.observability.tracing import set_documents, set_output, span
from ragops.providers.embeddings import Embedder, get_embedder
from ragops.retrieval.fusion import reciprocal_rank_fusion
from ragops.retrieval.rerank import Reranker
from ragops.retrieval.sparse import BM25Encoder
from ragops.retrieval.store import VectorStore, acl_filter, get_store
from ragops.types import PipelineOverrides, RetrievedChunk, UserContext


@dataclass
class RetrievalResult:
    contexts: list[RetrievedChunk]
    candidates: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        cfg: RagConfig | None = None,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.cfg = cfg or get_rag_config()
        self.store = store or get_store(self.cfg)
        self.embedder = embedder or get_embedder()
        self.reranker = reranker or Reranker(self.cfg.retrieval.rerank)
        self.bm25 = BM25Encoder()

    async def retrieve(
        self, query: str, user: UserContext, overrides: PipelineOverrides | None = None
    ) -> RetrievalResult:
        o = overrides or PipelineOverrides()
        r = self.cfg.retrieval
        fused_top_k = o.fused_top_k or r.fused_top_k
        rerank_enabled = r.rerank.enabled if o.rerank_enabled is None else o.rerank_enabled
        top_n = o.rerank_top_n or r.rerank.top_n
        max_chunks = o.max_context_chunks or self.cfg.generation.max_context_chunks
        timings: dict[str, float] = {}

        with span("retrieval.hybrid", "RETRIEVER", input_value=query, **{"user.id": user.user_id, "acl.groups": user.groups}) as sp:
            t0 = time.perf_counter()
            flt = acl_filter(user.groups)
            q_vec = await self.embedder.embed_query(query)
            timings["embed_query"] = _ms(t0)

            t1 = time.perf_counter()
            dense, sparse = await asyncio.gather(
                self.store.dense_search(q_vec, flt, r.dense_top_k),
                self.store.sparse_search(self.bm25.encode_query(query), flt, r.sparse_top_k),
            )
            timings["search"] = _ms(t1)

            candidates = reciprocal_rank_fusion({"dense": dense, "sparse": sparse}, k=r.rrf_k, top_k=fused_top_k)
            set_documents(sp, "retrieval.documents", candidates)
            sp.set_attribute("retrieval.dense_hits", len(dense))
            sp.set_attribute("retrieval.sparse_hits", len(sparse))

        if rerank_enabled and candidates:
            with span("rerank", "RERANKER", input_value=query, **{"reranker.model_name": self.reranker.model, "reranker.top_k": top_n}) as sp:
                t2 = time.perf_counter()
                set_documents(sp, "reranker.input_documents", candidates)
                reranked = await self.reranker.rerank(query, candidates, top_n)
                reranked = [c for c in reranked if (c.rerank_score or 0) >= r.min_relevance_score] or reranked[:1]
                set_documents(sp, "reranker.output_documents", reranked)
                set_output(sp, f"{len(reranked)} documents")
                timings["rerank"] = _ms(t2)
        else:
            reranked = candidates

        contexts = [c.model_copy(update={"final_rank": i}) for i, c in enumerate(reranked[:max_chunks], start=1)]
        timings["retrieval_total"] = _ms(t0)
        return RetrievalResult(contexts=contexts, candidates=candidates, timings_ms=timings)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
