"""Embedding client (LiteLLM) with batching, bounded concurrency and retries.

Vertex AI embedding models are asymmetric: documents and queries are embedded with
different ``task_type`` values (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY), which measurably
improves retrieval over embedding both sides the same way.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ragops.config import EmbeddingConfig, get_rag_config, get_settings

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, cfg: EmbeddingConfig, concurrency: int = 4):
        self.cfg = cfg
        self.offline = get_settings().ragops_offline
        self._sem = asyncio.Semaphore(concurrency)

    @property
    def model(self) -> str:
        return "offline/hash-embedding" if self.offline else self.cfg.model

    def _provider_params(self) -> dict:
        model = self.cfg.model
        if model.startswith("vertex_ai/"):
            from ragops.providers.gcp import litellm_vertex_params

            return litellm_vertex_params(location=get_settings().vertex_embedding_location)
        if model.startswith("bedrock/"):
            from ragops.providers.aws import litellm_bedrock_params

            return litellm_bedrock_params()
        return {}

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=20), reraise=True)
    async def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        import litellm

        kwargs: dict = {"dimensions": self.cfg.dimensions, **self._provider_params()}
        if self.cfg.model.startswith("vertex_ai/"):
            # LiteLLM maps `input_type` onto Vertex's per-instance `task_type`.
            kwargs["input_type"] = task_type
        async with self._sem:
            resp = await litellm.aembedding(model=self.cfg.model, input=texts, **kwargs)
        data = sorted(resp.data, key=lambda d: d["index"]) if isinstance(resp.data[0], dict) else resp.data
        return [d["embedding"] if isinstance(d, dict) else d.embedding for d in data]

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        if self.offline:
            from ragops.providers.offline import hash_embed

            return hash_embed(texts, self.cfg.dimensions)
        size = max(1, self.cfg.batch_size)
        batches = [texts[i : i + size] for i in range(0, len(texts), size)]
        results = await asyncio.gather(*(self._embed_batch(b, task_type) for b in batches))
        vectors = [v for batch in results for v in batch]
        if vectors and len(vectors[0]) != self.cfg.dimensions:
            raise ValueError(
                f"Embedding model returned {len(vectors[0])} dims, config expects {self.cfg.dimensions}"
            )
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]

    async def embed_for_similarity(self, texts: list[str]) -> list[list[float]]:
        """Used by semantic chunking to compare neighbouring sentences."""
        return await self._embed(texts, "SEMANTIC_SIMILARITY")


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_rag_config().embedding)
