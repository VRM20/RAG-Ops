"""Cross-encoder reranking (Cohere Rerank 3.5 by default, via LiteLLM).

Bi-encoder retrieval (dense + BM25) is fast but scores query and document separately.
A cross-encoder reads the query and each candidate *together*, which is far more
precise, but too slow to run over the whole index. Standard two-stage design: hybrid
retrieval narrows the index to ~20 candidates, the reranker picks the best ~5.

The reranker model is a LiteLLM string, so switching to Bedrock-hosted Cohere
(``bedrock/cohere.rerank-v3-5:0``) or another provider is a config change. If the
reranker fails, retrieval degrades to the RRF order instead of failing the request.
"""

from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ragops.config import RerankConfig, get_settings
from ragops.types import RetrievedChunk

log = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=2, max=30), reraise=True)
async def _remote_rerank(model: str, query: str, documents: list[str], top_n: int, params: dict) -> list[tuple[int, float]]:
    # Retries cover rate limits (Cohere trial keys allow ~10 rerank calls/minute).
    import litellm

    resp = await litellm.arerank(model=model, query=query, documents=documents, top_n=top_n, **params)
    return [(r["index"], float(r["relevance_score"])) for r in resp.results]


class Reranker:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        self.offline = get_settings().ragops_offline
        # Surfaced in eval runs: silent degradation to RRF order would skew scores.
        self.fallback_count = 0

    @property
    def model(self) -> str:
        return "offline/lexical" if self.offline else self.cfg.model

    def _params(self) -> dict:
        s = get_settings()
        if self.cfg.model.startswith("cohere/") and s.cohere_api_key:
            return {"api_key": s.cohere_api_key.get_secret_value()}
        if self.cfg.model.startswith("bedrock/"):
            from ragops.providers.aws import litellm_bedrock_params

            return litellm_bedrock_params()
        return {}

    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        documents = [c.chunk.embed_text for c in candidates]
        try:
            if self.offline:
                from ragops.providers.offline import lexical_rerank

                scored = list(enumerate(lexical_rerank(query, documents)))
            else:
                scored = await _remote_rerank(self.cfg.model, query, documents, min(top_n, len(documents)), self._params())
        except Exception as exc:
            log.warning("Reranker failed after retries (%s); falling back to RRF order", exc)
            self.fallback_count += 1
            return candidates[:top_n]

        scored.sort(key=lambda s: -s[1])
        out = []
        for rank, (idx, score) in enumerate(scored[:top_n], start=1):
            item = candidates[idx].model_copy()
            item.rerank_score = round(score, 4)
            item.final_rank = rank
            out.append(item)
        return out
