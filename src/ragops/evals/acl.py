"""Deterministic permission-leakage checks (no LLM judge, zero tolerance).

Inspects the *pre-rerank candidate pool* (top 50 fused results), not just the final
contexts, so a leak is caught even if the reranker happened to drop the chunk.
"""

from __future__ import annotations

import yaml

from ragops.config import EVALS_DIR
from ragops.evals.models import AclSummary
from ragops.retrieval.retriever import HybridRetriever
from ragops.security.acl import resolve_user
from ragops.types import PipelineOverrides

ACL_CASES = EVALS_DIR / "datasets" / "acl_cases.yaml"


def load_acl_cases() -> list[dict]:
    return yaml.safe_load(ACL_CASES.read_text(encoding="utf-8"))["cases"]


async def run_acl_checks(retriever: HybridRetriever | None = None, max_leaked: int = 0) -> AclSummary:
    retriever = retriever or HybridRetriever()
    overrides = PipelineOverrides(rerank_enabled=False, fused_top_k=50, max_context_chunks=50)
    details, total = [], 0
    for case in load_acl_cases():
        user = resolve_user(case["user_id"])
        result = await retriever.retrieve(case["question"], user, overrides)
        forbidden = set(case["forbidden_doc_ids"])
        leaked = [c.chunk.chunk_id for c in result.candidates if c.chunk.doc_id in forbidden]
        # Defence in depth: no candidate may lack an ACL group the user holds.
        unauthorised = [c.chunk.chunk_id for c in result.candidates if not set(c.chunk.acl_groups) & set(user.groups)]
        leaked = sorted(set(leaked) | set(unauthorised))
        total += len(leaked)
        details.append(
            {
                "id": case["id"],
                "user_id": user.user_id,
                "groups": user.groups,
                "candidates": len(result.candidates),
                "leaked_chunk_ids": leaked,
            }
        )
    return AclSummary(cases=len(details), leaked_chunks=total, passed=total <= max_leaked, details=details)
