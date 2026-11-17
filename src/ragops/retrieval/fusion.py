"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

Dense (semantic) and sparse (BM25) retrievers produce scores on incomparable scales:
cosine similarity in [-1, 1] versus unbounded BM25. RRF ignores the raw scores and
fuses on *rank* alone:

    RRF(d) = sum over retrievers r of  w_r / (k + rank_r(d))

with ranks starting at 1 and k = 60 by default. A document ranked highly by either
retriever surfaces, and one ranked well by both wins. k damps the influence of the very
top ranks so a single retriever cannot dominate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ragops.types import Chunk, RetrievedChunk

# One retriever's output, best first: (chunk, raw_score)
RankedList = Sequence[tuple[Chunk, float]]


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, RankedList],
    k: int = 60,
    top_k: int | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists keyed by retriever name ("dense", "sparse", ...)."""
    fused: dict[str, RetrievedChunk] = {}
    for source, items in ranked_lists.items():
        weight = (weights or {}).get(source, 1.0)
        for rank, (chunk, score) in enumerate(items, start=1):
            entry = fused.get(chunk.chunk_id)
            if entry is None:
                entry = fused[chunk.chunk_id] = RetrievedChunk(chunk=chunk)
            entry.rrf_score += weight / (k + rank)
            if source == "dense":
                entry.dense_rank, entry.dense_score = rank, score
            elif source == "sparse":
                entry.sparse_rank, entry.sparse_score = rank, score

    ordered = sorted(
        fused.values(),
        # Tie-break deterministically: best individual rank, then chunk id.
        key=lambda r: (
            -r.rrf_score,
            min(r.dense_rank or 10**9, r.sparse_rank or 10**9),
            r.chunk.chunk_id,
        ),
    )
    if top_k is not None:
        ordered = ordered[:top_k]
    for i, item in enumerate(ordered, start=1):
        item.final_rank = i
    return ordered
