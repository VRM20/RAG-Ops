"""Deterministic offline stand-ins for cloud services (RAGOPS_OFFLINE=true).

They let the full system - ingestion, hybrid retrieval, ACL filtering, the API, the UI
and the unit tests - run with zero credentials and zero cost. They are NOT meant to be
good models; evaluation scores are only meaningful with real providers.

* :func:`hash_embed` - feature-hashed unigram+bigram vectors (a lexical proxy for
  semantic embeddings, stable across runs and machines).
* :func:`extractive_answer` - picks the source sentences that best overlap the question
  and cites them, or refuses when nothing overlaps.
* :func:`lexical_rerank` - BM25-style query/document term overlap.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from ragops.ingestion.chunking import split_sentences
from ragops.retrieval.sparse import tokenize

REFUSAL = "I couldn't find this in the documents you have access to."


def _hash(feature: str) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
    return h, 1.0 if (h >> 63) & 1 else -1.0


def hash_embed(texts: list[str], dimensions: int = 768) -> list[list[float]]:
    out = []
    for text in texts:
        vec = np.zeros(dimensions, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feat in features:
            h, sign = _hash(feat)
            vec[h % dimensions] += sign
        norm = float(np.linalg.norm(vec))
        out.append((vec / norm if norm else vec).tolist())
    return out


_SOURCE_RE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", re.MULTILINE | re.DOTALL)


def extractive_answer(user_message: str, max_sentences: int = 2) -> str:
    """Answer from the rendered RAG prompt by picking the best-overlapping sentences."""
    question = user_message.rsplit("Question:", 1)[-1].strip()
    q_terms = set(tokenize(question))
    scored: list[tuple[float, int, str]] = []
    for match in _SOURCE_RE.finditer(user_message.split("Question:", 1)[0]):
        number, body = int(match.group(1)), match.group(2)
        for sentence in split_sentences(" ".join(body.split())):
            terms = set(tokenize(sentence))
            if not terms:
                continue
            overlap = len(q_terms & terms)
            if overlap:
                scored.append((overlap / math.sqrt(len(terms)), number, sentence))
    if not scored:
        return REFUSAL
    scored.sort(key=lambda s: -s[0])
    picked = scored[:max_sentences]
    return " ".join(f"{sentence} [{number}]" for _, number, sentence in picked)


def lexical_rerank(query: str, documents: list[str]) -> list[float]:
    q_terms = set(tokenize(query))
    scores = []
    for doc in documents:
        terms = tokenize(doc)
        if not terms or not q_terms:
            scores.append(0.0)
            continue
        overlap = sum(1 for t in terms if t in q_terms)
        coverage = len(q_terms & set(terms)) / len(q_terms)
        scores.append(round(0.7 * coverage + 0.3 * min(1.0, overlap / 10), 4))
    return scores
