"""BM25 sparse vectors for Qdrant.

Qdrant stores sparse vectors and, with ``Modifier.IDF`` on the sparse vector config,
computes the inverse-document-frequency term at query time across the collection.
That splits BM25 cleanly in two:

* **document side** (here, at index time): the saturated, length-normalised term
  frequency  ``tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``
* **query side** (here, at query time): every distinct query term with weight 1.0
* **IDF** (in Qdrant): ``log(1 + (N - n + 0.5) / (n + 0.5))``

Dot product of the two vectors then equals the classic BM25 score. Keeping the
document statistics in the index means new documents are searchable immediately,
with no global re-fit, which a separate in-memory BM25 index would need.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Snowball (Porter2): "sanctions" / "sanctioning" / "sanctioned" -> "sanction".
# The same stemmer family Qdrant's BM25 model uses.
_STEMMER = snowballstemmer.stemmer("english")

# Compact English stopword list. Domain words such as "not", "no", "above" and
# "below" are kept on purpose: they carry meaning in policy text.
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her
    his how i if in into is it its me my of on or our she should so than that the their them
    then there these they this those to us was we were what when where which who whom why
    will with would you your""".split()
)


@lru_cache(maxsize=65536)
def _stem(token: str) -> str:
    # Leave numbers and short codes (m4, sev1, e2) untouched.
    if len(token) <= 3 or any(ch.isdigit() for ch in token):
        return token
    return _STEMMER.stemWord(token)


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def token_id(token: str) -> int:
    """Stable 31-bit term id (Qdrant sparse indices are uint32)."""
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 150.0):
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len

    def encode_document(self, text: str) -> SparseVec:
        tokens = tokenize(text)
        if not tokens:
            return SparseVec([], [])
        tf = Counter(tokens)
        dl = len(tokens)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
        weights: dict[int, float] = {}
        for term, freq in tf.items():
            idx = token_id(term)
            # Accumulate in the (rare) case of a hash collision.
            weights[idx] = weights.get(idx, 0.0) + freq * (self.k1 + 1) / (freq + norm)
        return SparseVec(list(weights.keys()), list(weights.values()))

    def encode_query(self, text: str) -> SparseVec:
        ids = sorted({token_id(t) for t in tokenize(text)})
        return SparseVec(ids, [1.0] * len(ids))
