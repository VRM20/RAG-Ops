import math

from ragops.retrieval.fusion import reciprocal_rank_fusion
from ragops.retrieval.sparse import BM25Encoder, token_id, tokenize
from ragops.types import Chunk


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", doc_title="D", text=cid, embed_text=cid)


def test_tokenize_stems_inflections_and_drops_stopwords():
    assert tokenize("The Policies for Leaves and days") == tokenize("policy leave day")
    assert len(set(tokenize("sanctions sanctioning sanctioned"))) == 1
    assert "the" not in tokenize("the approval") and tokenize("approval") == tokenize("approve")
    # meaningful short / negation tokens and codes survive untouched
    assert "not" in tokenize("USB is not allowed")
    assert "m4" in tokenize("Grade M4 and above")
    assert "sev1" in tokenize("a SEV1 incident")


def test_bm25_document_weights_saturate_with_term_frequency():
    enc = BM25Encoder(k1=1.2, b=0.75, avg_doc_len=10)
    one = enc.encode_document("leave policy")
    many = enc.encode_document("leave leave leave leave policy")
    leave_id = token_id(tokenize("leave")[0])  # the index stores stems
    w1 = dict(zip(one.indices, one.values, strict=True))[leave_id]
    w4 = dict(zip(many.indices, many.values, strict=True))[leave_id]
    assert w4 > w1
    assert w4 < 4 * w1  # saturation: 4x the tf is far less than 4x the weight
    assert w4 < enc.k1 + 1  # upper bound of the BM25 tf component


def test_bm25_query_vector_is_binary_and_deduplicated():
    q = BM25Encoder().encode_query("leave leave policy")
    assert sorted(q.indices) == q.indices
    assert q.values == [1.0, 1.0]


def test_rrf_scores_follow_the_formula():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fused = reciprocal_rank_fusion({"dense": [(a, 0.9), (b, 0.8)], "sparse": [(b, 12.0), (c, 3.0)]}, k=60)
    by_id = {r.chunk.chunk_id: r for r in fused}
    assert math.isclose(by_id["b"].rrf_score, 1 / 62 + 1 / 61)
    assert math.isclose(by_id["a"].rrf_score, 1 / 61)
    assert fused[0].chunk.chunk_id == "b"  # found by both retrievers -> wins
    assert by_id["b"].dense_rank == 2 and by_id["b"].sparse_rank == 1
    assert [r.final_rank for r in fused] == [1, 2, 3]


def test_rrf_top_k_and_weights():
    a, b = _chunk("a"), _chunk("b")
    fused = reciprocal_rank_fusion({"dense": [(a, 1.0)], "sparse": [(b, 1.0)]}, k=60, top_k=1, weights={"sparse": 2.0})
    assert len(fused) == 1 and fused[0].chunk.chunk_id == "b"
