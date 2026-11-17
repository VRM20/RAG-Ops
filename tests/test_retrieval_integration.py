"""Integration: real ingestion (PDF render -> pypdf parse -> chunk -> index) + hybrid retrieval."""

from ragops.evals.acl import run_acl_checks
from ragops.ingestion.pipeline import load_manifest
from ragops.retrieval.retriever import HybridRetriever
from ragops.security.acl import resolve_user


def test_corpus_is_fully_indexed(indexed_corpus):
    assert len(indexed_corpus) == len(load_manifest())
    assert all(r.num_chunks > 0 for r in indexed_corpus)


async def test_hybrid_retrieval_finds_exact_terms(indexed_corpus):
    result = await HybridRetriever().retrieve("minimum CIBIL bureau score for a loan", resolve_user("priya.sharma"))
    top_docs = [c.chunk.doc_id for c in result.contexts[:3]]
    assert "credit-underwriting-policy" in top_docs
    top = result.contexts[0]
    assert top.rrf_score > 0 and top.final_rank == 1
    assert top.dense_rank is not None or top.sparse_rank is not None


async def test_bm25_matches_inflected_forms(indexed_corpus):
    # "sanctions" must match "Sanctioning Authority Matrix" (stemming regression test)
    result = await HybridRetriever().retrieve("Who sanctions a loan of INR 2 crore?", resolve_user("priya.sharma"))
    best_sparse = min((c for c in result.candidates if c.sparse_rank), key=lambda c: c.sparse_rank)
    assert best_sparse.chunk.section.startswith("4. Sanctioning Authority Matrix")


async def test_acl_filter_hides_restricted_documents(indexed_corpus):
    retriever = HybridRetriever()
    q = "pay band for grade M3 and bonus multiplier"
    as_hr = await retriever.retrieve(q, resolve_user("meera.iyer"))
    as_analyst = await retriever.retrieve(q, resolve_user("priya.sharma"))
    assert any(c.chunk.doc_id == "compensation-and-bonus-framework-2026" for c in as_hr.contexts)
    assert all(c.chunk.doc_id != "compensation-and-bonus-framework-2026" for c in as_analyst.candidates)


async def test_contractor_sees_only_public_documents(indexed_corpus):
    public = {d.doc_id for d in load_manifest() if "public" in d.acl_groups}
    result = await HybridRetriever().retrieve("privileged leave days and warranty", resolve_user("sam.contractor"))
    assert result.candidates, "public documents should still be searchable"
    assert {c.chunk.doc_id for c in result.candidates} <= public


async def test_cross_domain_retrieval_with_page_citations(indexed_corpus):
    result = await HybridRetriever().retrieve("What does error code E101 mean?", resolve_user("sam.contractor"))
    top = result.contexts[0].chunk
    assert top.doc_id == "aurora-x2-home-battery-manual"
    assert top.page_start is not None  # PDF chunks carry page numbers for citations


async def test_acl_suite_reports_zero_leaks(indexed_corpus):
    summary = await run_acl_checks()
    assert summary.cases >= 6
    assert summary.leaked_chunks == 0 and summary.passed
