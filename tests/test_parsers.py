"""Generic-PDF parsing: layout-aware fallback, page markers, batching, scanned detection."""

import re

import pytest
from fpdf import FPDF

from ragops.config import ChunkingConfig
from ragops.ingestion import parsers
from ragops.ingestion.chunking import Chunker, split_sections
from ragops.ingestion.parsers import (
    PAGE_MARKER_RE,
    ScannedPDFError,
    normalize_page_markers,
    parse_pdf_with_gemini,
    parse_pdf_with_pypdf,
    split_pdf,
)
from ragops.ingestion.pdf_builder import markdown_to_pdf
from ragops.ingestion.pipeline import corpus_file, load_manifest
from ragops.providers.router import LLMResult


def _corpus_pdf(doc_id: str) -> bytes:
    doc = next(d for d in load_manifest() if d.doc_id == doc_id)
    return corpus_file(doc).read_bytes()


def _long_pdf(tmp_path, sections: int = 12) -> bytes:
    body = "\n\n".join(
        f"## Chapter {i}\n\n" + " ".join(f"Sentence {j} of chapter {i} explains topic {i}." for j in range(40))
        for i in range(1, sections + 1)
    )
    path = markdown_to_pdf(f"# Long Field Guide\n\n{body}", tmp_path / "long.pdf", footer_text="Acme Field Guide")
    return path.read_bytes()


def test_numbered_headings_recovered_from_font_size_and_weight():
    md = parse_pdf_with_pypdf(_corpus_pdf("credit-underwriting-policy"))
    assert md.splitlines()[2].startswith("# Lumen Ledger Finance - Credit Underwriting Policy")
    assert "## 4. Sanctioning Authority Matrix" in md
    assert "### 2.2 Credit Bureau Score" in md
    assert "| Page" not in md  # footers dropped


def test_unnumbered_headings_detected_in_product_manual():
    md = parse_pdf_with_pypdf(_corpus_pdf("aurora-x2-home-battery-manual"))
    headings = [ln for ln in md.splitlines() if ln.startswith("#")]
    assert headings[0].startswith("# Aurora X2 Home Battery")
    assert "## Technical Specifications" in headings
    assert "## Warranty" in headings
    assert "### Location and Clearances" in headings
    # numbered *list items* ("1. Install the app ...") must not become headings
    assert not any("Install the Aurora Home app" in h for h in headings)


def test_wrapped_title_is_rejoined():
    md = parse_pdf_with_pypdf(_corpus_pdf("retrieval-depth-study"))
    headings = [ln for ln in md.splitlines() if ln.startswith("#")]
    assert headings[0] == "# Effect of Retrieval Depth and Reranking on Grounded Answer Quality"
    assert "## Abstract" in headings and "### 3.2 Number of Context Passages" in headings


def test_running_footer_removed_and_pages_marked(tmp_path):
    md = parse_pdf_with_pypdf(_long_pdf(tmp_path))
    pages = [int(n) for n in PAGE_MARKER_RE.findall(md)]
    assert len(pages) >= 3 and pages == list(range(1, len(pages) + 1))
    assert "Acme Field Guide" not in md


async def test_page_ranges_flow_into_chunks(tmp_path):
    md = parse_pdf_with_pypdf(_long_pdf(tmp_path))
    chunks = await Chunker(ChunkingConfig(strategy="structural", max_tokens=300)).chunk(
        doc_id="guide", markdown=md, acl_groups=["public"]
    )
    assert all(c.page_start is not None and c.page_end >= c.page_start for c in chunks)
    assert chunks[-1].page_end == max(int(n) for n in PAGE_MARKER_RE.findall(md))
    assert all("<!--" not in c.text for c in chunks)  # markers never leak into text
    last = chunks[-1]
    assert last.page_label.startswith("p")


def test_split_sections_consumes_page_markers():
    md = "# T\n\n<!-- page 1 -->\n## A\nalpha text\n<!-- page 2 -->\nmore alpha\n## B\n<!-- page 3 -->\nbeta"
    _, sections = split_sections(md)
    a, b = sections
    assert (a.page_start, a.page_end) == (1, 2) and "<!--" not in a.body
    assert (b.page_start, b.page_end) == (3, 3)


def test_scanned_pdf_is_rejected_with_actionable_error(tmp_path):
    pdf = FPDF()
    for _ in range(3):
        pdf.add_page()
        pdf.rect(20, 20, 150, 200, style="F")  # an "image" page with no text layer
    path = tmp_path / "scan.pdf"
    pdf.output(str(path))
    with pytest.raises(ScannedPDFError, match="Gemini parser"):
        parse_pdf_with_pypdf(path.read_bytes())


def test_split_pdf_into_page_batches(tmp_path):
    data = _long_pdf(tmp_path, sections=14)
    batches = split_pdf(data, pages_per_batch=2)
    assert batches[0][0] == 1 and batches[0][1] == 2
    assert all(last - first + 1 <= 2 for first, last, _ in batches)
    assert batches[-1][1] == sum(last - first + 1 for first, last, _ in batches)
    assert split_pdf(data, pages_per_batch=500)[0][2] == data  # small docs are not re-written


def test_normalize_page_markers():
    # relative numbering (1..k) is shifted to absolute pages
    fixed = normalize_page_markers("<!-- page 1 -->\na\n<!-- page 2 -->\nb", 11, 12)
    assert [int(n) for n in PAGE_MARKER_RE.findall(fixed)] == [11, 12]
    # missing leading marker is added, out-of-range markers are clamped
    fixed = normalize_page_markers("text\n<!-- page 99 -->\nmore", 5, 7)
    assert [int(n) for n in PAGE_MARKER_RE.findall(fixed)] == [5, 7]


async def test_gemini_batches_are_stitched_with_absolute_pages(tmp_path, monkeypatch):
    data = _long_pdf(tmp_path, sections=10)
    n_batches = len(split_pdf(data, 2))
    calls = []

    class FakeRouter:
        async def complete(self, route, messages, **kwargs):
            prompt = messages[1]["content"][1]["text"]
            first, last = map(int, re.search(r"pages (\d+) to (\d+)", prompt).groups())
            calls.append((route, first, last))
            # simulate a model that numbers pages relative to the excerpt
            body = "\n".join(f"<!-- page {i} -->\n## Part {first + i - 1}\ntext" for i in range(1, last - first + 2))
            return LLMResult(text=f"```markdown\n{body}\n```", route=route, model="fake")

    monkeypatch.setattr("ragops.providers.router.get_router", lambda: FakeRouter())
    md = await parse_pdf_with_gemini(data, pages_per_batch=2, concurrency=3)
    pages = [int(n) for n in PAGE_MARKER_RE.findall(md)]
    assert len(calls) == n_batches and all(r == "pdf_parsing" for r, _, _ in calls)
    assert pages == list(range(1, pages[-1] + 1))
    assert "```" not in md


def test_supported_suffixes():
    assert {".pdf", ".md", ".txt"} <= parsers.SUPPORTED_SUFFIXES
