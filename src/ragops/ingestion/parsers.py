"""Document parsers: any PDF / Markdown / text -> structured Markdown with page markers.

Output contract
---------------
Markdown with a heading hierarchy (``#`` title, ``##`` sections, ``###`` subsections)
and ``<!-- page N -->`` lines at every page start. The chunker turns the markers into
``page_start`` / ``page_end`` on each chunk, so answers can cite "p. 12".

PDF strategy
------------
* ``gemini`` (primary): the PDF is split into page batches (``pdf_pages_per_batch``)
  and each batch is sent to Gemini on Vertex AI (route ``pdf_parsing``), which reads
  the page images plus text layer. That handles multi-column layouts, tables,
  figures, and scanned pages (OCR). Batches run concurrently and are stitched back
  together with validated absolute page markers. Batching is what makes arbitrarily
  long PDFs work: one call cannot emit a 300-page document within the model's
  output-token limit.
* ``pypdf`` (fallback, no LLM cost): local extraction that rebuilds headings from
  *font size and weight* plus numbering (``2.1 ...``), works for unnumbered documents
  too, removes running headers/footers that repeat across pages, and fails loudly on
  scanned / image-only PDFs, which need OCR.

Gemini output is cached by content hash + prompt hash + batch size, so re-ingesting an
unchanged document (or re-running CI) never pays for parsing twice.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from ragops.config import IngestionConfig, get_rag_config, get_settings
from ragops.generation.prompts import load_prompt

log = logging.getLogger(__name__)

PDF_PROMPT = "pdf_parse_v1"
SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}
PAGE_MARKER = "<!-- page {n} -->"
PAGE_MARKER_RE = re.compile(r"^[ \t]*<!--\s*page[:\s]*(\d+)\s*-->[ \t]*$", re.IGNORECASE | re.MULTILINE)


class ScannedPDFError(ValueError):
    """The PDF has (almost) no text layer: it needs OCR (the Gemini parser)."""


@dataclass
class ParsedDocument:
    markdown: str
    parser: str
    pages: int | None = None
    sha256: str = ""
    warnings: list[str] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _open_pdf(data: bytes) -> PdfReader:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF is password-protected; remove the password before ingesting it")
    return reader


def _cache_path(digest: str, cfg: IngestionConfig) -> Path:
    prompt_sha = load_prompt(PDF_PROMPT).sha
    name = f"{digest[:24]}-gemini-{prompt_sha}-b{cfg.pdf_pages_per_batch}.md"
    return get_settings().state_dir / "parse_cache" / name


async def parse_document(filename: str, data: bytes, cfg: IngestionConfig | None = None) -> ParsedDocument:
    cfg = cfg or get_rag_config().ingestion
    suffix = Path(filename).suffix.lower()
    digest = sha256_bytes(data)
    if suffix in {".md", ".markdown"}:
        return ParsedDocument(markdown=data.decode("utf-8"), parser="markdown", sha256=digest)
    if suffix == ".txt":
        return ParsedDocument(markdown=data.decode("utf-8", errors="replace"), parser="text", sha256=digest)
    if suffix != ".pdf":
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}")

    pages = len(_open_pdf(data).pages)
    warnings: list[str] = []
    if cfg.pdf_parser == "gemini" and not get_settings().ragops_offline:
        cache = _cache_path(digest, cfg)
        if cache.exists():
            return ParsedDocument(cache.read_text(encoding="utf-8"), "gemini(cached)", pages, digest)
        try:
            markdown = await parse_pdf_with_gemini(data, cfg.pdf_pages_per_batch, cfg.pdf_parse_concurrency)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(markdown, encoding="utf-8")
            return ParsedDocument(markdown, "gemini", pages, digest)
        except Exception as exc:  # degrade gracefully, never fail ingestion on the parser
            log.warning("Gemini PDF parsing failed (%s); falling back to pypdf", exc)
            warnings.append(f"gemini parser failed: {type(exc).__name__}: {exc}"[:300])
    return ParsedDocument(parse_pdf_with_pypdf(data), "pypdf", pages, digest, warnings)


# --------------------------------------------------------------------------------------
# Gemini (multimodal) parsing in page batches
# --------------------------------------------------------------------------------------
def split_pdf(data: bytes, pages_per_batch: int) -> list[tuple[int, int, bytes]]:
    """Split a PDF into (first_page, last_page, pdf_bytes) batches, 1-based inclusive."""
    reader = _open_pdf(data)
    n = len(reader.pages)
    if n <= pages_per_batch:
        return [(1, n, data)]
    batches = []
    for start in range(0, n, pages_per_batch):
        end = min(n, start + pages_per_batch)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        batches.append((start + 1, end, buf.getvalue()))
    return batches


def normalize_page_markers(text: str, first: int, last: int) -> str:
    """Make a batch's page markers absolute, in range, and guarantee a leading marker."""
    nums = [int(n) for n in PAGE_MARKER_RE.findall(text)]
    # Models sometimes number pages relative to the attached excerpt (1..k).
    offset = first - 1 if nums and first > 1 and min(nums) == 1 and max(nums) <= last - first + 1 else 0
    text = PAGE_MARKER_RE.sub(lambda m: PAGE_MARKER.format(n=min(max(int(m.group(1)) + offset, first), last)), text)
    if not PAGE_MARKER_RE.match(text.lstrip().split("\n", 1)[0]):
        text = f"{PAGE_MARKER.format(n=first)}\n{text}"
    return text


async def _parse_batch(first: int, last: int, pdf_bytes: bytes) -> str:
    from ragops.providers.router import get_router

    prompt = load_prompt(PDF_PROMPT)
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    result = await get_router().complete(
        route="pdf_parsing",
        messages=[
            {"role": "system", "content": prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{b64}"}},
                    {"type": "text", "text": prompt.render_user(first_page=str(first), last_page=str(last))},
                ],
            },
        ],
        max_tokens=32000,
        span_name="parse_pdf_batch",
    )
    text = result.text.strip()
    # Models occasionally wrap output in a fence despite instructions.
    text = re.sub(r"^```(?:markdown|md)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    if len(PAGE_MARKER_RE.sub("", text).strip()) < 10:
        raise ValueError(f"parser returned (almost) no content for pages {first}-{last}")
    return normalize_page_markers(text, first, last)


async def parse_pdf_with_gemini(data: bytes, pages_per_batch: int = 10, concurrency: int = 4) -> str:
    batches = split_pdf(data, pages_per_batch)
    sem = asyncio.Semaphore(concurrency)

    async def run(first: int, last: int, chunk: bytes) -> str:
        async with sem:
            return await _parse_batch(first, last, chunk)

    parts = await asyncio.gather(*(run(*b) for b in batches))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Local fallback: layout-aware pypdf extraction
# --------------------------------------------------------------------------------------
_NUMBERED_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^\n]{1,100})$")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_TRAILING_PAGE_RE = re.compile(r"\|\s*page\s+\d+\s*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[-•▪●*–]\s+")
_BOLD_HINTS = ("bold", "black", "heavy", "semibold", "demi")


@dataclass
class _Line:
    text: str
    size: float | None
    bold: bool
    page: int


def _page_lines(page, page_no: int) -> list[_Line]:
    """Extract lines with their effective font size and weight.

    ``extract_text`` gives well-spaced lines; the visitor gives per-fragment font data.
    The two are aligned on their whitespace-free character streams.
    """
    chunks: list[str] = []
    meta: list[tuple[float, bool]] = []

    def visitor(text, cm, tm, font, font_size):
        t = re.sub(r"\s+", "", text or "")
        if not t:
            return
        try:
            scale = math.hypot(tm[2], tm[3]) * math.hypot(cm[2], cm[3])
        except (TypeError, IndexError):
            scale = 1.0
        size = round(float(font_size or 0) * (scale or 1.0), 1)
        base = str(font.get("/BaseFont", "")) if hasattr(font, "get") else ""
        chunks.append(t)
        meta.extend([(size, any(h in base.lower() for h in _BOLD_HINTS))] * len(t))

    text = page.extract_text(visitor_text=visitor) or ""
    flat, pos, lines = "".join(chunks), 0, []
    for raw in text.splitlines():
        line = raw.strip()
        target = re.sub(r"\s+", "", line)
        if not target:
            continue
        idx = flat.find(target, pos)
        if idx < 0:
            idx = flat.find(target)
        if idx < 0:
            lines.append(_Line(line, None, False, page_no))
            continue
        m = meta[idx : idx + len(target)]
        pos = idx + len(target)
        lines.append(_Line(line, max(s for s, _ in m) or None, all(b for _, b in m), page_no))
    return lines


def _edge_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _running_headers(pages: list[list[_Line]]) -> set[str]:
    """Lines that repeat at the top/bottom of most pages (headers, footers, watermarks)."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({_edge_key(ln.text) for ln in lines[:2] + lines[-2:]})
    return {k for k, c in counts.items() if c >= max(3, 0.5 * len(pages))}


def parse_pdf_with_pypdf(data: bytes) -> str:
    reader = _open_pdf(data)
    pages = [_page_lines(p, i) for i, p in enumerate(reader.pages, start=1)]

    total_chars = sum(len(ln.text) for lines in pages for ln in lines)
    if pages and total_chars / len(pages) < 25:
        raise ScannedPDFError(
            f"{len(pages)} page(s) with almost no extractable text: this looks like a scanned or "
            "image-only PDF. Enable the Gemini parser (PDF_PARSER=gemini with Vertex AI credentials) for OCR."
        )

    running = _running_headers(pages)
    sizes: Counter[float] = Counter()
    for lines in pages:
        for ln in lines:
            if ln.size:
                sizes[ln.size] += len(ln.text)
    body = sizes.most_common(1)[0][0] if sizes else None

    def is_short(t: str) -> bool:
        return len(t) <= 120 and len(t.split()) <= 16 and not t.endswith((".", ",", ";"))

    def keep(ln: _Line) -> bool:
        return not (_PAGE_NUMBER_RE.match(ln.text) or _TRAILING_PAGE_RE.search(ln.text) or _edge_key(ln.text) in running)

    # Heading levels from font-size ranks. The largest size is the document title only
    # if it is rare (a title appears once or twice); otherwise ranks start at ##.
    big = Counter(ln.size for lines in pages for ln in lines if keep(ln) and body and ln.size and ln.size >= body * 1.15 and is_short(ln.text))
    ranks = sorted(big, reverse=True)
    start_level = 1 if ranks and big[ranks[0]] <= 2 else 2
    level_for_size = {s: min(start_level + i, 3) for i, s in enumerate(ranks)}
    # Same-size bold lines (common for unnumbered subsections) sit one level below.
    bold_level = min(max(level_for_size.values(), default=1) + 1, 3)

    def is_bold_heading(ln: _Line) -> bool:
        return bool(
            ln.bold
            and body
            and ln.size
            and ln.size >= body * 0.98
            and is_short(ln.text)
            and len(ln.text.split()) <= 10
            and ln.text[:1].isupper()
        )

    out: list[str] = []
    paragraph: list[str] = []
    has_title = False
    # (index in `out`, level, font size, page) of a heading with no body after it yet.
    open_heading: tuple[int, int, float | None, int] | None = None

    def flush() -> None:
        if paragraph:
            out.extend([" ".join(paragraph), ""])
            paragraph.clear()

    for lines in pages:
        flush()
        open_heading = None
        if lines:
            out.extend([PAGE_MARKER.format(n=lines[0].page), ""])
        for ln in lines:
            if not keep(ln):
                continue
            text = ln.text
            level = None
            numbered = _NUMBERED_RE.match(text)
            if ln.size and ln.size in level_for_size:
                level = level_for_size[ln.size]
            elif numbered and is_short(text) and (ln.bold or (body and ln.size and ln.size >= body * 1.05)):
                level = min(2 + numbered.group(1).count("."), 4)
            elif is_bold_heading(ln):
                level = bold_level
            if level is not None:
                flush()
                if open_heading and not numbered and open_heading[1:] == (level, ln.size, ln.page):
                    # A long heading wrapped onto several lines: join it back together.
                    out[open_heading[0]] += f" {text}"
                    continue
                if level == 1 and has_title:
                    level = 2
                has_title = has_title or level == 1
                out.extend([f"{'#' * level} {text}", ""])
                open_heading = (len(out) - 2, level, ln.size, ln.page)
                continue
            open_heading = None
            if _BULLET_RE.match(text):
                flush()
                paragraph.append("- " + _BULLET_RE.sub("", text))
            else:
                # A wrapped line continues the paragraph unless the previous line ended a
                # sentence and this one starts a new capitalised sentence.
                if paragraph and paragraph[-1].endswith((".", ":")) and text[:1].isupper() and not paragraph[-1].startswith("- "):
                    flush()
                paragraph.append(text)
    flush()

    if not has_title:
        # No visually distinct title: promote the first content line if it looks like one.
        for i, line in enumerate(out):
            if line and not PAGE_MARKER_RE.match(line):
                if not line.startswith("#") and is_short(line):
                    out[i] = f"# {line}"
                break
    return "\n".join(out).strip() + "\n"
