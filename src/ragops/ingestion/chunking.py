"""Structure-aware semantic chunking.

Fixed-size token windows cut through the middle of a policy clause, so the clause's
condition lands in one chunk and its exception in the next. This chunker instead:

1. **Splits by document structure.** Markdown headings define sections; each section
   keeps its full heading path (e.g. ``4. Leave Policy > 4.2 Sick Leave``).
2. **Merges tiny sections** (< ``min_tokens``) into their following sibling, so a
   two-line clause is not orphaned.
3. **Splits oversized sections semantically.** The section is broken into units
   (sentences, list items, tables). Consecutive units are embedded and the section is
   cut where the cosine distance between neighbours exceeds the configured percentile,
   i.e. where the topic shifts, while respecting ``max_tokens``. Tables are never cut
   mid-row; long tables are split by rows with the header repeated.
4. **Adds a contextual header.** ``embed_text`` = ``Document: ... / Section: ...`` +
   text. A chunk that says "the limit is 30 days" is otherwise unretrievable for "USB
   exception validity"; the header restores the missing context for both the dense
   and the BM25 index.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from ragops.config import ChunkingConfig
from ragops.types import Chunk

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PAGE_MARKER_RE = re.compile(r"^[ \t]*<!--\s*page[:\s]*(\d+)\s*-->[ \t]*$", re.IGNORECASE)
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_ABBREVIATIONS = ("e.g.", "i.e.", "p.a.", "etc.", "vs.", "no.", "approx.", "dr.", "mr.", "ms.")


def count_tokens(text: str) -> int:
    """Fast, deterministic token estimate (words + punctuation).

    Within ~10-15% of BPE tokenizers for English prose, which is plenty for chunk
    budgeting, and needs no tokenizer download or provider-specific vocabulary.
    """
    return len(_TOKEN_RE.findall(text))


# --------------------------------------------------------------------------------------
# Markdown -> sections
# --------------------------------------------------------------------------------------
@dataclass
class Section:
    path: list[str]
    heading_line: str | None
    body: str
    parts: list[Section] = field(default_factory=list)  # original sections, when merged
    page_start: int | None = None
    page_end: int | None = None

    @property
    def leaves(self) -> list[Section]:
        return self.parts or [self]

    @property
    def pages(self) -> tuple[int | None, int | None]:
        starts = [p.page_start for p in self.leaves if p.page_start is not None]
        ends = [p.page_end for p in self.leaves if p.page_end is not None]
        return (min(starts) if starts else None, max(ends) if ends else None)

    @property
    def text(self) -> str:
        if not self.parts:
            return self.body.strip()
        # A merged part keeps its own heading whenever the merged path no longer names it.
        rendered = [
            f"{p.heading_line}\n{p.body.strip()}" if p.heading_line and p.path != self.path else p.body.strip()
            for p in self.parts
        ]
        return "\n\n".join(rendered).strip()

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


def split_sections(markdown: str) -> tuple[str | None, list[Section]]:
    """Return (document title, sections). The first H1 is the document title.

    ``<!-- page N -->`` marker lines (from the PDF parsers) are consumed here: they never
    appear in chunk text, they set ``page_start`` / ``page_end`` on each section.
    """
    title: str | None = None
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    heading_line: str | None = None
    body: list[str] = []
    body_pages: list[int] = []
    page: int | None = None
    in_code = False

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append(
                Section(
                    path=[t for _, t in stack],
                    heading_line=heading_line,
                    body=text,
                    page_start=min(body_pages) if body_pages else None,
                    page_end=max(body_pages) if body_pages else None,
                )
            )
        body.clear()
        body_pages.clear()

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and (marker := _PAGE_MARKER_RE.match(line)):
            page = int(marker.group(1))
            continue
        match = None if in_code else _HEADING_RE.match(line)
        if not match:
            body.append(line)
            if line.strip() and page is not None:
                body_pages.append(page)
            continue
        flush()
        level, text = len(match.group(1)), match.group(2).strip()
        if level == 1 and title is None:
            title, stack, heading_line = text, [], None
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        heading_line = line.strip()
    flush()
    return title, sections


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return out


def _combine(a: Section, b: Section) -> Section:
    path = _common_prefix(a.path, b.path) or b.path
    return Section(path=path, heading_line=None, body="", parts=a.leaves + b.leaves)


def merge_small_sections(sections: list[Section], min_tokens: int, max_tokens: int) -> list[Section]:
    """Attach sections below ``min_tokens`` to a neighbour without crossing chapters.

    Preference order: merge *backward* into the previous sibling (shared parent), else
    *forward* into the next sibling. A heading-less preamble may attach forward to the
    first section. Otherwise the small section stays on its own rather than being
    mislabelled under an unrelated chapter.
    """
    out: list[Section] = []
    carry: Section | None = None
    for i, sec in enumerate(sections):
        if carry is not None:
            sec, carry = _combine(carry, sec), None
        if sec.tokens >= min_tokens:
            out.append(sec)
            continue
        prev = out[-1] if out else None
        nxt = sections[i + 1] if i + 1 < len(sections) else None
        if prev and _common_prefix(prev.path, sec.path) and prev.tokens + sec.tokens <= max_tokens:
            out[-1] = _combine(prev, sec)
        elif nxt and (_common_prefix(sec.path, nxt.path) or not sec.path) and sec.tokens + nxt.tokens <= max_tokens:
            carry = sec
        else:
            out.append(sec)
    return out


# --------------------------------------------------------------------------------------
# Section -> units (sentences, list items, tables)
# --------------------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'*])", text.strip())
    sentences: list[str] = []
    for piece in pieces:
        if sentences and sentences[-1].lower().endswith(_ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def _split_table(lines: list[str], max_tokens: int) -> list[str]:
    if count_tokens("\n".join(lines)) <= max_tokens or len(lines) <= 3:
        return ["\n".join(lines)]
    header, rows = lines[:2], lines[2:]
    out, current = [], []
    for row in rows:
        if current and count_tokens("\n".join(header + current + [row])) > max_tokens:
            out.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        out.append("\n".join(header + current))
    return out


def split_units(text: str, max_tokens: int) -> list[str]:
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if all(ln.lstrip().startswith("|") for ln in lines):
            units.extend(_split_table(lines, max_tokens))
        elif any(_LIST_RE.match(ln) for ln in lines):
            item: list[str] = []
            for ln in lines:
                if _LIST_RE.match(ln) and item:
                    units.append(" ".join(item))
                    item = []
                item.append(ln.strip())
            if item:
                units.append(" ".join(item))
        else:
            units.extend(split_sentences(" ".join(ln.strip() for ln in lines)))
    return units


def _pack(units: list[str], max_tokens: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and count_tokens("\n".join(current + [unit])) > max_tokens:
            groups.append(current)
            current = []
        current.append(unit)
    if current:
        groups.append(current)
    return groups


def _cosine_distances(vectors: list[list[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return 1.0 - np.sum(m[:-1] * m[1:], axis=1)


async def _semantic_groups(
    units: list[str], embed_fn: EmbedFn, max_tokens: int, percentile: float
) -> list[list[str]]:
    if len(units) < 3:
        return _pack(units, max_tokens)
    distances = _cosine_distances(await embed_fn(units))
    threshold = float(np.percentile(distances, percentile))
    groups: list[list[str]] = []
    current = [units[0]]
    for dist, unit in zip(distances, units[1:], strict=True):
        too_big = count_tokens("\n".join(current + [unit])) > max_tokens
        if too_big or dist > threshold:
            groups.append(current)
            current = []
        current.append(unit)
    groups.append(current)
    return groups


def _is_block(unit: str) -> bool:
    return unit.startswith("|") or bool(_LIST_RE.match(unit))


def _join_units(units: list[str]) -> str:
    """Tables and list items keep their own lines; consecutive sentences form a paragraph."""
    out: list[str] = []
    for unit in units:
        if out and not _is_block(unit) and not _is_block(out[-1]):
            out[-1] = f"{out[-1]} {unit}"
        else:
            out.append(unit)
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# Chunker
# --------------------------------------------------------------------------------------
def contextual_header(doc_title: str, path: list[str]) -> str:
    return f"Document: {doc_title}\nSection: {' > '.join(path) if path else 'Overview'}"


class Chunker:
    def __init__(self, cfg: ChunkingConfig, embed_fn: EmbedFn | None = None):
        self.cfg = cfg
        self.embed_fn = embed_fn

    async def chunk(
        self,
        *,
        doc_id: str,
        markdown: str,
        acl_groups: list[str],
        title: str | None = None,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        parsed_title, sections = split_sections(markdown)
        doc_title = title or parsed_title or doc_id
        # (section path, text, (page_start, page_end))
        pieces: list[tuple[list[str], str, tuple[int | None, int | None]]] = []

        if self.cfg.strategy == "fixed":
            pieces = [([], text, (None, None)) for text in self._fixed_windows(markdown)]
        else:
            for sec in merge_small_sections(sections, self.cfg.min_tokens, self.cfg.max_tokens):
                if sec.tokens <= self.cfg.max_tokens:
                    pieces.append((sec.path, sec.text, sec.pages))
                    continue
                units = split_units(sec.text, self.cfg.max_tokens)
                if self.cfg.strategy == "semantic" and self.embed_fn is not None:
                    groups = await _semantic_groups(
                        units, self.embed_fn, self.cfg.max_tokens, self.cfg.semantic_breakpoint_percentile
                    )
                else:
                    groups = _pack(units, self.cfg.max_tokens)
                # Pieces of a split section inherit the section's page range.
                pieces.extend((sec.path, _join_units(g), sec.pages) for g in groups)

        chunks = []
        for i, (path, text, (page_start, page_end)) in enumerate(p for p in pieces if p[1].strip()):
            header = contextual_header(doc_title, path)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{i:04d}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=path,
                    text=text,
                    embed_text=f"{header}\n\n{text}" if self.cfg.contextual_header else text,
                    acl_groups=sorted(set(acl_groups)),
                    chunk_index=i,
                    token_count=count_tokens(text),
                    page_start=page_start,
                    page_end=page_end,
                    metadata=dict(metadata or {}),
                )
            )
        return chunks

    def _fixed_windows(self, markdown: str) -> list[str]:
        """Structure-unaware baseline: sliding windows with overlap."""
        plain = "\n".join(
            ln for ln in markdown.splitlines() if not _HEADING_RE.match(ln) and not _PAGE_MARKER_RE.match(ln)
        )
        units = split_units(plain, self.cfg.max_tokens)
        windows, current = [], []
        for unit in units:
            if current and count_tokens(" ".join(current + [unit])) > self.cfg.max_tokens:
                windows.append(" ".join(current))
                overlap: list[str] = []
                for u in reversed(current):
                    if count_tokens(" ".join([u, *overlap])) > self.cfg.overlap_tokens:
                        break
                    overlap.insert(0, u)
                current = overlap
            current.append(unit)
        if current:
            windows.append(" ".join(current))
        return windows
