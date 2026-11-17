import numpy as np

from ragops.config import ChunkingConfig
from ragops.ingestion.chunking import (
    Chunker,
    count_tokens,
    merge_small_sections,
    split_sections,
    split_sentences,
    split_units,
)

DOC = """# Leave Handbook

> Fictional preamble.

## 4. Leave Policy

### 4.1 Privileged Leave

Employees earn 24 days of privileged leave per year. Unused leave can be carried forward up to 45 days.

### 4.2 Sick Leave

Employees receive 12 days of sick leave per year.

## 5. Hotel Limits

| City tier | M1 to M3 | M4+ |
| --- | --- | --- |
| Tier-1 | INR 7,500 | INR 11,000 |
| Tier-2 | INR 4,500 | INR 7,000 |
"""


def test_split_sections_tracks_heading_paths_and_title():
    title, sections = split_sections(DOC)
    assert title == "Leave Handbook"
    paths = [s.path for s in sections]
    assert [] in paths  # preamble
    assert ["4. Leave Policy", "4.1 Privileged Leave"] in paths
    assert ["5. Hotel Limits"] in paths
    # a heading with no body of its own does not become an empty section
    assert ["4. Leave Policy"] not in paths


def test_merge_small_sections_keeps_common_parent_path():
    _, sections = split_sections(DOC)
    merged = merge_small_sections(sections, min_tokens=20, max_tokens=200)
    leave = [s for s in merged if s.path[:1] == ["4. Leave Policy"]]
    # 4.1 and 4.2 are tiny siblings -> merged under their parent, headings kept inline
    assert any("### 4.2 Sick Leave" in s.text and "24 days" in s.text for s in leave)


def test_sentence_splitter_respects_abbreviations():
    s = split_sentences("Rates are 14% p.a. for MSME loans. E.g. a two-wheeler loan differs. Done.")
    assert s[0] == "Rates are 14% p.a. for MSME loans."
    assert len(s) == 3


def test_tables_are_kept_whole_or_split_with_header_repeated():
    table = DOC.split("## 5. Hotel Limits")[1].strip()
    assert split_units(table, max_tokens=500) == [table]
    parts = split_units(table, max_tokens=25)
    assert len(parts) == 2
    assert all(p.startswith("| City tier") for p in parts)


async def test_chunker_adds_contextual_header_and_acl():
    chunks = await Chunker(ChunkingConfig(strategy="structural", max_tokens=200, min_tokens=5)).chunk(
        doc_id="hb", markdown=DOC, acl_groups=["hr", "all-employees", "hr"]
    )
    sick = next(c for c in chunks if "12 days of sick leave" in c.text)
    assert sick.embed_text.startswith("Document: Leave Handbook\nSection: 4. Leave Policy > 4.2 Sick Leave")
    assert sick.acl_groups == ["all-employees", "hr"]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "hb::0000"


async def test_semantic_strategy_splits_on_topic_shift():
    long_section = "# T\n\n## S\n\n" + " ".join(["Leave accrues monthly."] * 6 + ["Laptops use BitLocker encryption."] * 6)

    async def fake_embed(texts):
        # two clearly separated "topics"
        return [[1.0, 0.0] if "Leave" in t else [0.0, 1.0] for t in texts]

    # The section (~54 tokens) exceeds max_tokens, so it must be split; each topic
    # (24 and 30 tokens) fits, so the only cut should be exactly at the topic shift.
    cfg = ChunkingConfig(strategy="semantic", max_tokens=40, min_tokens=5, semantic_breakpoint_percentile=90)
    chunks = await Chunker(cfg, embed_fn=fake_embed).chunk(doc_id="x", markdown=long_section, acl_groups=["public"])
    assert len(chunks) == 2
    assert "Leave" in chunks[0].text and "BitLocker" not in chunks[0].text
    assert "BitLocker" in chunks[1].text and "Leave" not in chunks[1].text


async def test_oversized_section_respects_max_tokens():
    body = " ".join(f"Sentence number {i} talks about policy clause {i}." for i in range(80))
    md = f"# Doc\n\n## Big\n\n{body}"

    async def embed(texts):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), 8)).tolist()

    for strategy in ("semantic", "structural", "fixed"):
        cfg = ChunkingConfig(strategy=strategy, max_tokens=120, min_tokens=10, overlap_tokens=20)
        chunks = await Chunker(cfg, embed_fn=embed).chunk(doc_id="d", markdown=md, acl_groups=["public"])
        assert len(chunks) > 3, strategy
        assert all(count_tokens(c.text) <= 120 for c in chunks), strategy
