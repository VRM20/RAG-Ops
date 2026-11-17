"""Synthetic golden-dataset generation with Gemini (route ``synthetic_data``).

Pipeline per document section:
1. **Generate** - Gemini writes N question/answer pairs grounded in the section, mixing
   factual / procedural / multi-hop / comparative questions (prompt: synth_qa_v1).
2. **Critique** - a second pass (route ``judge``) rejects items whose answer is not
   fully supported by the excerpt, or whose question is ambiguous or not standalone.
   Unfiltered synthetic data produces noisy gates; this step is what makes a
   synthetic set trustworthy enough to block merges on.
3. **De-duplicate** - near-duplicate questions (token Jaccard > 0.8) are dropped.

Generation uses the Markdown *sources* (ground truth), not parsed PDFs, so parser
errors cannot leak into the reference answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ragops.config import EVALS_DIR
from ragops.evals.models import GoldenCase
from ragops.generation.prompts import load_prompt
from ragops.ingestion.chunking import merge_small_sections, split_sections
from ragops.ingestion.pipeline import CORPUS_DIR, load_manifest
from ragops.providers.router import get_router
from ragops.retrieval.sparse import tokenize

log = logging.getLogger(__name__)

SYNTHETIC_PATH = EVALS_DIR / "datasets" / "golden_synthetic.jsonl"


class _Item(BaseModel):
    question: str
    expected_answer: str
    question_type: Literal["factual", "procedural", "multi_hop", "comparative"]


class _Items(BaseModel):
    items: list[_Item]


class _Verdict(BaseModel):
    keep: bool
    reason: str


_CRITIC_PROMPT = """You are reviewing a candidate evaluation item for an enterprise RAG system.

Excerpt:
\"\"\"
{excerpt}
\"\"\"

Question: {question}
Reference answer: {answer}

Keep the item only if ALL are true:
1. Every fact in the reference answer is explicitly stated in the excerpt.
2. The question is unambiguous and answerable without seeing the excerpt (it does not say
   "this section", "the excerpt" or similar).
3. The reference answer fully answers the question.

Respond with JSON only: {{"keep": true|false, "reason": "<one sentence>"}}"""


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


async def _generate_for_section(doc_title: str, excerpt: str, n: int) -> list[_Item]:
    prompt = load_prompt("synth_qa_v1")
    result = await get_router().complete(
        "synthetic_data",
        [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.render_user(doc_title=doc_title, excerpt=excerpt, n=str(n))},
        ],
        max_tokens=8192,
        response_format=_Items,
    )
    return _Items.model_validate_json(result.text).items


async def _critic(excerpt: str, item: _Item) -> _Verdict:
    result = await get_router().complete(
        "judge",
        [{"role": "user", "content": _CRITIC_PROMPT.format(excerpt=excerpt, question=item.question, answer=item.expected_answer)}],
        max_tokens=4096,
        response_format=_Verdict,
    )
    return _Verdict.model_validate_json(result.text)


async def generate_synthetic_dataset(
    per_section: int = 2,
    min_section_tokens: int = 40,
    critic: bool = True,
    concurrency: int = 4,
    out_path: Path = SYNTHETIC_PATH,
) -> list[GoldenCase]:
    sem = asyncio.Semaphore(concurrency)
    jobs: list[tuple[str, str, str]] = []  # (doc_id, title, section text)
    for doc in load_manifest():
        markdown = (CORPUS_DIR / doc.source).read_text(encoding="utf-8")
        _, sections = split_sections(markdown)
        for sec in merge_small_sections(sections, min_tokens=60, max_tokens=600):
            if sec.tokens >= min_section_tokens:
                header = " > ".join(sec.path) or "Overview"
                jobs.append((doc.doc_id, doc.title, f"[{header}]\n{sec.text}"))

    async def run(job: tuple[str, str, str]) -> list[tuple[str, str, _Item]]:
        doc_id, title, excerpt = job
        async with sem:
            try:
                items = await _generate_for_section(title, excerpt, per_section)
            except Exception as exc:
                log.warning("generation failed for %s: %s", doc_id, exc)
                return []
            kept = []
            for item in items:
                if critic:
                    try:
                        verdict = await _critic(excerpt, item)
                    except Exception as exc:
                        log.warning("critic failed (%s); dropping item", exc)
                        continue
                    if not verdict.keep:
                        log.info("critic rejected: %s (%s)", item.question, verdict.reason)
                        continue
                kept.append((doc_id, excerpt.split("\n", 1)[0].strip("[]"), item))
            return kept

    results = [x for batch in await asyncio.gather(*(run(j) for j in jobs)) for x in batch]

    cases: list[GoldenCase] = []
    counters: dict[str, int] = {}
    for doc_id, section, item in results:
        if any(_jaccard(item.question, c.question) > 0.8 for c in cases):
            continue
        counters[doc_id] = counters.get(doc_id, 0) + 1
        cases.append(
            GoldenCase(
                id=f"syn-{doc_id}-{counters[doc_id]:02d}",
                question=item.question,
                expected_answer=item.expected_answer,
                source_doc_ids=[doc_id],
                question_type=item.question_type,
                user_id="eval.bot",
                origin="synthetic",
                source_section=section,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.model_dump(exclude={"origin"}), ensure_ascii=False) + "\n")
    log.info("Wrote %d synthetic cases from %d sections to %s", len(cases), len(jobs), out_path)
    return cases
