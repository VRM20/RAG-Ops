"""End-to-end RAG pipeline (blocking and streaming).

    query
      │  1. input guardrail   (Bedrock ApplyGuardrail: prompt attacks, PII)  -> block / anonymise
      ▼
      │  2. hybrid retrieval  (ACL-filtered dense + BM25 -> RRF -> rerank)
      ▼
      │  3. grounded generation (route "synthesis": Bedrock Claude, fallback Gemini)
      │     numbered sources -> answer with [n] citations or an explicit refusal
      ▼
      │  4. output guardrail  (PII leakage + contextual grounding score)
      ▼
    RAGResponse (answer, citations, contexts with full scoring trail, timings, usage)
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from typing import Any

from ragops.config import RagConfig, get_rag_config
from ragops.generation.prompts import PromptTemplate, load_prompt
from ragops.observability.tracing import current_trace_id, set_output, span
from ragops.providers.guardrails import BLOCKED_INPUT_MESSAGE, Guardrails
from ragops.providers.offline import REFUSAL
from ragops.providers.router import LLMResult, LLMRouter, get_router
from ragops.retrieval.retriever import HybridRetriever
from ragops.types import (
    Citation,
    GuardrailVerdict,
    PipelineOverrides,
    RAGResponse,
    RetrievedChunk,
    UserContext,
)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def render_context(contexts: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        label = " | ".join(x for x in (c.chunk.doc_title, c.chunk.section or "Overview", c.chunk.page_label) if x)
        blocks.append(f"[{i}] {label}\n{c.chunk.text}")
    return "\n\n".join(blocks) if blocks else "(no sources available)"


def extract_citations(answer: str, contexts: list[RetrievedChunk]) -> list[Citation]:
    seen: list[int] = []
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(contexts) and n not in seen:
            seen.append(n)
    return [
        Citation(
            source_number=n,
            chunk_id=contexts[n - 1].chunk.chunk_id,
            doc_id=contexts[n - 1].chunk.doc_id,
            doc_title=contexts[n - 1].chunk.doc_title,
            section=contexts[n - 1].chunk.section,
            page_start=contexts[n - 1].chunk.page_start,
            page_end=contexts[n - 1].chunk.page_end,
        )
        for n in sorted(seen)
    ]


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower().rstrip(".") in answer.lower()


class RAGPipeline:
    def __init__(
        self,
        cfg: RagConfig | None = None,
        retriever: HybridRetriever | None = None,
        router: LLMRouter | None = None,
        guardrails: Guardrails | None = None,
    ):
        self.cfg = cfg or get_rag_config()
        self.retriever = retriever or HybridRetriever(self.cfg)
        self.router = router or get_router()
        self.guardrails = guardrails or Guardrails()

    # ---- helpers -----------------------------------------------------------------------
    def prompt(self, o: PipelineOverrides) -> PromptTemplate:
        base = load_prompt(self.cfg.generation.prompt)
        if o.system_prompt or o.user_template:
            return PromptTemplate(
                name=f"{base.name}+override",
                description="sandbox override",
                system=o.system_prompt or base.system,
                user_template=o.user_template or base.user_template,
            )
        return base

    def build_messages(self, query: str, contexts: list[RetrievedChunk], prompt: PromptTemplate) -> list[dict]:
        return [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.render_user(context=render_context(contexts), question=query)},
        ]

    def _guards_on(self, o: PipelineOverrides, stage: str) -> bool:
        if o.guardrails_enabled is False:
            return False
        return getattr(self.cfg.guardrails, stage)

    async def _input_guard(self, query: str, o: PipelineOverrides) -> GuardrailVerdict:
        if not self._guards_on(o, "input"):
            return GuardrailVerdict(stage="input", provider="disabled")
        with span("guardrail.input", "GUARDRAIL", input_value=query) as sp:
            verdict = await self.guardrails.check_input(query)
            set_output(sp, verdict.model_dump())
        return verdict

    async def _output_guard(self, answer: str, query: str, contexts: list[RetrievedChunk], o: PipelineOverrides) -> GuardrailVerdict | None:
        if not self._guards_on(o, "output") or is_refusal(answer):
            return None
        with span("guardrail.output", "GUARDRAIL", input_value=answer) as sp:
            verdict = await self.guardrails.check_output(answer, query, [c.chunk.text for c in contexts])
            set_output(sp, verdict.model_dump())
        return verdict

    def _finalize(
        self,
        resp: RAGResponse,
        result: LLMResult,
        out_verdict: GuardrailVerdict | None,
        started: float,
    ) -> RAGResponse:
        answer = result.text.strip()
        if out_verdict:
            resp.guardrails.append(out_verdict)
            if out_verdict.action == "anonymize" and out_verdict.sanitized_text:
                answer = out_verdict.sanitized_text
        resp.answer = answer
        resp.citations = [] if is_refusal(answer) else extract_citations(answer, resp.contexts)
        resp.model, resp.route, resp.usage = result.model, result.route, result.usage
        resp.timings_ms["generation"] = result.latency_ms
        resp.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
        return resp

    # ---- blocking ----------------------------------------------------------------------
    async def answer(self, query: str, user: UserContext, overrides: PipelineOverrides | None = None) -> RAGResponse:
        o = overrides or PipelineOverrides()
        started = time.perf_counter()
        route = o.route or self.cfg.generation.route
        with span("rag.answer", "CHAIN", input_value=query, **{"user.id": user.user_id}) as root:
            resp = RAGResponse(query=query, answer="", route=route, trace_id=current_trace_id())

            verdict = await self._input_guard(query, o)
            resp.guardrails.append(verdict)
            resp.timings_ms["guardrail_input"] = verdict.latency_ms
            if verdict.action == "block":
                resp.answer, resp.blocked = BLOCKED_INPUT_MESSAGE, True
                set_output(root, resp.answer)
                return resp
            effective_query = verdict.sanitized_text or query

            retrieval = await self.retriever.retrieve(effective_query, user, o)
            resp.contexts = retrieval.contexts
            resp.timings_ms.update(retrieval.timings_ms)

            messages = self.build_messages(effective_query, resp.contexts, self.prompt(o))
            result = await self.router.complete(route, messages, max_tokens=self.cfg.generation.max_tokens, span_name="generate")
            out_verdict = await self._output_guard(result.text, effective_query, resp.contexts, o)
            resp = self._finalize(resp, result, out_verdict, started)
            set_output(root, resp.answer)
            return resp

    # ---- streaming ---------------------------------------------------------------------
    async def stream(
        self, query: str, user: UserContext, overrides: PipelineOverrides | None = None
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield (event, payload) tuples: guardrail, blocked | sources, token*, done."""
        o = overrides or PipelineOverrides()
        started = time.perf_counter()
        route = o.route or self.cfg.generation.route
        with span("rag.answer", "CHAIN", input_value=query, **{"user.id": user.user_id, "streaming": True}) as root:
            resp = RAGResponse(query=query, answer="", route=route, trace_id=current_trace_id())

            verdict = await self._input_guard(query, o)
            resp.guardrails.append(verdict)
            resp.timings_ms["guardrail_input"] = verdict.latency_ms
            yield "guardrail", verdict.model_dump()
            if verdict.action == "block":
                yield "blocked", {"message": BLOCKED_INPUT_MESSAGE, "guardrail": verdict.model_dump()}
                return
            effective_query = verdict.sanitized_text or query

            retrieval = await self.retriever.retrieve(effective_query, user, o)
            resp.contexts = retrieval.contexts
            resp.timings_ms.update(retrieval.timings_ms)
            yield "sources", {"contexts": [c.model_dump() for c in resp.contexts]}

            messages = self.build_messages(effective_query, resp.contexts, self.prompt(o))
            result: LLMResult | None = None
            async for item in self.router.stream(route, messages, max_tokens=self.cfg.generation.max_tokens):
                if isinstance(item, LLMResult):
                    result = item
                else:
                    yield "token", {"text": item}
            assert result is not None
            out_verdict = await self._output_guard(result.text, effective_query, resp.contexts, o)
            resp = self._finalize(resp, result, out_verdict, started)
            set_output(root, resp.answer)
            yield "done", resp.model_dump(include={"answer", "citations", "guardrails", "model", "route", "usage", "timings_ms", "trace_id"})
