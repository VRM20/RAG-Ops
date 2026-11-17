"""OpenTelemetry tracing with OpenInference semantic conventions.

Every RAG request produces one trace:

    rag.answer (CHAIN)
    |- guardrail.input   (GUARDRAIL)
    |- retrieval.hybrid  (RETRIEVER)   query, dense/sparse/fused documents with scores
    |- rerank            (RERANKER)    input/output documents, model, top_n
    |- litellm completion (LLM)        auto-instrumented: model, prompt, tokens, cost
    `- guardrail.output  (GUARDRAIL)   grounding / relevance scores

OpenInference attributes make Phoenix (and Langfuse, Arize) render retrieved
documents, rerank scores and LLM messages natively. Export is plain OTLP/HTTP, so the
backend is a config choice. Tracing is a no-op when no endpoint is configured.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

from ragops.config import get_settings

log = logging.getLogger(__name__)

SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
METADATA = "metadata"
USER_ID = "user.id"

_enabled = False


def _parse_headers(raw: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for part in (raw or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            headers[k.strip()] = v.strip()
    return headers


def setup_tracing(service_name: str | None = None) -> bool:
    """Configure the global tracer provider once per process. Returns True if enabled."""
    global _enabled
    if _enabled:
        return True
    s = get_settings()
    if not s.otel_exporter_otlp_traces_endpoint:
        return False
    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        name = service_name or s.otel_service_name
        resource = Resource.create({"service.name": name, "openinference.project.name": name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=s.otel_exporter_otlp_traces_endpoint, headers=_parse_headers(s.otel_exporter_otlp_headers)
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        LiteLLMInstrumentor().instrument(tracer_provider=provider)
        _enabled = True
        log.info("Tracing enabled -> %s", s.otel_exporter_otlp_traces_endpoint)
    except Exception as exc:  # observability must never take the service down
        log.warning("Tracing setup failed: %s", exc)
    return _enabled


def tracing_enabled() -> bool:
    return _enabled


def _tracer():
    return trace.get_tracer("ragops")


def _safe(value: Any) -> Any:
    if isinstance(value, str | bool | int | float):
        return value
    return json.dumps(value, default=str)[:20_000]


@contextmanager
def span(name: str, kind: str, input_value: Any = None, **attributes: Any) -> Iterator[trace.Span]:
    with _tracer().start_as_current_span(name) as sp:
        sp.set_attribute(SPAN_KIND, kind)
        if input_value is not None:
            sp.set_attribute(INPUT_VALUE, _safe(input_value))
        for key, value in attributes.items():
            if value is not None:
                sp.set_attribute(key, _safe(value))
        yield sp


def set_output(sp: trace.Span, value: Any) -> None:
    sp.set_attribute(OUTPUT_VALUE, _safe(value))


def set_documents(sp: trace.Span, prefix: str, docs: list[Any]) -> None:
    """Record retrieved/reranked chunks using OpenInference document attributes.

    ``prefix`` is ``retrieval.documents``, ``reranker.input_documents`` or
    ``reranker.output_documents``.
    """
    for i, d in enumerate(docs):
        chunk = d.chunk
        score = d.rerank_score if d.rerank_score is not None else d.rrf_score
        sp.set_attribute(f"{prefix}.{i}.document.id", chunk.chunk_id)
        sp.set_attribute(f"{prefix}.{i}.document.content", chunk.text[:4000])
        sp.set_attribute(f"{prefix}.{i}.document.score", float(score))
        sp.set_attribute(
            f"{prefix}.{i}.document.metadata",
            json.dumps({"doc_id": chunk.doc_id, "section": chunk.section, "dense_rank": d.dense_rank, "sparse_rank": d.sparse_rank}),
        )


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx and ctx.trace_id else None
