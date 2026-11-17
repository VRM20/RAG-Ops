"""Core domain types shared by ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of a document."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: list[str] = Field(default_factory=list)
    text: str
    # What is embedded / BM25-indexed: the text prefixed with its contextual header.
    embed_text: str
    acl_groups: list[str] = Field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0
    # 1-based page range in the source PDF (None for Markdown/text sources).
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def section(self) -> str:
        return " > ".join(self.section_path)

    @property
    def page_label(self) -> str | None:
        if self.page_start is None:
            return None
        if self.page_end is None or self.page_end == self.page_start:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


class RetrievedChunk(BaseModel):
    """A chunk plus the full scoring trail that led to its selection."""

    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_rank: int = 0


class UserContext(BaseModel):
    user_id: str
    name: str = ""
    groups: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    source_number: int
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    page_start: int | None = None
    page_end: int | None = None


class GuardrailVerdict(BaseModel):
    stage: Literal["input", "output"]
    provider: Literal["bedrock", "local", "disabled"]
    action: Literal["allow", "block", "anonymize", "flag"] = "allow"
    reasons: list[str] = Field(default_factory=list)
    sanitized_text: str | None = None
    grounding_score: float | None = None
    relevance_score: float | None = None
    latency_ms: float = 0.0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


class RAGResponse(BaseModel):
    query: str
    answer: str
    blocked: bool = False
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    guardrails: list[GuardrailVerdict] = Field(default_factory=list)
    model: str | None = None
    route: str | None = None
    usage: Usage = Usage()
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None


class PipelineOverrides(BaseModel):
    """Per-request overrides used by the prompt sandbox and experiments.

    Only generation-time and retrieval-time knobs can be overridden; ingestion knobs
    (chunking, embedding model) need a re-index and are changed in config/rag.yaml.
    """

    system_prompt: str | None = None
    user_template: str | None = None
    route: str | None = None
    rerank_enabled: bool | None = None
    rerank_top_n: int | None = None
    fused_top_k: int | None = None
    max_context_chunks: int | None = None
    guardrails_enabled: bool | None = None
