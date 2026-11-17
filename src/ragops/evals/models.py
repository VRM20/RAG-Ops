"""Evaluation data models: golden cases, per-case results, run summaries, comparisons."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ragops.types import PipelineOverrides

QuestionType = Literal["factual", "procedural", "multi_hop", "comparative"]


class GoldenCase(BaseModel):
    id: str
    question: str
    expected_answer: str
    source_doc_ids: list[str] = Field(default_factory=list)
    question_type: QuestionType = "factual"
    user_id: str = "eval.bot"
    origin: Literal["curated", "synthetic"] = "curated"
    source_section: str | None = None


class MetricScore(BaseModel):
    score: float | None = None
    reason: str | None = None
    error: str | None = None


class CaseContext(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    text: str
    rerank_score: float | None = None


class EvalCaseResult(BaseModel):
    case_id: str
    question: str
    expected_answer: str
    question_type: str
    origin: str = "curated"
    answer: str = ""
    source_doc_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    contexts: list[CaseContext] = Field(default_factory=list)
    metrics: dict[str, MetricScore] = Field(default_factory=dict)
    doc_recall: bool = False
    latency_ms: float = 0.0
    error: str | None = None


class MetricSummary(BaseModel):
    mean: float | None = None
    threshold: float
    blocking: bool
    passed: bool
    n: int = 0
    baseline: float | None = None
    delta: float | None = None
    regressed: bool = False


class AclSummary(BaseModel):
    cases: int = 0
    leaked_chunks: int = 0
    passed: bool = True
    details: list[dict] = Field(default_factory=list)


class RunConfig(BaseModel):
    index_name: str = ""
    embedding_model: str = ""
    chunking: dict = Field(default_factory=dict)
    prompt_name: str = ""
    prompt_sha: str = ""
    synthesis_route: str = ""
    synthesis_model: str = ""
    judge_model: str = ""
    rerank_model: str | None = None
    git_sha: str | None = None
    git_branch: str | None = None
    offline: bool = False
    overrides: PipelineOverrides | None = None


class EvalRunSummary(BaseModel):
    run_id: str
    label: str = ""
    status: Literal["running", "completed", "failed"] = "running"
    created_at: str
    completed_at: str | None = None
    duration_s: float | None = None
    num_cases: int = 0
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    gate_passed: bool | None = None
    gate_failures: list[str] = Field(default_factory=list)
    acl: AclSummary = AclSummary()
    config: RunConfig = RunConfig()
    # Cases where the reranker failed and retrieval degraded to RRF order.
    rerank_fallbacks: int = 0
    error: str | None = None


class EvalRun(EvalRunSummary):
    cases: list[EvalCaseResult] = Field(default_factory=list)

    def summary(self) -> EvalRunSummary:
        return EvalRunSummary.model_validate(self.model_dump(exclude={"cases"}))


class MetricDelta(BaseModel):
    base: float | None
    head: float | None
    delta: float | None


class CaseDelta(BaseModel):
    case_id: str
    question: str
    base: dict[str, float | None]
    head: dict[str, float | None]
    delta: dict[str, float | None]
    base_answer: str = ""
    head_answer: str = ""


class EvalComparison(BaseModel):
    base: EvalRunSummary
    head: EvalRunSummary
    metric_deltas: dict[str, MetricDelta]
    cases: list[CaseDelta]
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
