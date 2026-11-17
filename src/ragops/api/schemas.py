"""Request / response schemas of the HTTP API (the contract with the Next.js frontend)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ragops.types import PipelineOverrides, RetrievedChunk


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    user_id: str
    overrides: PipelineOverrides | None = None


class RetrieveResponse(BaseModel):
    query: str
    user_id: str
    groups: list[str]
    contexts: list[RetrievedChunk]
    candidates: list[RetrievedChunk]  # fused list before reranking (for the inspector)
    timings_ms: dict[str, float]


class ProviderStatus(BaseModel):
    vertex: bool
    bedrock: bool
    cohere: bool
    guardrail: Literal["bedrock", "local"]
    qdrant: Literal["server", "embedded"]
    task_queue: Literal["inline", "celery"]
    tracing: bool


class IndexStatus(BaseModel):
    name: str
    chunks: int
    documents: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    offline: bool
    providers: ProviderStatus
    index: IndexStatus


class RouteInfo(BaseModel):
    primary: str
    fallbacks: list[str]
    models: list[str]  # resolved LiteLLM model ids, primary first


class ConfigResponse(BaseModel):
    rag: dict
    routes: dict[str, RouteInfo]
    gates: dict
    prompt: dict


class PromptResponse(BaseModel):
    name: str
    description: str
    system: str
    user_template: str
    sha: str


class UserResponse(BaseModel):
    user_id: str
    name: str
    title: str
    groups: list[str]


class StartEvalRequest(BaseModel):
    label: str = ""
    max_cases: int | None = Field(default=None, ge=1, le=500)
    dataset: Literal["all", "curated", "synthetic"] = "all"
    overrides: PipelineOverrides | None = None


class ScoreAnswerRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    expected_answer: str | None = None


class ScoreAnswerResponse(BaseModel):
    metrics: dict[str, dict]  # name -> {score, reason, threshold, passed}
    judge_model: str


class IngestCorpusResponse(BaseModel):
    queued: list[str]
