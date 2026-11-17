// TypeScript mirrors of the backend pydantic schemas.
// Sources: src/ragops/types.py, src/ragops/api/schemas.py, src/ragops/evals/models.py,
// src/ragops/ingestion/registry.py. Field names are snake_case exactly as the API returns them.

// ---- core types (ragops/types.py) ------------------------------------------------------
export interface Chunk {
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  section_path: string[];
  text: string;
  embed_text: string;
  acl_groups: string[];
  chunk_index: number;
  token_count: number;
  /** 1-based page range in the source PDF; null for Markdown/text sources. */
  page_start: number | null;
  page_end: number | null;
  metadata: Record<string, unknown>;
}

export interface RetrievedChunk {
  chunk: Chunk;
  dense_rank: number | null;
  dense_score: number | null;
  sparse_rank: number | null;
  sparse_score: number | null;
  rrf_score: number;
  rerank_score: number | null;
  final_rank: number;
}

export interface UserContext {
  user_id: string;
  name: string;
  groups: string[];
}

export interface Citation {
  source_number: number;
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  section: string;
  page_start: number | null;
  page_end: number | null;
}

export type GuardrailAction = "allow" | "block" | "anonymize" | "flag";

export interface GuardrailVerdict {
  stage: "input" | "output";
  provider: "bedrock" | "local" | "disabled";
  action: GuardrailAction;
  reasons: string[];
  sanitized_text: string | null;
  grounding_score: number | null;
  relevance_score: number | null;
  latency_ms: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
}

export interface RAGResponse {
  query: string;
  answer: string;
  blocked: boolean;
  contexts: RetrievedChunk[];
  citations: Citation[];
  guardrails: GuardrailVerdict[];
  model: string | null;
  route: string | null;
  usage: Usage;
  timings_ms: Record<string, number>;
  trace_id: string | null;
}

export interface PipelineOverrides {
  system_prompt?: string | null;
  user_template?: string | null;
  route?: string | null;
  rerank_enabled?: boolean | null;
  rerank_top_n?: number | null;
  fused_top_k?: number | null;
  max_context_chunks?: number | null;
  guardrails_enabled?: boolean | null;
}

// ---- api/schemas.py ---------------------------------------------------------------------
export interface ChatRequest {
  query: string;
  user_id: string;
  overrides?: PipelineOverrides | null;
}

export interface RetrieveResponse {
  query: string;
  user_id: string;
  groups: string[];
  contexts: RetrievedChunk[];
  candidates: RetrievedChunk[];
  timings_ms: Record<string, number>;
}

export interface ProviderStatus {
  vertex: boolean;
  bedrock: boolean;
  cohere: boolean;
  guardrail: "bedrock" | "local";
  qdrant: "server" | "embedded";
  task_queue: "inline" | "celery";
  tracing: boolean;
}

export interface IndexStatus {
  name: string;
  chunks: number;
  documents: number;
}

export interface HealthResponse {
  status: "ok";
  version: string;
  offline: boolean;
  providers: ProviderStatus;
  index: IndexStatus;
}

export interface RouteInfo {
  primary: string;
  fallbacks: string[];
  models: string[];
}

export interface MetricGate {
  threshold: number;
  blocking: boolean;
}

/** `gates` is EvalGatesConfig.model_dump() from src/ragops/config.py. */
export interface GatesConfig {
  metrics: Record<string, MetricGate>;
  regression_max_drop?: number;
  acl_max_leaked_chunks?: number;
  judge_route?: string;
  judge_concurrency?: number;
  dataset_files?: string[];
  max_cases?: number | null;
  seed?: number;
}

export interface ConfigResponse {
  rag: Record<string, unknown>;
  routes: Record<string, RouteInfo>;
  gates: GatesConfig;
  prompt: { name: string; sha: string; [k: string]: unknown };
}

export interface PromptResponse {
  name: string;
  description: string;
  system: string;
  user_template: string;
  sha: string;
}

export interface UserResponse {
  user_id: string;
  name: string;
  title: string;
  groups: string[];
}

export type DatasetChoice = "all" | "curated" | "synthetic";

export interface StartEvalRequest {
  label?: string;
  max_cases?: number | null;
  dataset?: DatasetChoice;
  overrides?: PipelineOverrides | null;
}

export interface ScoreAnswerRequest {
  question: string;
  answer: string;
  contexts: string[];
  expected_answer?: string | null;
}

export interface LiveMetric {
  score: number | null;
  reason: string | null;
  threshold: number;
  passed: boolean;
  error?: string | null;
}

export interface ScoreAnswerResponse {
  metrics: Record<string, LiveMetric>;
  judge_model: string;
}

export interface IngestCorpusResponse {
  queued: string[];
}

// ---- ingestion/registry.py --------------------------------------------------------------
export type DocumentStatus = "queued" | "parsing" | "chunking" | "embedding" | "indexed" | "failed";

export interface DocumentRecord {
  index_name: string;
  doc_id: string;
  title: string;
  filename: string;
  source_path: string;
  acl_groups: string[];
  status: DocumentStatus;
  error: string | null;
  parser: string | null;
  pages: number | null;
  num_chunks: number;
  size_bytes: number;
  sha256: string | null;
  created_at: string;
  updated_at: string;
}

// ---- evals/models.py --------------------------------------------------------------------
export type QuestionType = "factual" | "procedural" | "multi_hop" | "comparative";

export interface GoldenCase {
  id: string;
  question: string;
  expected_answer: string;
  source_doc_ids: string[];
  question_type: QuestionType;
  user_id: string;
  origin: "curated" | "synthetic";
  source_section: string | null;
}

export interface MetricScore {
  score: number | null;
  reason: string | null;
  error: string | null;
}

export interface CaseContext {
  chunk_id: string;
  doc_id: string;
  section: string;
  text: string;
  rerank_score: number | null;
}

export interface EvalCaseResult {
  case_id: string;
  question: string;
  expected_answer: string;
  question_type: string;
  origin: string;
  answer: string;
  source_doc_ids: string[];
  retrieved_doc_ids: string[];
  contexts: CaseContext[];
  metrics: Record<string, MetricScore>;
  doc_recall: boolean;
  latency_ms: number;
  error: string | null;
}

export interface MetricSummary {
  mean: number | null;
  threshold: number;
  blocking: boolean;
  passed: boolean;
  n: number;
  baseline: number | null;
  delta: number | null;
  regressed: boolean;
}

export interface AclSummary {
  cases: number;
  leaked_chunks: number;
  passed: boolean;
  details: Record<string, unknown>[];
}

export interface RunConfig {
  index_name: string;
  embedding_model: string;
  chunking: Record<string, unknown>;
  prompt_name: string;
  prompt_sha: string;
  synthesis_route: string;
  synthesis_model: string;
  judge_model: string;
  rerank_model: string | null;
  git_sha: string | null;
  git_branch: string | null;
  offline: boolean;
  overrides: PipelineOverrides | null;
}

export type RunStatus = "running" | "completed" | "failed";

export interface EvalRunSummary {
  run_id: string;
  label: string;
  status: RunStatus;
  created_at: string;
  completed_at: string | null;
  duration_s: number | null;
  num_cases: number;
  metrics: Record<string, MetricSummary>;
  gate_passed: boolean | null;
  gate_failures: string[];
  acl: AclSummary;
  config: RunConfig;
  /** Cases where the reranker failed and retrieval degraded to RRF order. */
  rerank_fallbacks: number;
  error: string | null;
}

export interface EvalRun extends EvalRunSummary {
  cases: EvalCaseResult[];
}

export interface MetricDelta {
  base: number | null;
  head: number | null;
  delta: number | null;
}

export interface CaseDelta {
  case_id: string;
  question: string;
  base: Record<string, number | null>;
  head: Record<string, number | null>;
  delta: Record<string, number | null>;
  base_answer: string;
  head_answer: string;
}

export interface EvalComparison {
  base: EvalRunSummary;
  head: EvalRunSummary;
  metric_deltas: Record<string, MetricDelta>;
  cases: CaseDelta[];
  regressions: string[];
  improvements: string[];
}

// ---- SSE payloads of /api/chat/stream -----------------------------------------------------
export interface StreamDone {
  answer: string;
  citations: Citation[];
  guardrails: GuardrailVerdict[];
  model: string | null;
  route: string | null;
  usage: Usage;
  timings_ms: Record<string, number>;
  trace_id: string | null;
}

export interface StreamBlocked {
  message: string;
  guardrail: GuardrailVerdict;
}
