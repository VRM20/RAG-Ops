# HTTP API Contract

Base URL: `http://localhost:8000` (FastAPI). Interactive docs: `/docs` (Swagger) and `/redoc`.
All endpoints are prefixed with `/api`. Schemas live in `src/ragops/api/schemas.py`,
`src/ragops/types.py` and `src/ragops/evals/models.py`.

Identity: the demo resolves `user_id` against `config/users.yaml`. In production the
user and groups come from a verified OIDC token instead.

Write endpoints (upload, delete, ingest, start eval) require the header
`X-Admin-Key: <ADMIN_API_KEY>` **only if** `ADMIN_API_KEY` is set in `.env`.

## System

| Method | Path | Response |
| --- | --- | --- |
| GET | `/api/health` | `HealthResponse`: version, offline flag, provider status, active index (name, chunks, documents) |
| GET | `/api/config` | `ConfigResponse`: rag.yaml, model routes with resolved model ids, eval gates, active prompt name+sha |
| GET | `/api/users` | `UserResponse[]`: demo personas |
| GET | `/api/acl-groups` | `string[]` |
| GET | `/api/prompts` | `string[]`: prompt names |
| GET | `/api/prompts/{name}` | `PromptResponse`: system, user_template, sha |

## Documents (knowledge base admin)

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| GET | `/api/documents` | | `DocumentRecord[]` |
| POST | `/api/documents` | multipart: `file` (.pdf/.md/.txt), `acl_groups` (comma-separated), `title?`, `doc_id?` | `202 DocumentRecord` (status `queued`) |
| POST | `/api/documents/ingest-corpus` | | `202 {queued: string[]}`: (re)ingests `data/corpus/manifest.yaml` |
| GET | `/api/documents/{doc_id}` | | `DocumentRecord` |
| GET | `/api/documents/{doc_id}/chunks` | | `Chunk[]` |
| DELETE | `/api/documents/{doc_id}` | | `204` |

`DocumentRecord.status` goes through: `queued -> parsing -> chunking -> embedding -> indexed` (or `failed`, with `error`). Poll until it is terminal.

## Chat / retrieval

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| POST | `/api/chat` | `ChatRequest {query, user_id, overrides?}` | `RAGResponse` |
| POST | `/api/chat/stream` | `ChatRequest` | `text/event-stream` (see below) |
| POST | `/api/retrieve` | `ChatRequest` | `RetrieveResponse`: reranked contexts + fused candidates with dense/sparse/RRF/rerank scores |

### SSE events of `/api/chat/stream`

Each event is `event: <name>\ndata: <json>\n\n`.

| event | data |
| --- | --- |
| `guardrail` | `GuardrailVerdict` (input stage) |
| `blocked` | `{message, guardrail}`: terminal, no answer follows |
| `sources` | `{contexts: RetrievedChunk[]}`: numbered in order, `[1]` is `contexts[0]` |
| `token` | `{text}`: incremental answer text |
| `done` | `{answer, citations, guardrails, model, route, usage, timings_ms, trace_id}`: terminal |
| `error` | `{message}`: terminal |

## Evaluations

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| GET | `/api/evals/runs` | | `EvalRunSummary[]`, newest first |
| POST | `/api/evals/runs` | `StartEvalRequest {label, max_cases?, dataset, overrides?}` | `202 EvalRunSummary` (status `running`) |
| GET | `/api/evals/runs/{run_id}` | | `EvalRun` (summary + `cases`) |
| GET | `/api/evals/baseline` | | `EvalRun` (committed CI baseline) or `404` |
| GET | `/api/evals/compare?base=<run_id or "baseline">&head=<run_id>` | | `EvalComparison` |
| GET | `/api/evals/dataset` | | `GoldenCase[]` |
| POST | `/api/evals/score` | `ScoreAnswerRequest {question, answer, contexts, expected_answer?}` | `ScoreAnswerResponse`: live DeepEval scores for one answer |

The prompt sandbox is `POST /api/evals/runs` with `overrides.system_prompt` (and other
`PipelineOverrides`), followed by `GET /api/evals/compare?base=baseline&head=<run_id>`.

Metric keys used everywhere: `faithfulness`, `answer_relevancy`, `contextual_precision`,
`contextual_recall`, `doc_recall`.
