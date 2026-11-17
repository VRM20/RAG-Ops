# RAG-Ops: Enterprise Knowledge Engine with CI/CD Evaluation Gates

**Problem:** enterprise teams cannot safely change a RAG system's prompt, chunk size,
embedding model or LLM, because they have no way to know whether answer quality or
permission safety just regressed.

**This project:** a production-grade, domain-agnostic hybrid-search RAG platform for any
PDF / Markdown / text collection (policies, manuals, research papers, contracts, reports),
on **GCP Vertex AI (Gemini)** and **AWS Bedrock (Claude)**, where every behavioural change goes through an
automated **DeepEval quality gate** in GitHub Actions. PRs are blocked when
faithfulness or context recall drop below 0.85, regress against the baseline, or when
any user can retrieve a document they are not allowed to see.

```
PR changes config/rag.yaml (chunk size 350 -> 800)
  └─ GitHub Actions: RAG Eval Gate
       ├─ Qdrant service container, ingest corpus
       ├─ 38 curated + 50+ synthetic golden questions -> RAG pipeline
       ├─ DeepEval judge (Gemini) scores every answer
       └─ ❌ contextual_recall 0.79 < 0.85  -> merge blocked, report posted on the PR
```

## Highlights

| Area | What is implemented |
| --- | --- |
| **Dual-cloud model routing** | Task routes (`synthesis`, `pdf_parsing`, `judge`, ...) map to primary + fallback models across Vertex AI and Bedrock through the LiteLLM Router (retries, cooldowns, cross-provider failover). Unconfigured providers are skipped automatically. |
| **Any PDF, any domain** | Gemini multimodal parsing in concurrent page batches (long documents, multi-column layouts, tables, figures, scanned pages via OCR) with page-level citations. A layout-aware local fallback detects headings from font size and weight, strips running headers/footers and flags scanned files. Content-hash parse cache. |
| **Contextual semantic chunking** | Heading-aware sections, embedding-similarity breakpoints for long sections, intact tables, page ranges per chunk, and contextual headers prepended before embedding. |
| **Hybrid retrieval** | Qdrant dense vectors + BM25 sparse vectors (IDF in Qdrant), **Reciprocal Rank Fusion**, **Cohere Rerank 3.5** cross-encoder. |
| **Document ACLs** | `acl_groups` payload filter applied inside both searches (security trimming), plus a zero-tolerance leakage test suite including a prompt-injection case. |
| **Guardrails** | **AWS Bedrock Guardrails** `ApplyGuardrail` on input (prompt attacks, PII, Aadhaar/PAN regexes, denied topics) and output (PII + contextual-grounding hallucination check). |
| **Evaluation** | DeepEval faithfulness, answer relevancy, contextual precision/recall + deterministic doc recall; Gemini-generated synthetic golden set with an LLM critic filter; RAGAS cross-check. |
| **CI/CD gate** | GitHub Actions eval gate with change detection, sticky PR report, baseline regression check, reviewed baseline refresh workflow. |
| **Observability** | OpenTelemetry + OpenInference spans (chain, guardrail, retriever, reranker, LLM) exported to Arize Phoenix or Langfuse. |
| **Platform** | FastAPI with SSE streaming, Celery + Redis workers, Docker Compose stack, Next.js 14 admin and analytics UI. |

Deep dive: **[ARCHITECTURE.md](ARCHITECTURE.md)** · API contract: **[docs/API.md](docs/API.md)** ·
Credentials: **[docs/SETUP_CREDENTIALS.md](docs/SETUP_CREDENTIALS.md)** · Code walkthrough: **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)**

## Quickstart

Requirements: Python 3.11+ (developed on 3.13), Node 22, optional Docker.

```bash
python3 -m venv ragenv && source ragenv/bin/activate
make install                         # pip install -e ".[eval,dev]"
cp .env.example .env                 # fill in: docs/SETUP_CREDENTIALS.md
```

### Option A: zero credentials (explore the system)

```bash
make offline                         # API on :8000 with deterministic local stand-ins, corpus auto-ingested
cd web && npm install && npm run dev # UI on :3000
```

### Option B: real models

```bash
ragops doctor                        # checks every credential with a tiny real call
ragops create-guardrail              # once: provisions the Bedrock Guardrail, prints its id
ragops ingest                        # parse (Gemini), chunk, embed and index the demo corpus
ragops ask "Who sanctions a INR 2 crore loan?" --user priya.sharma
make api                             # FastAPI on http://localhost:8000/docs
make web                             # Next.js on http://localhost:3000
```

### Option C: full stack in Docker

```bash
docker compose up --build            # qdrant, redis, phoenix (:6006), api, celery worker, web
```

## The evaluation loop

```bash
ragops synth                         # generate the synthetic golden set with Gemini (+ critic filter)
ragops eval --max-cases 10           # quick local run: scores, gates and baseline diff in the terminal
ragops eval --save-baseline          # set the bar (normally done through the baseline workflow PR)
python scripts/ragas_crosscheck.py   # re-score with RAGAS (own venv: requirements-ragas.txt)
pytest evals/test_rag_quality.py -m eval   # exactly what CI runs
```

Gates live in [`config/eval_gates.yaml`](config/eval_gates.yaml). The prompt sandbox in the UI runs
the same evaluation with a modified system prompt, top-k, reranker or model route and diffs it
against the baseline before you open a PR.

## Bring your own documents

Nothing in the engine is tied to a domain. Ingest any PDF, Markdown or text file with ACL groups:

```bash
ragops ingest --file report.pdf --acl engineering,leadership
ragops ingest --dir ./contracts --acl legal          # every supported file, recursively
```

or upload it on the **Documents** page. Then grow the golden set for your collection with
`ragops synth` (point `data/corpus/manifest.yaml` at your sources) or add hand-written cases to
`evals/datasets/`.

## Demo corpus and personas

A fictional, deliberately mixed collection of 11 documents (6 rendered as PDFs) with ACL groups.
It covers enterprise policies of a company, an energy-storage **product manual** (unnumbered
headings, spec and fault-code tables), a **research report** (abstract, numbered sections, results
table, references) and **public-library rules**. Try the same question as different users in the chat UI:

| Persona | Groups | "What is the pay band for Grade M3?" |
| --- | --- | --- |
| `meera.iyer` (HR) | hr | INR 16 to 24 lakh, with citation |
| `priya.sharma` (Credit) | credit-risk | refuses: the compensation framework is never retrieved |
| `sam.contractor` | contractors | sees only public documents (company overview, product manual, library rules) |

## Repository layout

```
config/            rag.yaml · models.yaml · eval_gates.yaml · users.yaml · prompts/*.yaml   (versioned behaviour)
src/ragops/
  providers/       router (LiteLLM, Vertex+Bedrock) · gcp · aws · embeddings · guardrails · offline stand-ins
  ingestion/       parsers (Gemini/pypdf) · chunking · pipeline · registry · pdf_builder
  retrieval/       store (Qdrant) · sparse (BM25) · fusion (RRF) · rerank · retriever
  generation/      prompts · pipeline (guard -> retrieve -> generate -> guard, SSE streaming)
  evals/           judge · metrics · runner · acl · synthesizer · report · store
  api/             FastAPI app, routes, schemas
  worker/          Celery app + tasks
  observability/   OpenTelemetry / OpenInference tracing
evals/             golden datasets · ACL cases · baseline · test_rag_quality.py (CI gate)
tests/             offline unit + integration tests (no credentials)
scripts/           ragas_crosscheck.py (RAGAS judge-agreement check, isolated venv)
web/               Next.js 14 + Tailwind + shadcn/ui + Recharts
.github/workflows/ ci.yml · rag-eval-gate.yml · rag-eval-baseline.yml
```

## License

Apache-2.0. The demo corpus is entirely fictional.
