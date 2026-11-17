# RAG-Ops Console (Next.js 14)

Admin and analytics UI for the RAG-Ops knowledge engine: live RAG chat with citations,
knowledge-base management with ACLs, DeepEval analytics, run comparisons and a prompt sandbox.

Stack: Next.js 14 (App Router) · TypeScript · Tailwind CSS · shadcn/ui-style components ·
Radix Tabs · Recharts · react-markdown.

## Run locally

```bash
cd web
cp .env.local.example .env.local   # point NEXT_PUBLIC_API_URL at the FastAPI backend
npm install
npm run dev                        # http://localhost:3000
```

The backend must be running (`make api`, http://localhost:8000) and must allow the UI origin
via `CORS_ORIGINS` in the root `.env`. If `ADMIN_API_KEY` is set in the backend, set the same
value as `NEXT_PUBLIC_ADMIN_KEY` (upload, delete, ingest and eval runs send it as `X-Admin-Key`).

Production build: `npm run build && npm start`, or `docker build -t ragops-web --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 .`
(`NEXT_PUBLIC_*` values are inlined at build time).

## Pages

| Route | What it shows |
| --- | --- |
| `/` | Index and provider health, latest gate verdict, metric trend across runs, model routing table |
| `/chat` | Streaming RAG chat (SSE). Left: answers with clickable `[n]` citation chips. Right: sources with dense/BM25/RRF/rerank scores, live DeepEval scores, guardrail verdicts, latency trace |
| `/documents` | Upload PDF/MD/TXT with ACL groups, ingest the demo corpus, live ingestion status, chunk viewer |
| `/evals` | Eval runs with gate status and per-metric means, start a run, CI baseline |
| `/evals/[runId]` | Metric cards vs thresholds and baseline, ACL leakage, score histograms, per-question-type chart, sortable/filterable per-case table |
| `/evals/compare` | Base vs head deltas, grouped bar chart, regressions/improvements, per-case answer diff |
| `/sandbox` | Edit the system prompt and generation knobs, quick single-question test, run a golden-set slice and compare with the baseline |

The persona selector in the sidebar sets the `user_id` for chat requests, so you can see ACL
trimming change the answer (e.g. ask for the M3 pay band as Priya (credit-risk), then as Meera (hr)).

## Code map

- `lib/types.ts`: TypeScript mirrors of the backend pydantic schemas (snake_case)
- `lib/api.ts`: typed API client with FastAPI error parsing; `streamChat()` for SSE
- `lib/sse.ts`: incremental SSE parser for `fetch()` streams (POST bodies, partial chunks)
- `components/ui/*`: shadcn-style primitives; `components/charts.tsx`: themed Recharts wrappers
