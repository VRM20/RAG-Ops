// Typed client for the RAG-Ops FastAPI backend (contract: docs/API.md).

import { readSSE } from "./sse";
import type {
  ChatRequest,
  Chunk,
  ConfigResponse,
  DocumentRecord,
  EvalComparison,
  EvalRun,
  EvalRunSummary,
  GoldenCase,
  GuardrailVerdict,
  HealthResponse,
  IngestCorpusResponse,
  PromptResponse,
  RAGResponse,
  RetrievedChunk,
  RetrieveResponse,
  ScoreAnswerRequest,
  ScoreAnswerResponse,
  StartEvalRequest,
  StreamBlocked,
  StreamDone,
  UserResponse,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
const ADMIN_KEY = process.env.NEXT_PUBLIC_ADMIN_KEY || "";

export class ApiError extends Error {
  status: number;
  /** true when the request never reached the server (API down, CORS, DNS). */
  network: boolean;

  constructor(message: string, status = 0, network = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.network = network;
  }
}

function formatDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc: [...], msg, type}]
    return detail
      .map((d) => {
        if (d && typeof d === "object" && "msg" in d) {
          const loc = Array.isArray((d as { loc?: unknown[] }).loc) ? (d as { loc: unknown[] }).loc.join(".") : "";
          return loc ? `${loc}: ${(d as { msg: string }).msg}` : (d as { msg: string }).msg;
        }
        return JSON.stringify(d);
      })
      .join("; ");
  }
  return detail ? JSON.stringify(detail) : "";
}

async function errorFromResponse(res: Response): Promise<ApiError> {
  let message = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    const detail = formatDetail(body?.detail);
    if (detail) message = detail;
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(message, res.status);
}

function headers(json: boolean, admin: boolean): HeadersInit {
  const h: Record<string, string> = { Accept: "application/json" };
  if (json) h["Content-Type"] = "application/json";
  if (admin && ADMIN_KEY) h["X-Admin-Key"] = ADMIN_KEY;
  return h;
}

async function send(path: string, init: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(`Cannot reach the API at ${API_URL}. Start the API: make api`, 0, true);
  }
  if (!res.ok) throw await errorFromResponse(res);
  return res;
}

async function request<T>(
  path: string,
  { method = "GET", body, admin = false }: { method?: string; body?: unknown; admin?: boolean } = {},
): Promise<T> {
  const res = await send(path, {
    method,
    headers: headers(body !== undefined, admin),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

const enc = encodeURIComponent;

export const api = {
  // ---- system ----
  health: () => request<HealthResponse>("/api/health"),
  config: () => request<ConfigResponse>("/api/config"),
  users: () => request<UserResponse[]>("/api/users"),
  aclGroups: () => request<string[]>("/api/acl-groups"),
  prompts: () => request<string[]>("/api/prompts"),
  prompt: (name: string) => request<PromptResponse>(`/api/prompts/${enc(name)}`),

  // ---- documents ----
  documents: () => request<DocumentRecord[]>("/api/documents"),
  document: (docId: string) => request<DocumentRecord>(`/api/documents/${enc(docId)}`),
  documentChunks: (docId: string) => request<Chunk[]>(`/api/documents/${enc(docId)}/chunks`),
  deleteDocument: (docId: string) => request<void>(`/api/documents/${enc(docId)}`, { method: "DELETE", admin: true }),
  ingestCorpus: () => request<IngestCorpusResponse>("/api/documents/ingest-corpus", { method: "POST", admin: true }),
  uploadDocument: async (input: { file: File; aclGroups: string[]; title?: string; docId?: string }) => {
    const form = new FormData();
    form.append("file", input.file);
    form.append("acl_groups", input.aclGroups.join(","));
    if (input.title) form.append("title", input.title);
    if (input.docId) form.append("doc_id", input.docId);
    // No Content-Type header: the browser sets the multipart boundary itself.
    const res = await send("/api/documents", { method: "POST", headers: headers(false, true), body: form });
    return (await res.json()) as DocumentRecord;
  },

  // ---- chat / retrieval ----
  chat: (req: ChatRequest) => request<RAGResponse>("/api/chat", { method: "POST", body: req }),
  retrieve: (req: ChatRequest) => request<RetrieveResponse>("/api/retrieve", { method: "POST", body: req }),

  // ---- evals ----
  evalRuns: () => request<EvalRunSummary[]>("/api/evals/runs"),
  evalRun: (runId: string) => request<EvalRun>(`/api/evals/runs/${enc(runId)}`),
  startEval: (req: StartEvalRequest) =>
    request<EvalRunSummary>("/api/evals/runs", { method: "POST", body: req, admin: true }),
  /** Returns null when no baseline has been committed yet (404). */
  baseline: async (): Promise<EvalRun | null> => {
    try {
      return await request<EvalRun>("/api/evals/baseline");
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) return null;
      throw err;
    }
  },
  compare: (base: string, head: string) =>
    request<EvalComparison>(`/api/evals/compare?base=${enc(base)}&head=${enc(head)}`),
  dataset: () => request<GoldenCase[]>("/api/evals/dataset"),
  scoreAnswer: (req: ScoreAnswerRequest) =>
    request<ScoreAnswerResponse>("/api/evals/score", { method: "POST", body: req }),
};

// ---- streaming chat -------------------------------------------------------------------
export interface StreamHandlers {
  onGuardrail?: (v: GuardrailVerdict) => void;
  onBlocked?: (b: StreamBlocked) => void;
  onSources?: (contexts: RetrievedChunk[]) => void;
  onToken?: (text: string) => void;
  onDone?: (d: StreamDone) => void;
  onError?: (message: string) => void;
}

/** POST /api/chat/stream and dispatch SSE events. Resolves when the stream ends. */
export async function streamChat(req: ChatRequest, handlers: StreamHandlers, signal?: AbortSignal): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/chat/stream`, {
      method: "POST",
      headers: { ...headers(true, false), Accept: "text/event-stream" },
      body: JSON.stringify(req),
      signal,
      cache: "no-store",
    });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    throw new ApiError(`Cannot reach the API at ${API_URL}. Start the API: make api`, 0, true);
  }
  if (!res.ok) throw await errorFromResponse(res);

  let terminal = false;
  try {
    for await (const msg of readSSE(res, signal)) {
      let data: unknown = null;
      try {
        data = msg.data ? JSON.parse(msg.data) : null;
      } catch {
        data = msg.data;
      }
      switch (msg.event) {
        case "guardrail":
          handlers.onGuardrail?.(data as GuardrailVerdict);
          break;
        case "blocked":
          terminal = true;
          handlers.onBlocked?.(data as StreamBlocked);
          break;
        case "sources":
          handlers.onSources?.((data as { contexts: RetrievedChunk[] })?.contexts ?? []);
          break;
        case "token":
          handlers.onToken?.((data as { text: string })?.text ?? "");
          break;
        case "done":
          terminal = true;
          handlers.onDone?.(data as StreamDone);
          break;
        case "error":
          terminal = true;
          handlers.onError?.((data as { message?: string })?.message ?? String(data));
          break;
        default:
          break;
      }
    }
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    throw err;
  }
  if (!terminal && !signal?.aborted) handlers.onError?.("The stream ended unexpectedly.");
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}
