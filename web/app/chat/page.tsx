"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Bot, Loader2, Send, ShieldAlert, Square, Trash2, User } from "lucide-react";

import { GuardrailsPanel, ScoresPanel, SourcesPanel, sourceElementId, TracePanel } from "@/components/chat-panels";
import { MarkdownAnswer } from "@/components/markdown-answer";
import { useApp } from "@/components/providers";
import { PageHeader } from "@/components/shared";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { api, errorMessage, streamChat } from "@/lib/api";
import { fmtMs } from "@/lib/format";
import type { Citation, GuardrailVerdict, RetrievedChunk, ScoreAnswerResponse, Usage } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ChatTurn {
  id: string;
  question: string;
  userId: string;
  userName: string;
  status: "streaming" | "done" | "blocked" | "error";
  answer: string;
  contexts: RetrievedChunk[];
  citations: Citation[];
  guardrails: GuardrailVerdict[];
  model: string | null;
  route: string | null;
  usage: Usage | null;
  timings: Record<string, number>;
  traceId: string | null;
  error?: string;
  blockedMessage?: string;
  scores?: ScoreAnswerResponse;
  scoring?: boolean;
  scoreError?: unknown;
}

// Spans the demo corpus domains: product manual, research report, public-sector rules,
// enterprise policies (the pay-band question shows ACL filtering across personas).
const EXAMPLES = [
  "The battery shows error E101. What should I do?",
  "Which retrieval configuration had the best faithfulness, and at what latency cost?",
  "What are the library's late fees for adult items?",
  "What are the acknowledgement and status-update targets for a SEV1 incident?",
  "What is the pay band for Grade M3?",
];

type Tab = "sources" | "scores" | "guardrails" | "trace";

let turnCounter = 0;

export default function ChatPage() {
  const { userId, user } = useApp();
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("sources");
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const streaming = turns.some((t) => t.status === "streaming");
  const selected = turns.find((t) => t.id === selectedId) ?? turns[turns.length - 1];

  const patch = useCallback((id: string, update: (t: ChatTurn) => Partial<ChatTurn>) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...update(t) } : t)));
  }, []);

  useEffect(() => {
    const el = threadRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const send = useCallback(
    async (raw: string) => {
      const question = raw.trim();
      if (!question || streaming || !userId) return;
      const id = `t${++turnCounter}`;
      const turn: ChatTurn = {
        id,
        question,
        userId,
        userName: user?.name ?? userId,
        status: "streaming",
        answer: "",
        contexts: [],
        citations: [],
        guardrails: [],
        model: null,
        route: null,
        usage: null,
        timings: {},
        traceId: null,
      };
      setTurns((prev) => [...prev, turn]);
      setSelectedId(id);
      setActiveSource(null);
      setInput("");
      stickToBottom.current = true;

      const controller = new AbortController();
      abortRef.current = controller;
      try {
        await streamChat(
          { query: question, user_id: userId },
          {
            onGuardrail: (v) => patch(id, (t) => ({ guardrails: [...t.guardrails.filter((g) => g.stage !== v.stage), v] })),
            onBlocked: (b) =>
              patch(id, (t) => ({
                status: "blocked",
                blockedMessage: b.message,
                guardrails: b.guardrail ? [...t.guardrails.filter((g) => g.stage !== b.guardrail.stage), b.guardrail] : t.guardrails,
              })),
            onSources: (contexts) => patch(id, () => ({ contexts })),
            onToken: (text) => patch(id, (t) => ({ answer: t.answer + text })),
            onDone: (d) =>
              patch(id, (t) => ({
                status: "done",
                answer: d.answer ?? t.answer,
                citations: d.citations ?? [],
                guardrails: d.guardrails?.length ? d.guardrails : t.guardrails,
                model: d.model,
                route: d.route,
                usage: d.usage,
                timings: d.timings_ms ?? {},
                traceId: d.trace_id,
              })),
            onError: (message) => patch(id, () => ({ status: "error", error: message })),
          },
          controller.signal,
        );
        if (controller.signal.aborted) patch(id, (t) => (t.status === "streaming" ? { status: "error", error: "Stopped." } : {}));
      } catch (err) {
        patch(id, () => ({ status: "error", error: errorMessage(err) }));
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [patch, streaming, user?.name, userId],
  );

  const stop = () => abortRef.current?.abort();

  const showSource = (turnId: string, n: number) => {
    setSelectedId(turnId);
    setTab("sources");
    setActiveSource(n);
    window.requestAnimationFrame(() =>
      document.getElementById(sourceElementId(turnId, n))?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  };

  const scoreSelected = async () => {
    if (!selected) return;
    const id = selected.id;
    patch(id, () => ({ scoring: true, scoreError: undefined }));
    setTab("scores");
    try {
      const scores = await api.scoreAnswer({
        question: selected.question,
        answer: selected.answer,
        contexts: selected.contexts.map((c) => c.chunk.text),
      });
      patch(id, () => ({ scores, scoring: false }));
    } catch (err) {
      patch(id, () => ({ scoreError: err, scoring: false }));
    }
  };

  return (
    <div className="mx-auto max-w-[1500px]">
      <PageHeader
        title="RAG Chat"
        description={
          <>
            Answers are grounded only in chunks the current persona may read (ACL filter applied inside Qdrant). Switch persona in the sidebar to see
            permission trimming.
          </>
        }
        actions={
          turns.length > 0 && (
            <Button variant="outline" size="sm" onClick={() => setTurns([])} disabled={streaming}>
              <Trash2 /> Clear
            </Button>
          )
        }
      />

      <div className="grid gap-4 lg:h-[calc(100vh-10.5rem)] lg:grid-cols-[minmax(0,1fr)_minmax(360px,440px)]">
        {/* LEFT: conversation */}
        <Card className="flex min-h-[60vh] flex-col overflow-hidden lg:min-h-0">
          <div
            ref={threadRef}
            onScroll={(e) => {
              const el = e.currentTarget;
              stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
            }}
            className="scrollbar-thin flex-1 space-y-6 overflow-y-auto p-4 sm:p-5"
            aria-live="polite"
          >
            {turns.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-4 py-10 text-center">
                <Bot className="h-8 w-8 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">Ask the knowledge base</div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    Acting as <span className="font-medium text-foreground">{user?.name ?? "…"}</span>
                    {user && <> ({user.groups.join(", ")})</>}
                  </div>
                </div>
              </div>
            ) : (
              turns.map((t) => {
                const isSelected = selected?.id === t.id;
                return (
                  <div key={t.id} className="space-y-3">
                    <div className="flex justify-end">
                      <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground">
                        {t.question}
                        <div className="mt-1 flex items-center justify-end gap-1 text-[10px] opacity-80">
                          <User className="h-3 w-3" /> {t.userName}
                        </div>
                      </div>
                    </div>
                    <div
                      className={cn(
                        "rounded-2xl rounded-bl-sm border px-4 py-3 transition-colors",
                        isSelected ? "border-primary/40 bg-primary/[0.03]" : "bg-card",
                      )}
                    >
                      {t.status === "blocked" ? (
                        <div className="flex items-start gap-2 text-sm text-destructive">
                          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
                          <span>{t.blockedMessage}</span>
                        </div>
                      ) : t.status === "error" && !t.answer ? (
                        <div className="flex items-start gap-2 text-sm text-destructive">
                          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                          <span>{t.error}</span>
                        </div>
                      ) : t.answer ? (
                        <MarkdownAnswer
                          text={t.answer}
                          onCite={(n) => showSource(t.id, n)}
                          activeSource={isSelected ? activeSource : null}
                          maxSource={t.contexts.length || undefined}
                        />
                      ) : (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          {t.contexts.length ? "Generating grounded answer…" : "Guardrail check and hybrid retrieval…"}
                        </div>
                      )}
                      {t.status === "error" && t.answer && <div className="mt-2 text-xs text-destructive">{t.error}</div>}

                      <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t pt-2 text-[11px] text-muted-foreground">
                        {t.status === "streaming" && t.answer && <Loader2 className="h-3 w-3 animate-spin" />}
                        {t.route && <Badge variant="secondary">{t.route}</Badge>}
                        {t.model && (
                          <span className="max-w-[220px] truncate font-mono" title={t.model}>
                            {t.model}
                          </span>
                        )}
                        {t.timings.total !== undefined && <span>· {fmtMs(t.timings.total)}</span>}
                        {t.contexts.length > 0 && <span>· {t.contexts.length} sources</span>}
                        {t.guardrails.some((g) => g.action === "flag") && <Badge variant="warning">low grounding</Badge>}
                        <Button
                          variant="ghost"
                          size="sm"
                          className="ml-auto h-6 px-2 text-[11px]"
                          onClick={() => {
                            setSelectedId(t.id);
                            setActiveSource(null);
                          }}
                        >
                          Inspect
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          <div className="border-t p-3 sm:p-4">
            {turns.length === 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => void send(q)}
                    disabled={!userId}
                    className="rounded-full border px-3 py-1 text-left text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void send(input);
              }}
              className="flex items-end gap-2"
            >
              <label htmlFor="chat-input" className="sr-only">
                Question
              </label>
              <Textarea
                id="chat-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void send(input);
                  }
                }}
                rows={2}
                placeholder={userId ? "Ask anything about the documents you can access… (Enter to send)" : "Loading personas…"}
                className="min-h-[44px] resize-none"
                disabled={!userId}
              />
              {streaming ? (
                <Button type="button" variant="outline" size="icon" onClick={stop} aria-label="Stop generating">
                  <Square />
                </Button>
              ) : (
                <Button type="submit" size="icon" disabled={!input.trim() || !userId} aria-label="Send question">
                  <Send />
                </Button>
              )}
            </form>
          </div>
        </Card>

        {/* RIGHT: inspector */}
        <Card className="flex min-h-[50vh] flex-col overflow-hidden lg:min-h-0">
          {!selected ? (
            <CardContent className="flex flex-1 items-center justify-center p-6 text-center text-sm text-muted-foreground">
              Sources, DeepEval scores, guardrail verdicts and latency for each answer appear here.
            </CardContent>
          ) : (
            <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)} className="flex min-h-0 flex-1 flex-col">
              <div className="border-b p-3">
                <div className="mb-2 truncate text-xs text-muted-foreground" title={selected.question}>
                  Inspecting: <span className="text-foreground">{selected.question}</span>
                </div>
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="sources">Sources{selected.contexts.length ? ` (${selected.contexts.length})` : ""}</TabsTrigger>
                  <TabsTrigger value="scores">Scores</TabsTrigger>
                  <TabsTrigger value="guardrails">Guards</TabsTrigger>
                  <TabsTrigger value="trace">Trace</TabsTrigger>
                </TabsList>
              </div>
              <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto p-3">
                <TabsContent value="sources" className="mt-0">
                  <SourcesPanel
                    turnId={selected.id}
                    contexts={selected.contexts}
                    citations={selected.citations}
                    activeSource={activeSource}
                    onSelect={(n) => setActiveSource(n)}
                    pending={selected.status === "streaming"}
                  />
                </TabsContent>
                <TabsContent value="scores" className="mt-0">
                  <ScoresPanel
                    scores={selected.scores}
                    scoring={Boolean(selected.scoring)}
                    error={selected.scoreError}
                    canScore={selected.status === "done" && Boolean(selected.answer) && selected.contexts.length > 0}
                    onScore={() => void scoreSelected()}
                  />
                </TabsContent>
                <TabsContent value="guardrails" className="mt-0">
                  <GuardrailsPanel verdicts={selected.guardrails} pending={selected.status === "streaming"} />
                </TabsContent>
                <TabsContent value="trace" className="mt-0">
                  <TracePanel
                    model={selected.model}
                    route={selected.route}
                    traceId={selected.traceId}
                    usage={selected.usage}
                    timings={selected.timings}
                  />
                </TabsContent>
              </div>
            </Tabs>
          )}
        </Card>
      </div>
    </div>
  );
}
