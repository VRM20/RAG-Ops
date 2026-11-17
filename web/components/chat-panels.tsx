"use client";

import { useState } from "react";
import { Gauge, Loader2, Sparkles } from "lucide-react";

import { EmptyState, ErrorState, GuardrailBadge, ScoreBadge } from "@/components/shared";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/misc";
import { fmtMs, fmtScore, pageLabel } from "@/lib/format";
import { metricLabel, METRIC_DESCRIPTIONS, orderedMetricKeys } from "@/lib/metrics";
import type { Citation, GuardrailVerdict, RetrievedChunk, ScoreAnswerResponse, Usage } from "@/lib/types";
import { cn } from "@/lib/utils";

export function sourceElementId(turnId: string, n: number) {
  return `src-${turnId}-${n}`;
}

// ---- Sources ---------------------------------------------------------------------------------
function SourceCard({
  turnId,
  n,
  ctx,
  cited,
  active,
  onSelect,
}: {
  turnId: string;
  n: number;
  ctx: RetrievedChunk;
  cited: boolean;
  active: boolean;
  onSelect: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const text = ctx.chunk.text;
  const long = text.length > 320;
  return (
    <div
      id={sourceElementId(turnId, n)}
      className={cn(
        "scroll-mt-4 rounded-lg border p-3 transition-colors",
        active ? "border-primary bg-primary/5 ring-1 ring-primary/40" : "hover:bg-muted/40",
      )}
    >
      <button type="button" onClick={onSelect} className="flex w-full items-start gap-2 text-left">
        <span
          className={cn(
            "mt-0.5 flex h-5 min-w-5 items-center justify-center rounded px-1 font-mono text-[11px] font-semibold",
            active ? "bg-primary text-primary-foreground" : "bg-primary/15 text-primary",
          )}
        >
          {n}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{ctx.chunk.doc_title}</span>
          <span className="block truncate text-xs text-muted-foreground">
            {[ctx.chunk.section_path.join(" › ") || "Overview", pageLabel(ctx.chunk.page_start, ctx.chunk.page_end)]
              .filter(Boolean)
              .join(" · ")}
          </span>
        </span>
        {cited && <Badge variant="success" className="text-[10px]">cited</Badge>}
      </button>

      <div className="mt-2 flex flex-wrap gap-1.5 text-[11px]">
        <Badge variant="secondary" className="font-mono" title="Cross-encoder rerank relevance">
          rerank {fmtScore(ctx.rerank_score, 3)}
        </Badge>
        <Badge variant="outline" className="font-mono" title="Rank in dense (vector) results">
          dense #{ctx.dense_rank ?? "-"}
        </Badge>
        <Badge variant="outline" className="font-mono" title="Rank in sparse (BM25) results">
          bm25 #{ctx.sparse_rank ?? "-"}
        </Badge>
        <Badge variant="outline" className="font-mono" title="Reciprocal Rank Fusion score">
          rrf {ctx.rrf_score.toFixed(4)}
        </Badge>
      </div>

      <p className="mt-2 whitespace-pre-line text-xs leading-relaxed text-muted-foreground">
        {expanded || !long ? text : `${text.slice(0, 320)}…`}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-1">
        {ctx.chunk.acl_groups.map((g) => (
          <Badge key={g} variant="muted" className="font-mono text-[10px]">
            {g}
          </Badge>
        ))}
        {long && (
          <button type="button" onClick={() => setExpanded((v) => !v)} className="ml-auto text-[11px] text-primary hover:underline">
            {expanded ? "Show less" : "Show more"}
          </button>
        )}
      </div>
    </div>
  );
}

export function SourcesPanel({
  turnId,
  contexts,
  citations,
  activeSource,
  onSelect,
  pending,
}: {
  turnId: string;
  contexts: RetrievedChunk[];
  citations: Citation[];
  activeSource: number | null;
  onSelect: (n: number) => void;
  pending: boolean;
}) {
  if (!contexts.length)
    return pending ? (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" /> Retrieving (dense + BM25 → RRF → rerank)…
      </div>
    ) : (
      <EmptyState title="No sources" description="Nothing in the documents this persona can access matched the question." />
    );
  const cited = new Set(citations.map((c) => c.source_number));
  return (
    <div className="space-y-2">
      {contexts.map((ctx, i) => (
        <SourceCard
          key={ctx.chunk.chunk_id}
          turnId={turnId}
          n={i + 1}
          ctx={ctx}
          cited={cited.has(i + 1)}
          active={activeSource === i + 1}
          onSelect={() => onSelect(i + 1)}
        />
      ))}
    </div>
  );
}

// ---- Scores ----------------------------------------------------------------------------------
export function ScoresPanel({
  scores,
  scoring,
  error,
  canScore,
  onScore,
}: {
  scores?: ScoreAnswerResponse;
  scoring: boolean;
  error?: unknown;
  canScore: boolean;
  onScore: () => void;
}) {
  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-2 rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground sm:flex-row sm:items-center">
        <Gauge className="h-4 w-4 shrink-0" />
        <span className="flex-1">
          Scores this answer live with the DeepEval judge (faithfulness and answer relevancy against the retrieved context). Each run costs judge tokens.
        </span>
        <Button size="sm" onClick={onScore} disabled={!canScore || scoring}>
          {scoring ? <Loader2 className="animate-spin" /> : <Sparkles />}
          {scores ? "Re-evaluate" : "Evaluate answer"}
        </Button>
      </div>
      {error ? <ErrorState error={error} compact /> : null}
      {scores ? (
        <div className="space-y-2">
          {orderedMetricKeys(Object.keys(scores.metrics)).map((k) => {
            const m = scores.metrics[k];
            return (
              <div key={k} className="rounded-lg border p-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="text-sm font-medium">{metricLabel(k)}</div>
                    <div className="text-[11px] text-muted-foreground">{METRIC_DESCRIPTIONS[k]}</div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <span className="text-[11px] text-muted-foreground">≥ {m.threshold.toFixed(2)}</span>
                    <ScoreBadge score={m.score} threshold={m.threshold} className="text-sm" />
                    <Badge variant={m.passed ? "success" : "destructive"}>{m.passed ? "pass" : "fail"}</Badge>
                  </div>
                </div>
                {m.reason && <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{m.reason}</p>}
                {m.error && <p className="mt-2 text-xs text-destructive">{m.error}</p>}
              </div>
            );
          })}
          <div className="text-[11px] text-muted-foreground">
            Judge: <span className="font-mono">{scores.judge_model}</span>
          </div>
        </div>
      ) : (
        !scoring && <EmptyState title="Not scored yet" description="Run the judge on the selected answer to see DeepEval score badges." />
      )}
    </div>
  );
}

// ---- Guardrails ------------------------------------------------------------------------------
export function GuardrailsPanel({ verdicts, pending }: { verdicts: GuardrailVerdict[]; pending: boolean }) {
  if (!verdicts.length)
    return pending ? (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" /> Checking input…
      </div>
    ) : (
      <EmptyState title="No guardrail verdicts" />
    );
  return (
    <div className="space-y-2">
      {verdicts.map((v, i) => (
        <div key={`${v.stage}-${i}`} className="rounded-lg border p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-sm font-medium capitalize">{v.stage} guardrail</div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="font-mono text-[10px]">
                {v.provider}
              </Badge>
              <GuardrailBadge action={v.action} />
            </div>
          </div>
          {v.reasons.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {v.reasons.map((r) => (
                <Badge key={r} variant="warning" className="font-mono text-[10px]">
                  {r}
                </Badge>
              ))}
            </div>
          )}
          {(v.grounding_score !== null || v.relevance_score !== null) && (
            <div className="mt-3 space-y-2">
              {v.grounding_score !== null && (
                <div>
                  <div className="mb-1 flex justify-between text-xs">
                    <span>Contextual grounding</span>
                    <span className="font-mono">{fmtScore(v.grounding_score)}</span>
                  </div>
                  <Progress value={v.grounding_score} label="Grounding score" indicatorClassName={v.grounding_score < 0.75 ? "bg-warning" : "bg-success"} />
                </div>
              )}
              {v.relevance_score !== null && (
                <div>
                  <div className="mb-1 flex justify-between text-xs">
                    <span>Relevance</span>
                    <span className="font-mono">{fmtScore(v.relevance_score)}</span>
                  </div>
                  <Progress value={v.relevance_score} label="Relevance score" indicatorClassName={v.relevance_score < 0.75 ? "bg-warning" : "bg-success"} />
                </div>
              )}
            </div>
          )}
          {v.sanitized_text && (
            <div className="mt-2 rounded bg-muted p-2 font-mono text-[11px] text-muted-foreground">{v.sanitized_text}</div>
          )}
          <div className="mt-2 text-[11px] text-muted-foreground">{fmtMs(v.latency_ms)}</div>
        </div>
      ))}
    </div>
  );
}

// ---- Trace -----------------------------------------------------------------------------------
export function TracePanel({
  model,
  route,
  traceId,
  usage,
  timings,
}: {
  model: string | null;
  route: string | null;
  traceId: string | null;
  usage: Usage | null;
  timings: Record<string, number>;
}) {
  const entries = Object.entries(timings).filter(([k]) => k !== "total");
  const total = timings.total ?? entries.reduce((s, [, v]) => s + v, 0);
  const max = Math.max(total, ...entries.map(([, v]) => v), 1);
  return (
    <div className="space-y-4">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        <dt className="text-muted-foreground">Model</dt>
        <dd className="truncate font-mono text-xs" title={model ?? ""}>{model ?? "-"}</dd>
        <dt className="text-muted-foreground">Route</dt>
        <dd>{route ? <Badge variant="secondary">{route}</Badge> : "-"}</dd>
        <dt className="text-muted-foreground">Trace ID</dt>
        <dd className="break-all font-mono text-xs">{traceId ?? "tracing disabled"}</dd>
      </dl>

      {usage && (
        <div className="grid grid-cols-3 gap-2">
          {[
            ["Prompt", usage.prompt_tokens],
            ["Completion", usage.completion_tokens],
            ["Total", usage.total_tokens],
          ].map(([label, v]) => (
            <div key={label} className="rounded-lg border p-2 text-center">
              <div className="text-[11px] text-muted-foreground">{label} tokens</div>
              <div className="font-mono text-sm font-semibold tabular-nums">{Number(v).toLocaleString()}</div>
            </div>
          ))}
          {usage.cost_usd !== null && usage.cost_usd !== undefined && (
            <div className="col-span-3 text-right text-[11px] text-muted-foreground">≈ ${usage.cost_usd.toFixed(5)}</div>
          )}
        </div>
      )}

      <div>
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="font-medium">Latency breakdown</span>
          <span className="font-mono text-muted-foreground">{fmtMs(total)}</span>
        </div>
        {entries.length === 0 ? (
          <p className="text-xs text-muted-foreground">No timings yet.</p>
        ) : (
          <div className="space-y-2">
            {entries.map(([k, v]) => (
              <div key={k}>
                <div className="mb-1 flex justify-between text-[11px]">
                  <span className="text-muted-foreground">{k.replace(/_/g, " ")}</span>
                  <span className="font-mono">{fmtMs(v)}</span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full" style={{ width: `${(v / max) * 100}%`, background: "hsl(var(--chart-1))" }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
