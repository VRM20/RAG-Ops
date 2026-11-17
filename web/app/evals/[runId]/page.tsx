"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { Fragment, useMemo, useState } from "react";
import { ArrowDown, ArrowLeft, ArrowUp, ArrowUpDown, Check, ChevronDown, ChevronRight, GitCompare, ShieldCheck, X } from "lucide-react";

import { GroupedBarChart, Histogram } from "@/components/charts";
import { MarkdownAnswer } from "@/components/markdown-answer";
import { EmptyState, ErrorState, GateBadge, LoadingCards, LoadingRows, PageHeader, ScoreBadge, toneText } from "@/components/shared";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/misc";
import { Select } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { fmtDate, fmtDelta, fmtDuration, fmtMs, fmtScore, runName, shortSha } from "@/lib/format";
import { useAsync, useInterval } from "@/lib/hooks";
import { deltaTone, JUDGED_METRICS, METRIC_COLORS, METRIC_DESCRIPTIONS, metricLabel, orderedMetricKeys } from "@/lib/metrics";
import type { EvalCaseResult, EvalRun, MetricSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

type SortKey = "question" | "question_type" | "latency_ms" | "doc_recall" | (typeof JUDGED_METRICS)[number];

function MetricCard({ name, m }: { name: string; m: MetricSummary }) {
  const tone = m.passed ? "success" : m.blocking ? "destructive" : "warning";
  return (
    <Card className={cn(!m.passed && m.blocking && "border-destructive/40")}>
      <CardContent className="space-y-2 p-4">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium text-muted-foreground" title={METRIC_DESCRIPTIONS[name]}>
            {metricLabel(name)}
          </span>
          {m.blocking ? <Badge variant="outline" className="text-[10px]">blocking</Badge> : <Badge variant="muted" className="text-[10px]">advisory</Badge>}
        </div>
        <div className={cn("text-2xl font-semibold tabular-nums", toneText(tone))}>{fmtScore(m.mean, 3)}</div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
          <span>gate ≥ {m.threshold.toFixed(2)}</span>
          <span>n={m.n}</span>
          {m.baseline !== null && (
            <span>
              base {fmtScore(m.baseline, 3)}{" "}
              <span className={cn("font-medium", toneText(deltaTone(m.delta)))}>({fmtDelta(m.delta)})</span>
            </span>
          )}
          {m.regressed && <Badge variant="destructive" className="text-[10px]">regressed</Badge>}
        </div>
      </CardContent>
    </Card>
  );
}

function SortHeader({
  label,
  k,
  sort,
  setSort,
  className,
}: {
  label: string;
  k: SortKey;
  sort: { key: SortKey; dir: 1 | -1 };
  setSort: (s: { key: SortKey; dir: 1 | -1 }) => void;
  className?: string;
}) {
  const active = sort.key === k;
  const Icon = !active ? ArrowUpDown : sort.dir === 1 ? ArrowUp : ArrowDown;
  return (
    <TableHead className={className} aria-sort={active ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
      <button
        type="button"
        onClick={() => setSort({ key: k, dir: active ? (sort.dir === 1 ? -1 : 1) : 1 })}
        className="inline-flex items-center gap-1 hover:text-foreground"
      >
        {label}
        <Icon className="h-3 w-3" />
      </button>
    </TableHead>
  );
}

function sortValue(c: EvalCaseResult, key: SortKey): number | string {
  switch (key) {
    case "question":
      return c.question.toLowerCase();
    case "question_type":
      return c.question_type;
    case "latency_ms":
      return c.latency_ms;
    case "doc_recall":
      return c.doc_recall ? 1 : 0;
    default:
      return c.metrics[key]?.score ?? -1;
  }
}

function CaseDetail({ c, gates }: { c: EvalCaseResult; gates: Record<string, MetricSummary> }) {
  return (
    <div className="grid gap-4 p-3 lg:grid-cols-2">
      <div className="space-y-3">
        <div>
          <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Answer</div>
          <div className="rounded-lg border bg-background p-3">
            {c.answer ? <MarkdownAnswer text={c.answer} maxSource={c.contexts.length || undefined} /> : <span className="text-sm text-muted-foreground">-</span>}
          </div>
        </div>
        <div>
          <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Expected answer</div>
          <div className="rounded-lg border bg-muted/40 p-3 text-sm">{c.expected_answer}</div>
        </div>
        {c.error && <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">{c.error}</div>}
        <div className="space-y-1.5">
          <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Metric reasons</div>
          {orderedMetricKeys(Object.keys(c.metrics)).map((k) => (
            <div key={k} className="rounded-lg border p-2 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{metricLabel(k)}</span>
                <ScoreBadge score={c.metrics[k].score} threshold={gates[k]?.threshold} />
              </div>
              {c.metrics[k].reason && <p className="mt-1 leading-relaxed text-muted-foreground">{c.metrics[k].reason}</p>}
              {c.metrics[k].error && <p className="mt-1 text-destructive">{c.metrics[k].error}</p>}
            </div>
          ))}
        </div>
      </div>
      <div className="space-y-3">
        <div className="text-xs">
          <span className="text-muted-foreground">Expected docs: </span>
          {c.source_doc_ids.map((d) => (
            <Badge key={d} variant={c.retrieved_doc_ids.includes(d) ? "success" : "destructive"} className="mr-1 font-mono text-[10px]">
              {d}
            </Badge>
          ))}
        </div>
        <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Retrieved context ({c.contexts.length})</div>
        <div className="scrollbar-thin max-h-[420px] space-y-2 overflow-y-auto pr-1">
          {c.contexts.map((ctx, i) => (
            <div key={ctx.chunk_id} className="rounded-lg border p-2">
              <div className="flex items-center gap-2 text-[11px]">
                <Badge variant="secondary" className="font-mono">
                  {i + 1}
                </Badge>
                <span className="min-w-0 flex-1 truncate font-mono text-muted-foreground" title={`${ctx.doc_id} · ${ctx.section}`}>
                  {ctx.doc_id} · {ctx.section || "Overview"}
                </span>
                <span className="font-mono">{fmtScore(ctx.rerank_score, 3)}</span>
              </div>
              <p className="mt-1 whitespace-pre-line text-xs leading-relaxed text-muted-foreground">{ctx.text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function RunDetailPage() {
  const params = useParams<{ runId: string }>();
  const runId = decodeURIComponent(params.runId);
  const isBaseline = runId === "baseline";
  const run = useAsync<EvalRun | null>(() => (isBaseline ? api.baseline() : api.evalRun(runId)), [runId]);
  useInterval(() => void run.reload(), 3000, run.data?.status === "running");

  const [histMetric, setHistMetric] = useState<string>("faithfulness");
  const [query, setQuery] = useState("");
  const [qtype, setQtype] = useState("all");
  const [origin, setOrigin] = useState("all");
  const [failingOnly, setFailingOnly] = useState(false);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "faithfulness", dir: 1 });
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const data = run.data;
  const gates = data?.metrics ?? {};
  const cases = useMemo(() => data?.cases ?? [], [data]);
  const qtypes = useMemo(() => Array.from(new Set(cases.map((c) => c.question_type))).sort(), [cases]);
  const origins = useMemo(() => Array.from(new Set(cases.map((c) => c.origin))).sort(), [cases]);

  const isFailing = (c: EvalCaseResult) =>
    Boolean(c.error) ||
    !c.doc_recall ||
    JUDGED_METRICS.some((k) => {
      const s = c.metrics[k]?.score;
      const g = gates[k];
      return s !== null && s !== undefined && g?.blocking && s < g.threshold;
    });

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = cases.filter(
      (c) =>
        (qtype === "all" || c.question_type === qtype) &&
        (origin === "all" || c.origin === origin) &&
        (!q || c.question.toLowerCase().includes(q) || c.case_id.toLowerCase().includes(q)) &&
        (!failingOnly || isFailing(c)),
    );
    return [...rows].sort((a, b) => {
      const va = sortValue(a, sort.key);
      const vb = sortValue(b, sort.key);
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases, query, qtype, origin, failingOnly, sort, gates]);

  const histValues = cases.map((c) => c.metrics[histMetric]?.score).filter((v): v is number => typeof v === "number");

  const byType = qtypes.map((t) => {
    const rows = cases.filter((c) => c.question_type === t);
    const row: Record<string, number | string | null> = { type: t.replace("_", "-"), n: rows.length };
    for (const m of JUDGED_METRICS) {
      const vals = rows.map((c) => c.metrics[m]?.score).filter((v): v is number => typeof v === "number");
      row[m] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    }
    return row;
  });

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  if (run.error)
    return (
      <div className="mx-auto max-w-7xl">
        <BackLink />
        <ErrorState error={run.error} onRetry={() => void run.reload()} />
      </div>
    );
  if (run.loading && run.data === undefined)
    return (
      <div className="mx-auto max-w-7xl space-y-6">
        <BackLink />
        <LoadingCards count={5} className="xl:grid-cols-5" />
        <LoadingRows rows={8} />
      </div>
    );
  if (!data)
    return (
      <div className="mx-auto max-w-7xl">
        <BackLink />
        <EmptyState title="No baseline yet" description="Generate one with `make baseline` on main and commit evals/baselines/baseline.json." />
      </div>
    );

  const cfg = data.config;
  return (
    <div className="mx-auto max-w-7xl">
      <BackLink />
      <PageHeader
        title={isBaseline ? `Baseline · ${runName(data)}` : runName(data)}
        description={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span>{fmtDate(data.created_at)}</span>
            <span>{data.num_cases} cases</span>
            <span>{fmtDuration(data.duration_s)}</span>
            <span className="font-mono">git {shortSha(cfg.git_sha)}</span>
            {cfg.offline && <Badge variant="warning">offline run</Badge>}
            {(data.rerank_fallbacks ?? 0) > 0 && (
              <Badge variant="warning" title="Reranker failed for these cases; retrieval fell back to RRF order">
                {data.rerank_fallbacks} rerank fallbacks
              </Badge>
            )}
          </span>
        }
        actions={
          <>
            <GateBadge passed={data.gate_passed} status={data.status} />
            {!isBaseline && (
              <Link href={`/evals/compare?base=baseline&head=${encodeURIComponent(data.run_id)}`} className={buttonVariants({ variant: "outline", size: "sm" })}>
                <GitCompare /> Compare with baseline
              </Link>
            )}
          </>
        }
      />

      {data.error && (
        <div className="mb-4">
          <ErrorState error={new Error(data.error)} compact />
        </div>
      )}
      {data.status === "running" && (
        <div className="mb-4 rounded-lg border bg-muted/40 p-3 text-sm text-muted-foreground">Run in progress. This page refreshes automatically.</div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {orderedMetricKeys(Object.keys(data.metrics)).map((k) => (
          <MetricCard key={k} name={k} m={data.metrics[k]} />
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4" /> ACL leakage
            </CardTitle>
            <CardDescription>Deterministic permission tests: zero forbidden chunks allowed.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span>{data.acl.cases} cases</span>
              <Badge variant={data.acl.passed ? "success" : "destructive"}>
                {data.acl.passed ? <Check className="h-3 w-3" /> : <X className="h-3 w-3" />} {data.acl.leaked_chunks} leaked chunks
              </Badge>
            </div>
            {data.acl.details.length > 0 && (
              <ul className="scrollbar-thin max-h-40 space-y-1 overflow-y-auto text-xs text-muted-foreground">
                {data.acl.details.map((d, i) => (
                  <li key={i} className="rounded border p-1.5 font-mono text-[11px]">
                    {Object.entries(d)
                      .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : String(v)}`)
                      .join(" · ")}
                  </li>
                ))}
              </ul>
            )}
            {data.gate_failures.length > 0 && (
              <div className="pt-2">
                <div className="mb-1 text-xs font-medium text-destructive">Gate failures</div>
                <ul className="list-disc space-y-1 pl-5 text-xs text-destructive">
                  {data.gate_failures.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="space-y-2">
            <CardTitle>Score distribution</CardTitle>
            <Tabs value={histMetric} onValueChange={setHistMetric}>
              <TabsList className="h-auto flex-wrap justify-start">
                {JUDGED_METRICS.map((m) => (
                  <TabsTrigger key={m} value={m} className="text-[11px]">
                    {metricLabel(m)}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </CardHeader>
          <CardContent>
            {histValues.length ? (
              <Histogram values={histValues} metric={histMetric} threshold={gates[histMetric]?.threshold} />
            ) : (
              <EmptyState title="No scores for this metric" />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>By question type</CardTitle>
            <CardDescription>Mean score per type: multi-hop questions usually stress recall.</CardDescription>
          </CardHeader>
          <CardContent>
            {byType.length ? (
              <GroupedBarChart
                data={byType}
                categoryKey="type"
                series={JUDGED_METRICS.map((m) => ({ key: m, label: metricLabel(m), color: METRIC_COLORS[m] }))}
                height={240}
              />
            ) : (
              <EmptyState title="No cases" />
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Cases</CardTitle>
          <CardDescription>
            {filtered.length} of {cases.length} shown. Click a row for the answer, expected answer, judge reasons and retrieved context.
          </CardDescription>
          <div className="grid gap-3 pt-2 sm:grid-cols-2 lg:grid-cols-[1fr_160px_160px_auto] lg:items-end">
            <div className="space-y-1.5">
              <Label htmlFor="case-search">Search</Label>
              <Input id="case-search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Question text or case id" />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="case-type">Question type</Label>
              <Select id="case-type" value={qtype} onChange={(e) => setQtype(e.target.value)}>
                <option value="all">All types</option>
                {qtypes.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="case-origin">Origin</Label>
              <Select id="case-origin" value={origin} onChange={(e) => setOrigin(e.target.value)}>
                <option value="all">All origins</option>
                {origins.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </Select>
            </div>
            <div className="flex h-9 items-center gap-2">
              <Switch id="failing-only" checked={failingOnly} onCheckedChange={setFailingOnly} label="Failing cases only" />
              <Label htmlFor="failing-only" className="text-foreground">
                Failing only
              </Label>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {filtered.length === 0 ? (
            <EmptyState title="No cases match the filters" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-6" />
                  <SortHeader label="Question" k="question" sort={sort} setSort={setSort} />
                  <SortHeader label="Type" k="question_type" sort={sort} setSort={setSort} />
                  {JUDGED_METRICS.map((m) => (
                    <SortHeader key={m} label={metricLabel(m)} k={m} sort={sort} setSort={setSort} />
                  ))}
                  <SortHeader label="Docs" k="doc_recall" sort={sort} setSort={setSort} />
                  <SortHeader label="Latency" k="latency_ms" sort={sort} setSort={setSort} className="text-right" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((c) => {
                  const open = expanded.has(c.case_id);
                  return (
                    <Fragment key={c.case_id}>
                      <TableRow
                        className="cursor-pointer"
                        onClick={() => toggle(c.case_id)}
                        aria-expanded={open}
                        data-state={open ? "selected" : undefined}
                      >
                        <TableCell className="px-2">{open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}</TableCell>
                        <TableCell className="max-w-[360px]">
                          <div className="truncate" title={c.question}>
                            {c.question}
                          </div>
                          <div className="font-mono text-[10px] text-muted-foreground">
                            {c.case_id} · {c.origin}
                            {c.error && <span className="ml-1 text-destructive">· error</span>}
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-[10px]">
                            {c.question_type}
                          </Badge>
                        </TableCell>
                        {JUDGED_METRICS.map((m) => (
                          <TableCell key={m}>
                            <ScoreBadge score={c.metrics[m]?.score} threshold={gates[m]?.threshold} />
                          </TableCell>
                        ))}
                        <TableCell>
                          {c.doc_recall ? <Check className="h-4 w-4 text-success" aria-label="all expected docs retrieved" /> : <X className="h-4 w-4 text-destructive" aria-label="expected doc missing" />}
                        </TableCell>
                        <TableCell className="whitespace-nowrap text-right text-xs tabular-nums">{fmtMs(c.latency_ms)}</TableCell>
                      </TableRow>
                      {open && (
                        <TableRow className="hover:bg-transparent">
                          <TableCell colSpan={9} className="bg-muted/20 p-0">
                            <CaseDetail c={c} gates={gates} />
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Run configuration</CardTitle>
          <CardDescription>Everything needed to reproduce this score.</CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 lg:grid-cols-3">
            {[
              ["Index", cfg.index_name],
              ["Embedding model", cfg.embedding_model],
              ["Prompt", `${cfg.prompt_name} @ ${shortSha(cfg.prompt_sha)}`],
              ["Synthesis route", cfg.synthesis_route],
              ["Synthesis model", cfg.synthesis_model],
              ["Judge model", cfg.judge_model],
              ["Reranker", cfg.rerank_model ?? "disabled"],
              ["Git", `${shortSha(cfg.git_sha)} ${cfg.git_branch ?? ""}`],
              ["Chunking", Object.entries(cfg.chunking ?? {}).map(([k, v]) => `${k}=${String(v)}`).join(", ") || "-"],
            ].map(([k, v]) => (
              <div key={k} className="min-w-0">
                <dt className="text-xs text-muted-foreground">{k}</dt>
                <dd className="truncate font-mono text-xs" title={String(v)}>
                  {v || "-"}
                </dd>
              </div>
            ))}
          </dl>
          {cfg.overrides && Object.values(cfg.overrides).some((v) => v !== null && v !== undefined) && (
            <div className="mt-4">
              <div className="mb-1 text-xs text-muted-foreground">Sandbox overrides</div>
              <pre className="scrollbar-thin max-h-48 overflow-auto rounded-lg bg-muted p-3 text-[11px]">
                {JSON.stringify(
                  Object.fromEntries(Object.entries(cfg.overrides).filter(([, v]) => v !== null && v !== undefined)),
                  null,
                  2,
                )}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function BackLink() {
  return (
    <Link href="/evals" className="mb-4 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
      <ArrowLeft className="h-3 w-3" /> All runs
    </Link>
  );
}
