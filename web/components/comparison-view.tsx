"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowDownRight, ArrowRight, ArrowUpRight, Minus } from "lucide-react";

import { GroupedBarChart } from "@/components/charts";
import { EmptyState, GateBadge, ScoreBadge, toneText } from "@/components/shared";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fmtDate, fmtDelta, fmtScore, runName } from "@/lib/format";
import { deltaTone, metricLabel, orderedMetricKeys } from "@/lib/metrics";
import type { CaseDelta, EvalComparison } from "@/lib/types";
import { cn } from "@/lib/utils";

function caseScore(c: CaseDelta): number {
  // Sum of metric deltas: most negative = worst regression.
  return Object.values(c.delta).reduce<number>((s, v) => s + (typeof v === "number" ? v : 0), 0);
}

function DeltaIcon({ delta }: { delta: number | null }) {
  const tone = deltaTone(delta);
  const Icon = tone === "success" ? ArrowUpRight : tone === "destructive" ? ArrowDownRight : Minus;
  return <Icon className={cn("h-4 w-4", toneText(tone))} aria-hidden />;
}

export function ComparisonView({ cmp, thresholds }: { cmp: EvalComparison; thresholds: Record<string, number> }) {
  const metrics = orderedMetricKeys(Object.keys(cmp.metric_deltas));
  const chartData = metrics.map((m) => ({
    metric: metricLabel(m),
    base: cmp.metric_deltas[m].base,
    head: cmp.metric_deltas[m].head,
  }));
  const cases = [...cmp.cases].sort((a, b) => caseScore(a) - caseScore(b));
  const caseMetrics = orderedMetricKeys(new Set(cases.flatMap((c) => Object.keys(c.delta))));
  const [open, setOpen] = useState<string | null>(null);
  const regressions = new Set(cmp.regressions);
  const improvements = new Set(cmp.improvements);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2">
        {[
          { side: "Base", run: cmp.base },
          { side: "Head", run: cmp.head },
        ].map(({ side, run }) => (
          <Card key={side}>
            <CardContent className="flex items-center justify-between gap-3 p-4">
              <div className="min-w-0">
                <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{side}</div>
                <Link href={`/evals/${encodeURIComponent(run.run_id)}`} className="block truncate font-medium text-primary hover:underline">
                  {runName(run)}
                </Link>
                <div className="text-xs text-muted-foreground">
                  {fmtDate(run.created_at)} · {run.num_cases} cases · prompt {run.config.prompt_sha?.slice(0, 7) || "-"}
                </div>
              </div>
              <GateBadge passed={run.gate_passed} status={run.status} />
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {metrics.map((m) => {
          const d = cmp.metric_deltas[m];
          return (
            <Card key={m}>
              <CardContent className="space-y-2 p-4">
                <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
                  {metricLabel(m)}
                  <DeltaIcon delta={d.delta} />
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-sm text-muted-foreground">{fmtScore(d.base, 3)}</span>
                  <ArrowRight className="h-3 w-3 text-muted-foreground" />
                  <span className="text-xl font-semibold tabular-nums">{fmtScore(d.head, 3)}</span>
                </div>
                <div className={cn("font-mono text-xs font-medium", toneText(deltaTone(d.delta)))}>{fmtDelta(d.delta)}</div>
                {thresholds[m] !== undefined && <div className="text-[11px] text-muted-foreground">gate ≥ {thresholds[m].toFixed(2)}</div>}
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <Card>
          <CardHeader>
            <CardTitle>Base vs head</CardTitle>
          </CardHeader>
          <CardContent>
            <GroupedBarChart
              data={chartData}
              categoryKey="metric"
              series={[
                { key: "base", label: "Base", color: "hsl(var(--muted-foreground))" },
                { key: "head", label: "Head", color: "hsl(var(--chart-1))" },
              ]}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Case movement</CardTitle>
            <CardDescription>Cases whose scores moved past the regression tolerance.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="mb-1 flex items-center gap-2 font-medium text-destructive">
                <ArrowDownRight className="h-4 w-4" /> {cmp.regressions.length} regressions
              </div>
              <div className="flex flex-wrap gap-1">
                {cmp.regressions.length ? (
                  cmp.regressions.map((id) => (
                    <button key={id} type="button" onClick={() => setOpen(id)} className="focus-visible:outline-none">
                      <Badge variant="destructive" className="font-mono text-[10px]">
                        {id}
                      </Badge>
                    </button>
                  ))
                ) : (
                  <span className="text-xs text-muted-foreground">None</span>
                )}
              </div>
            </div>
            <div>
              <div className="mb-1 flex items-center gap-2 font-medium text-success">
                <ArrowUpRight className="h-4 w-4" /> {cmp.improvements.length} improvements
              </div>
              <div className="flex flex-wrap gap-1">
                {cmp.improvements.length ? (
                  cmp.improvements.map((id) => (
                    <button key={id} type="button" onClick={() => setOpen(id)} className="focus-visible:outline-none">
                      <Badge variant="success" className="font-mono text-[10px]">
                        {id}
                      </Badge>
                    </button>
                  ))
                ) : (
                  <span className="text-xs text-muted-foreground">None</span>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Per-case diff</CardTitle>
          <CardDescription>Sorted by worst combined regression. Click a row to compare answers side by side.</CardDescription>
        </CardHeader>
        <CardContent>
          {cases.length === 0 ? (
            <EmptyState title="No overlapping cases" description="The two runs share no case ids." />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Case</TableHead>
                  {caseMetrics.map((m) => (
                    <TableHead key={m}>{metricLabel(m)}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {cases.map((c) => (
                  <CaseDiffRow
                    key={c.case_id}
                    c={c}
                    metrics={caseMetrics}
                    thresholds={thresholds}
                    open={open === c.case_id}
                    onToggle={() => setOpen(open === c.case_id ? null : c.case_id)}
                    tag={regressions.has(c.case_id) ? "regression" : improvements.has(c.case_id) ? "improvement" : null}
                  />
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function CaseDiffRow({
  c,
  metrics,
  thresholds,
  open,
  onToggle,
  tag,
}: {
  c: CaseDelta;
  metrics: string[];
  thresholds: Record<string, number>;
  open: boolean;
  onToggle: () => void;
  tag: "regression" | "improvement" | null;
}) {
  return (
    <>
      <TableRow className="cursor-pointer" onClick={onToggle} aria-expanded={open} data-state={open ? "selected" : undefined}>
        <TableCell className="max-w-[340px]">
          <div className="truncate" title={c.question}>
            {c.question}
          </div>
          <div className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
            {c.case_id}
            {tag && (
              <Badge variant={tag === "regression" ? "destructive" : "success"} className="px-1 py-0 text-[9px]">
                {tag}
              </Badge>
            )}
          </div>
        </TableCell>
        {metrics.map((m) => (
          <TableCell key={m} className="whitespace-nowrap">
            <div className="flex items-center gap-1.5">
              <ScoreBadge score={c.base[m]} threshold={thresholds[m]} className="opacity-70" />
              <ArrowRight className="h-3 w-3 text-muted-foreground" />
              <ScoreBadge score={c.head[m]} threshold={thresholds[m]} />
              <span className={cn("font-mono text-[11px]", toneText(deltaTone(c.delta[m])))}>{fmtDelta(c.delta[m], 2)}</span>
            </div>
          </TableCell>
        ))}
      </TableRow>
      {open && (
        <TableRow className="hover:bg-transparent">
          <TableCell colSpan={metrics.length + 1} className="bg-muted/20">
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Base answer</div>
                <div className="whitespace-pre-line rounded-lg border bg-background p-3 text-sm">{c.base_answer || "-"}</div>
              </div>
              <div>
                <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Head answer</div>
                <div className="whitespace-pre-line rounded-lg border bg-background p-3 text-sm">{c.head_answer || "-"}</div>
              </div>
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}
