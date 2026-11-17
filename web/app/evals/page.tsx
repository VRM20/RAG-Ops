"use client";

import Link from "next/link";
import { useState } from "react";
import { FlaskConical, GitCompare, Loader2, Play, RefreshCw } from "lucide-react";

import { EmptyState, ErrorState, GateBadge, LoadingRows, PageHeader, ScoreBadge } from "@/components/shared";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import { fmtDate, fmtDuration, runName, shortSha } from "@/lib/format";
import { useAsync, useInterval } from "@/lib/hooks";
import { metricLabel, orderedMetricKeys } from "@/lib/metrics";
import type { DatasetChoice, EvalRunSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

function RunMetricsInline({ run }: { run: EvalRunSummary }) {
  const keys = orderedMetricKeys(Object.keys(run.metrics));
  if (!keys.length) return <span className="text-xs text-muted-foreground">-</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {keys.map((k) => (
        <span key={k} title={`${metricLabel(k)} (gate ${run.metrics[k].threshold})`}>
          <ScoreBadge score={run.metrics[k].mean} threshold={run.metrics[k].threshold} />
        </span>
      ))}
    </div>
  );
}

function StartRunCard({ onStarted }: { onStarted: (run: EvalRunSummary) => void }) {
  const [label, setLabel] = useState("");
  const [maxCases, setMaxCases] = useState("");
  const [dataset, setDataset] = useState<DatasetChoice>("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const run = await api.startEval({
        label: label.trim() || "manual",
        max_cases: maxCases ? Number(maxCases) : null,
        dataset,
      });
      onStarted(run);
      setLabel("");
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-4 w-4" /> Run evaluation
        </CardTitle>
        <CardDescription>Runs the golden set through the live pipeline and scores it with the DeepEval judge. This calls real models.</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="grid gap-3 sm:grid-cols-[1fr_120px_160px_auto] sm:items-end">
          <div className="space-y-1.5">
            <Label htmlFor="run-label">Label</Label>
            <Input id="run-label" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. chunk-350-rerank" />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="run-max">Max cases</Label>
            <Input id="run-max" type="number" min={1} max={500} value={maxCases} onChange={(e) => setMaxCases(e.target.value)} placeholder="all" />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="run-dataset">Dataset</Label>
            <Select id="run-dataset" value={dataset} onChange={(e) => setDataset(e.target.value as DatasetChoice)}>
              <option value="all">All cases</option>
              <option value="curated">Curated only</option>
              <option value="synthetic">Synthetic only</option>
            </Select>
          </div>
          <Button type="submit" disabled={busy}>
            {busy ? <Loader2 className="animate-spin" /> : <Play />} Start
          </Button>
        </form>
        {error ? (
          <div className="mt-3">
            <ErrorState error={error} compact />
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

export default function EvalsPage() {
  const runs = useAsync(() => api.evalRuns(), []);
  const baseline = useAsync(() => api.baseline(), []);
  const running = (runs.data ?? []).some((r) => r.status === "running");
  useInterval(() => void runs.reload(), 3000, running);

  const base = baseline.data;

  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader
        title="Evaluations"
        description="DeepEval runs over the golden dataset. Blocking gates (faithfulness, context recall) and the ACL leakage check decide whether a PR can merge."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={() => void runs.reload()}>
              <RefreshCw className={cn(runs.loading && "animate-spin")} /> Refresh
            </Button>
            <Link href="/evals/compare" className={buttonVariants({ variant: "outline", size: "sm" })}>
              <GitCompare /> Compare
            </Link>
            <Link href="/sandbox" className={buttonVariants({ size: "sm" })}>
              <FlaskConical /> Prompt sandbox
            </Link>
          </>
        }
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <StartRunCard onStarted={(run) => runs.setData((prev) => [run, ...(prev ?? []).filter((r) => r.run_id !== run.run_id)])} />

        <Card>
          <CardHeader>
            <CardTitle>CI baseline</CardTitle>
            <CardDescription>evals/baselines/baseline.json: what PRs are compared against.</CardDescription>
          </CardHeader>
          <CardContent>
            {baseline.error ? (
              <ErrorState error={baseline.error} compact />
            ) : baseline.loading && baseline.data === undefined ? (
              <LoadingRows rows={3} />
            ) : !base ? (
              <p className="text-sm text-muted-foreground">No baseline yet. Generate one with `make baseline` on main and commit it.</p>
            ) : (
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <Link href="/evals/baseline" className="font-medium text-primary hover:underline">
                    {runName(base)}
                  </Link>
                  <GateBadge passed={base.gate_passed} status={base.status} />
                </div>
                <div className="text-xs text-muted-foreground">
                  {fmtDate(base.created_at)} · {base.num_cases} cases · git {shortSha(base.config.git_sha)}
                </div>
                <RunMetricsInline run={base} />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Runs</CardTitle>
          <CardDescription>Metric badges are dataset means, colored against each gate threshold.</CardDescription>
        </CardHeader>
        <CardContent>
          {runs.error && !runs.data ? (
            <ErrorState error={runs.error} onRetry={() => void runs.reload()} />
          ) : !runs.data ? (
            <LoadingRows />
          ) : runs.data.length === 0 ? (
            <EmptyState title="No eval runs yet" description="Start a run above, or run `make eval` locally. CI runs appear when their reports are synced." />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run</TableHead>
                  <TableHead>Gate</TableHead>
                  <TableHead>Metrics</TableHead>
                  <TableHead>ACL</TableHead>
                  <TableHead className="text-right">Cases</TableHead>
                  <TableHead>Prompt</TableHead>
                  <TableHead>Git</TableHead>
                  <TableHead className="text-right">Duration</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.data.map((r) => (
                  <TableRow key={r.run_id}>
                    <TableCell className="max-w-[220px]">
                      <Link href={`/evals/${encodeURIComponent(r.run_id)}`} className="block truncate font-medium text-primary hover:underline">
                        {runName(r)}
                      </Link>
                      <div className="text-[11px] text-muted-foreground">{fmtDate(r.created_at)}</div>
                    </TableCell>
                    <TableCell>
                      <GateBadge passed={r.gate_passed} status={r.status} />
                      {r.error && (
                        <div className="mt-1 max-w-[180px] truncate text-[11px] text-destructive" title={r.error}>
                          {r.error}
                        </div>
                      )}
                    </TableCell>
                    <TableCell>
                      <RunMetricsInline run={r} />
                    </TableCell>
                    <TableCell>
                      {r.acl.cases > 0 ? (
                        <Badge variant={r.acl.passed ? "success" : "destructive"}>{r.acl.leaked_chunks} leaked</Badge>
                      ) : (
                        <span className="text-xs text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{r.num_cases}</TableCell>
                    <TableCell className="font-mono text-[11px] text-muted-foreground">
                      <div>{r.config.prompt_name || "-"}</div>
                      <div>{shortSha(r.config.prompt_sha)}</div>
                    </TableCell>
                    <TableCell className="font-mono text-[11px] text-muted-foreground">
                      <div>{shortSha(r.config.git_sha)}</div>
                      <div className="max-w-[100px] truncate">{r.config.git_branch ?? ""}</div>
                    </TableCell>
                    <TableCell className="whitespace-nowrap text-right text-xs tabular-nums">{fmtDuration(r.duration_s)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
