"use client";

import Link from "next/link";
import { Activity, Boxes, Database, FileText, ShieldCheck } from "lucide-react";

import { TrendChart, type TrendPoint } from "@/components/charts";
import { useApp } from "@/components/providers";
import { EmptyState, ErrorState, GateBadge, LoadingCards, PageHeader, ScoreBadge, StatTile } from "@/components/shared";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/misc";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import { fmtDate, runName, shortSha } from "@/lib/format";
import { useAsync } from "@/lib/hooks";
import { JUDGED_METRICS, metricLabel } from "@/lib/metrics";
import type { ProviderStatus } from "@/lib/types";

function ProviderChips({ p }: { p: ProviderStatus }) {
  const chips: { label: string; ok: boolean; value?: string }[] = [
    { label: "Vertex AI", ok: p.vertex },
    { label: "Bedrock", ok: p.bedrock },
    { label: "Cohere rerank", ok: p.cohere },
    { label: "Guardrail", ok: p.guardrail === "bedrock", value: p.guardrail },
    { label: "Qdrant", ok: true, value: p.qdrant },
    { label: "Tasks", ok: true, value: p.task_queue },
    { label: "Tracing", ok: p.tracing, value: p.tracing ? "OTLP" : "off" },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((c) => (
        <Badge key={c.label} variant={c.ok ? "success" : "muted"} className="gap-1.5 py-1">
          <span className={c.ok ? "h-1.5 w-1.5 rounded-full bg-success" : "h-1.5 w-1.5 rounded-full bg-muted-foreground/60"} />
          {c.label}
          {c.value && <span className="font-mono opacity-80">{c.value}</span>}
        </Badge>
      ))}
    </div>
  );
}

export default function OverviewPage() {
  const { health, healthError, refreshHealth } = useApp();
  const config = useAsync(() => api.config(), []);
  const runs = useAsync(() => api.evalRuns(), []);

  const gates = config.data?.gates.metrics ?? {};
  const completed = (runs.data ?? []).filter((r) => r.status === "completed");
  const latest = completed[0];
  const trend: TrendPoint[] = [...completed]
    .reverse()
    .slice(-20)
    .map((r) => {
      const point: TrendPoint = { name: runName(r) };
      for (const m of JUDGED_METRICS) point[m] = r.metrics[m]?.mean ?? null;
      return point;
    });
  const thresholds = Object.fromEntries(
    Object.entries(gates)
      .filter(([, g]) => g.blocking)
      .map(([k, g]) => [k, g.threshold]),
  );

  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader
        title="Overview"
        description="Health of the knowledge engine: index, providers, model routing and the evaluation quality gate."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={() => void refreshHealth()}>
              Refresh
            </Button>
            <Link href="/chat" className={buttonVariants({ size: "sm" })}>
              Open RAG chat
            </Link>
          </>
        }
      />

      {healthError && !health ? (
        <ErrorState error={healthError} onRetry={() => void refreshHealth()} />
      ) : !health ? (
        <LoadingCards />
      ) : (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatTile label="Documents" value={health.index.documents} icon={<FileText className="h-4 w-4" />} hint="indexed in the active collection" />
            <StatTile label="Chunks" value={health.index.chunks.toLocaleString()} icon={<Boxes className="h-4 w-4" />} hint="dense + BM25 vectors" />
            <StatTile
              label="Active index"
              value={<span className="font-mono text-base">{health.index.name}</span>}
              icon={<Database className="h-4 w-4" />}
              hint="fingerprint of embedding + chunking config"
            />
            <StatTile
              label="Mode"
              value={health.offline ? <Badge variant="warning" className="text-sm">Offline</Badge> : <Badge variant="success" className="text-sm">Cloud</Badge>}
              icon={<Activity className="h-4 w-4" />}
              hint={health.offline ? "deterministic local stand-ins, scores not meaningful" : "Vertex AI + Bedrock providers"}
            />
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Providers</CardTitle>
              <CardDescription>Credentials detected by the API. Grey chips fall back to local stand-ins.</CardDescription>
            </CardHeader>
            <CardContent>
              <ProviderChips p={health.providers} />
            </CardContent>
          </Card>
        </div>
      )}

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader className="flex-row items-start justify-between gap-4 space-y-0">
            <div className="space-y-1">
              <CardTitle>Quality trend</CardTitle>
              <CardDescription>Dataset mean per eval run. Dashed lines mark blocking gates.</CardDescription>
            </div>
            <Link href="/evals" className="text-xs text-primary hover:underline">
              All runs
            </Link>
          </CardHeader>
          <CardContent>
            {runs.error ? (
              <ErrorState error={runs.error} onRetry={() => void runs.reload()} compact />
            ) : runs.loading && !runs.data ? (
              <Skeleton className="h-[260px] w-full" />
            ) : trend.length === 0 ? (
              <EmptyState title="No completed eval runs yet" description="Start one from the Evaluations page or run `make eval`." />
            ) : (
              <TrendChart data={trend} metrics={[...JUDGED_METRICS]} thresholds={thresholds} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4" /> Latest gate
            </CardTitle>
            <CardDescription>
              {latest ? (
                <>
                  <Link href={`/evals/${latest.run_id}`} className="text-primary hover:underline">
                    {runName(latest)}
                  </Link>{" "}
                  · {fmtDate(latest.created_at)} · {shortSha(latest.config.git_sha)}
                </>
              ) : (
                "No runs yet"
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {latest ? (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Gate status</span>
                  <GateBadge passed={latest.gate_passed} status={latest.status} />
                </div>
                {Object.entries(latest.metrics).map(([k, m]) => (
                  <div key={k} className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      {metricLabel(k)}
                      {m.blocking && <Badge variant="outline" className="text-[10px]">blocking</Badge>}
                    </span>
                    <span className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">≥ {m.threshold.toFixed(2)}</span>
                      <ScoreBadge score={m.mean} threshold={m.threshold} />
                    </span>
                  </div>
                ))}
                <div className="flex items-center justify-between text-sm">
                  <span>ACL leakage</span>
                  <Badge variant={latest.acl.passed ? "success" : "destructive"}>
                    {latest.acl.leaked_chunks} leaked / {latest.acl.cases} cases
                  </Badge>
                </div>
                {latest.gate_failures.length > 0 && (
                  <ul className="list-disc space-y-1 pl-5 text-xs text-destructive">
                    {latest.gate_failures.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <p className="text-sm text-muted-foreground">Run an evaluation to see the gate verdict here.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Model routing</CardTitle>
          <CardDescription>Task routes from config/models.yaml: primary model with ordered cross-provider fallbacks.</CardDescription>
        </CardHeader>
        <CardContent>
          {config.error ? (
            <ErrorState error={config.error} onRetry={() => void config.reload()} compact />
          ) : !config.data ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Route</TableHead>
                  <TableHead>Primary</TableHead>
                  <TableHead>Fallbacks</TableHead>
                  <TableHead>Resolved models</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(config.data.routes).map(([name, r]) => (
                  <TableRow key={name}>
                    <TableCell className="font-medium">{name}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{r.primary}</Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {r.fallbacks.length ? r.fallbacks.map((f) => <Badge key={f} variant="outline">{f}</Badge>) : <span className="text-muted-foreground">-</span>}
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {r.models.map((m, i) => (
                        <div key={`${m}-${i}`} className="truncate" title={m}>
                          {i + 1}. {m}
                        </div>
                      ))}
                    </TableCell>
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
