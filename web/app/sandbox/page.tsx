"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { FlaskConical, Info, Loader2, Play, RotateCcw, Send } from "lucide-react";

import { ComparisonView } from "@/components/comparison-view";
import { MarkdownAnswer } from "@/components/markdown-answer";
import { useApp } from "@/components/providers";
import { ErrorState, GateBadge, LoadingRows, PageHeader, ScoreBadge, toneText } from "@/components/shared";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/misc";
import { Select } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { api, ApiError } from "@/lib/api";
import { fmtDelta, fmtMs, fmtScore } from "@/lib/format";
import { useAsync } from "@/lib/hooks";
import { deltaTone, metricLabel, orderedMetricKeys } from "@/lib/metrics";
import type { ConfigResponse, DatasetChoice, EvalComparison, EvalRun, PipelineOverrides, RAGResponse } from "@/lib/types";

function get<T>(obj: unknown, path: string[], fallback: T): T {
  let cur: unknown = obj;
  for (const key of path) {
    if (cur && typeof cur === "object" && key in (cur as Record<string, unknown>)) cur = (cur as Record<string, unknown>)[key];
    else return fallback;
  }
  return (cur as T) ?? fallback;
}

interface Defaults {
  route: string;
  rerankEnabled: boolean;
  rerankTopN: number;
  maxContextChunks: number;
}

function defaultsFrom(config: ConfigResponse | undefined): Defaults {
  return {
    route: get(config?.rag, ["generation", "route"], "synthesis"),
    rerankEnabled: get(config?.rag, ["retrieval", "rerank", "enabled"], true),
    rerankTopN: get(config?.rag, ["retrieval", "rerank", "top_n"], 5),
    maxContextChunks: get(config?.rag, ["generation", "max_context_chunks"], 5),
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function SandboxPage() {
  const { userId } = useApp();
  const config = useAsync(() => api.config(), []);
  const promptName = config.data?.prompt.name;
  const prompt = useAsync(() => (promptName ? api.prompt(promptName) : Promise.resolve(undefined)), [promptName]);
  const defaults = useMemo(() => defaultsFrom(config.data), [config.data]);

  const [system, setSystem] = useState("");
  const [userTemplate, setUserTemplate] = useState("");
  const [route, setRoute] = useState("");
  const [rerankEnabled, setRerankEnabled] = useState(true);
  const [rerankTopN, setRerankTopN] = useState(5);
  const [maxChunks, setMaxChunks] = useState(5);
  const [maxCases, setMaxCases] = useState(8);
  const [dataset, setDataset] = useState<DatasetChoice>("curated");

  const [testQuery, setTestQuery] = useState("What does error code E101 mean and how do I fix it?");
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<RAGResponse | null>(null);
  const [testError, setTestError] = useState<unknown>(null);

  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<EvalRun | null>(null);
  const [cmp, setCmp] = useState<EvalComparison | null>(null);
  const [noBaseline, setNoBaseline] = useState(false);
  const [runError, setRunError] = useState<unknown>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    if (prompt.data) {
      setSystem(prompt.data.system);
      setUserTemplate(prompt.data.user_template);
    }
  }, [prompt.data]);

  useEffect(() => {
    setRoute(defaults.route);
    setRerankEnabled(defaults.rerankEnabled);
    setRerankTopN(defaults.rerankTopN);
    setMaxChunks(defaults.maxContextChunks);
  }, [defaults]);

  useEffect(() => {
    cancelled.current = false;
    return () => {
      cancelled.current = true;
    };
  }, []);

  const promptChanged = Boolean(prompt.data) && (system !== prompt.data?.system || userTemplate !== prompt.data?.user_template);
  const templateMissing = !userTemplate.includes("{context}") || !userTemplate.includes("{question}");

  const overrides: PipelineOverrides = useMemo(
    () => ({
      system_prompt: prompt.data && system !== prompt.data.system ? system : null,
      user_template: prompt.data && userTemplate !== prompt.data.user_template ? userTemplate : null,
      route: route && route !== defaults.route ? route : null,
      rerank_enabled: rerankEnabled !== defaults.rerankEnabled ? rerankEnabled : null,
      rerank_top_n: rerankTopN !== defaults.rerankTopN ? rerankTopN : null,
      max_context_chunks: maxChunks !== defaults.maxContextChunks ? maxChunks : null,
    }),
    [prompt.data, system, userTemplate, route, rerankEnabled, rerankTopN, maxChunks, defaults],
  );
  const activeOverrides = Object.entries(overrides).filter(([, v]) => v !== null);

  const reset = () => {
    if (prompt.data) {
      setSystem(prompt.data.system);
      setUserTemplate(prompt.data.user_template);
    }
    setRoute(defaults.route);
    setRerankEnabled(defaults.rerankEnabled);
    setRerankTopN(defaults.rerankTopN);
    setMaxChunks(defaults.maxContextChunks);
  };

  const quickTest = async () => {
    if (!testQuery.trim() || !userId) return;
    setTesting(true);
    setTestError(null);
    try {
      setTestResult(await api.chat({ query: testQuery.trim(), user_id: userId, overrides }));
    } catch (err) {
      setTestError(err);
    } finally {
      setTesting(false);
    }
  };

  const runGolden = async () => {
    setRunning(true);
    setRunError(null);
    setRun(null);
    setCmp(null);
    setNoBaseline(false);
    try {
      const started = await api.startEval({ label: "sandbox", max_cases: maxCases, dataset, overrides });
      let current: EvalRun = await api.evalRun(started.run_id);
      setRun(current);
      while (current.status === "running" && !cancelled.current) {
        await sleep(3000);
        current = await api.evalRun(started.run_id);
        setRun(current);
      }
      if (current.status === "completed") {
        try {
          setCmp(await api.compare("baseline", current.run_id));
        } catch (err) {
          if (err instanceof ApiError && err.status === 404) setNoBaseline(true);
          else throw err;
        }
      }
    } catch (err) {
      setRunError(err);
    } finally {
      setRunning(false);
    }
  };

  const gates = config.data?.gates.metrics ?? {};
  const thresholds = Object.fromEntries(Object.entries(gates).map(([k, g]) => [k, g.threshold]));
  const routes = Object.keys(config.data?.routes ?? {});

  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader
        title="Prompt Sandbox"
        description="Edit the system prompt and generation knobs, try them on one question, then score them on a slice of the golden set against the CI baseline before opening a PR."
        actions={
          <Button variant="outline" size="sm" onClick={reset} disabled={!promptChanged && activeOverrides.length === 0}>
            <RotateCcw /> Reset to current config
          </Button>
        }
      />

      <div className="mb-6 flex items-start gap-2 rounded-lg border bg-muted/40 p-3 text-xs text-muted-foreground">
        <Info className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          Only generation- and retrieval-time knobs can be overridden here. Chunking and embedding changes need a re-index: edit{" "}
          <span className="font-mono">config/rag.yaml</span>, re-ingest, and let the CI eval gate judge the PR.
        </span>
      </div>

      {config.error ? (
        <ErrorState error={config.error} onRetry={() => void config.reload()} />
      ) : (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
          <Card>
            <CardHeader className="flex-row items-start justify-between gap-3 space-y-0">
              <div className="space-y-1">
                <CardTitle>
                  Prompt <span className="font-mono text-xs text-muted-foreground">{promptName ?? "…"}</span>
                </CardTitle>
                <CardDescription>
                  {prompt.data ? <>sha {prompt.data.sha} · {prompt.data.description}</> : "Loading prompt…"}
                </CardDescription>
              </div>
              {promptChanged && <Badge variant="warning">modified</Badge>}
            </CardHeader>
            <CardContent className="space-y-4">
              {prompt.error ? (
                <ErrorState error={prompt.error} onRetry={() => void prompt.reload()} compact />
              ) : !prompt.data ? (
                <LoadingRows rows={6} />
              ) : (
                <>
                  <div className="space-y-1.5">
                    <Label htmlFor="sb-system">System prompt</Label>
                    <Textarea id="sb-system" value={system} onChange={(e) => setSystem(e.target.value)} rows={14} className="font-mono text-xs leading-relaxed" />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="sb-user">User template</Label>
                    <Textarea id="sb-user" value={userTemplate} onChange={(e) => setUserTemplate(e.target.value)} rows={5} className="font-mono text-xs" />
                    {templateMissing && (
                      <p className="text-xs text-destructive">
                        The template must contain <span className="font-mono">{"{context}"}</span> and <span className="font-mono">{"{question}"}</span>.
                      </p>
                    )}
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Generation knobs</CardTitle>
                <CardDescription>Defaults come from config/rag.yaml.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="sb-route">Model route</Label>
                  <Select id="sb-route" value={route} onChange={(e) => setRoute(e.target.value)}>
                    {routes.map((r) => (
                      <option key={r} value={r}>
                        {r} → {config.data?.routes[r].primary}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex items-center justify-between">
                  <Label htmlFor="sb-rerank" className="text-foreground">
                    Cohere rerank
                  </Label>
                  <Switch id="sb-rerank" checked={rerankEnabled} onCheckedChange={setRerankEnabled} label="Enable reranking" />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="sb-topn">Rerank top-n</Label>
                    <Input id="sb-topn" type="number" min={1} max={20} value={rerankTopN} onChange={(e) => setRerankTopN(Number(e.target.value) || 1)} disabled={!rerankEnabled} />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="sb-chunks">Context chunks</Label>
                    <Input id="sb-chunks" type="number" min={1} max={20} value={maxChunks} onChange={(e) => setMaxChunks(Number(e.target.value) || 1)} />
                  </div>
                </div>
                <div className="flex flex-wrap gap-1">
                  {activeOverrides.length ? (
                    activeOverrides.map(([k]) => (
                      <Badge key={k} variant="secondary" className="font-mono text-[10px]">
                        {k}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-xs text-muted-foreground">No overrides: identical to the current config.</span>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Send className="h-4 w-4" /> Quick test
                </CardTitle>
                <CardDescription>One question, as the persona selected in the sidebar.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Textarea aria-label="Test question" value={testQuery} onChange={(e) => setTestQuery(e.target.value)} rows={2} />
                <Button onClick={() => void quickTest()} disabled={testing || !testQuery.trim() || templateMissing || !userId} className="w-full">
                  {testing ? <Loader2 className="animate-spin" /> : <Send />} Ask with these settings
                </Button>
                {testError ? <ErrorState error={testError} compact /> : null}
                {testResult && (
                  <div className="space-y-2 rounded-lg border p-3">
                    {testResult.blocked ? (
                      <p className="text-sm text-destructive">{testResult.answer}</p>
                    ) : (
                      <MarkdownAnswer text={testResult.answer} maxSource={testResult.contexts.length || undefined} />
                    )}
                    <div className="flex flex-wrap gap-1.5 border-t pt-2 text-[11px] text-muted-foreground">
                      {testResult.route && <Badge variant="secondary">{testResult.route}</Badge>}
                      <span className="max-w-[200px] truncate font-mono">{testResult.model}</span>
                      {testResult.timings_ms.total !== undefined && <span>· {fmtMs(testResult.timings_ms.total)}</span>}
                      <span>· {testResult.contexts.length} sources</span>
                    </div>
                    <ol className="space-y-0.5 text-[11px] text-muted-foreground">
                      {testResult.contexts.map((c, i) => (
                        <li key={c.chunk.chunk_id} className="truncate">
                          [{i + 1}] {c.chunk.doc_title} › {c.chunk.section_path.join(" › ")}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4" /> Run against the golden set
          </CardTitle>
          <CardDescription>Starts an eval run labelled &quot;sandbox&quot; with these overrides, then compares it with the CI baseline.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-[140px_180px_auto] sm:items-end">
            <div className="space-y-1.5">
              <Label htmlFor="sb-cases">Max cases</Label>
              <Input id="sb-cases" type="number" min={1} max={500} value={maxCases} onChange={(e) => setMaxCases(Math.max(1, Number(e.target.value) || 1))} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="sb-dataset">Dataset</Label>
              <Select id="sb-dataset" value={dataset} onChange={(e) => setDataset(e.target.value as DatasetChoice)}>
                <option value="curated">Curated</option>
                <option value="synthetic">Synthetic</option>
                <option value="all">All</option>
              </Select>
            </div>
            <Button onClick={() => void runGolden()} disabled={running || templateMissing || !prompt.data}>
              {running ? <Loader2 className="animate-spin" /> : <Play />} {running ? "Evaluating…" : "Run evaluation"}
            </Button>
          </div>

          {runError ? <ErrorState error={runError} compact /> : null}

          {run && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <GateBadge passed={run.gate_passed} status={run.status} />
                <Link href={`/evals/${encodeURIComponent(run.run_id)}`} className="font-mono text-xs text-primary hover:underline">
                  {run.run_id}
                </Link>
                <span className="text-xs text-muted-foreground">
                  {run.status === "running" ? `${run.cases.length} / ${run.num_cases || maxCases} cases scored…` : `${run.num_cases} cases`}
                </span>
                {cmp && (
                  <Link
                    href={`/evals/compare?base=baseline&head=${encodeURIComponent(run.run_id)}`}
                    className={buttonVariants({ variant: "outline", size: "sm", className: "ml-auto" })}
                  >
                    Open full comparison
                  </Link>
                )}
              </div>
              {run.error && <ErrorState error={new Error(run.error)} compact />}

              {run.status === "completed" && (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Metric</TableHead>
                      <TableHead>Baseline</TableHead>
                      <TableHead>Sandbox</TableHead>
                      <TableHead>Δ</TableHead>
                      <TableHead>Gate</TableHead>
                      <TableHead>Verdict</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {orderedMetricKeys(Object.keys(run.metrics)).map((k) => {
                      const m = run.metrics[k];
                      const base = cmp?.metric_deltas[k]?.base ?? null;
                      const delta = cmp?.metric_deltas[k]?.delta ?? null;
                      return (
                        <TableRow key={k}>
                          <TableCell className="font-medium">
                            {metricLabel(k)} {m.blocking && <Badge variant="outline" className="ml-1 text-[10px]">blocking</Badge>}
                          </TableCell>
                          <TableCell className="font-mono text-xs">{noBaseline ? "n/a" : fmtScore(base, 3)}</TableCell>
                          <TableCell>
                            <ScoreBadge score={m.mean} threshold={m.threshold} digits={3} />
                          </TableCell>
                          <TableCell className={`font-mono text-xs ${toneText(deltaTone(delta))}`}>{noBaseline ? "-" : fmtDelta(delta)}</TableCell>
                          <TableCell className="text-xs text-muted-foreground">≥ {m.threshold.toFixed(2)}</TableCell>
                          <TableCell>
                            <Badge variant={m.passed ? "success" : m.blocking ? "destructive" : "warning"}>{m.passed ? "PASS" : "FAIL"}</Badge>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
              {noBaseline && <p className="text-xs text-muted-foreground">No CI baseline committed yet: showing absolute scores against the gates only.</p>}
              {run.status === "failed" && !run.error && <p className="text-sm text-destructive">The evaluation run failed. Check the API logs.</p>}
            </div>
          )}
        </CardContent>
      </Card>

      {cmp && (
        <div className="mt-6">
          <h2 className="mb-3 text-sm font-semibold">Comparison with baseline</h2>
          <ComparisonView cmp={cmp} thresholds={thresholds} />
        </div>
      )}
    </div>
  );
}
