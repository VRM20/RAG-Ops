"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";

import { ComparisonView } from "@/components/comparison-view";
import { EmptyState, ErrorState, LoadingCards, LoadingRows, PageHeader } from "@/components/shared";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { fmtDate, runName } from "@/lib/format";
import { useAsync } from "@/lib/hooks";
import type { EvalComparison } from "@/lib/types";

function CompareInner() {
  const router = useRouter();
  const params = useSearchParams();
  const runs = useAsync(() => api.evalRuns(), []);
  const config = useAsync(() => api.config(), []);
  const [base, setBase] = useState(params.get("base") ?? "baseline");
  const [head, setHead] = useState(params.get("head") ?? "");

  const completed = useMemo(() => (runs.data ?? []).filter((r) => r.status === "completed"), [runs.data]);

  useEffect(() => {
    if (!head && completed.length) setHead(completed[0].run_id);
  }, [completed, head]);

  useEffect(() => {
    if (!head) return;
    const qs = new URLSearchParams({ base, head });
    router.replace(`/evals/compare?${qs.toString()}`, { scroll: false });
  }, [base, head, router]);

  const cmp = useAsync<EvalComparison | null>(() => (head ? api.compare(base, head) : Promise.resolve(null)), [base, head]);
  const thresholds = Object.fromEntries(Object.entries(config.data?.gates.metrics ?? {}).map(([k, g]) => [k, g.threshold]));

  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader title="Compare runs" description="Metric deltas and per-case regressions between two evaluation runs: the same view the CI gate posts on a pull request." />

      <Card className="mb-6">
        <CardContent className="grid gap-3 p-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="cmp-base">Base</Label>
            <Select id="cmp-base" value={base} onChange={(e) => setBase(e.target.value)}>
              <option value="baseline">CI baseline (evals/baselines)</option>
              {completed.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {runName(r)} · {fmtDate(r.created_at)}
                </option>
              ))}
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cmp-head">Head</Label>
            <Select id="cmp-head" value={head} onChange={(e) => setHead(e.target.value)} disabled={!completed.length}>
              {!completed.length && <option value="">No completed runs</option>}
              {completed.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {runName(r)} · {fmtDate(r.created_at)}
                </option>
              ))}
            </Select>
          </div>
        </CardContent>
      </Card>

      {runs.error ? (
        <ErrorState error={runs.error} onRetry={() => void runs.reload()} />
      ) : !runs.data ? (
        <LoadingCards />
      ) : !head ? (
        <EmptyState title="Nothing to compare yet" description="Complete at least one eval run first." />
      ) : cmp.error ? (
        <ErrorState error={cmp.error} onRetry={() => void cmp.reload()} />
      ) : cmp.loading || !cmp.data ? (
        <div className="space-y-6">
          <LoadingCards count={5} className="xl:grid-cols-5" />
          <LoadingRows rows={6} />
        </div>
      ) : (
        <ComparisonView cmp={cmp.data} thresholds={thresholds} />
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<LoadingCards />}>
      <CompareInner />
    </Suspense>
  );
}
