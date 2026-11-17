"use client";

import { AlertTriangle, CheckCircle2, CircleDashed, Inbox, Loader2, ServerOff, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/misc";
import { ApiError, errorMessage } from "@/lib/api";
import { fmtScore } from "@/lib/format";
import { scoreTone, type Tone } from "@/lib/metrics";
import type { DocumentStatus, GuardrailAction, RunStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry, compact = false }: { error: unknown; onRetry?: () => void; compact?: boolean }) {
  const network = error instanceof ApiError && error.network;
  const Icon = network ? ServerOff : AlertTriangle;
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 text-sm sm:flex-row sm:items-center",
        compact ? "p-3" : "p-4",
      )}
    >
      <Icon className="h-5 w-5 shrink-0 text-destructive" />
      <div className="min-w-0 flex-1">
        <div className="font-medium text-destructive">{network ? "Backend unreachable" : "Request failed"}</div>
        <div className="break-words text-muted-foreground">
          {network ? (
            <>
              {errorMessage(error)} <span className="font-mono text-xs">(uvicorn on :8000)</span>
            </>
          ) : (
            errorMessage(error)
          )}
        </div>
      </div>
      {onRetry && (
        <Button size="sm" variant="outline" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function EmptyState({ title, description, action }: { title: string; description?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed p-8 text-center">
      <Inbox className="h-6 w-6 text-muted-foreground" />
      <div className="text-sm font-medium">{title}</div>
      {description && <div className="max-w-md text-xs text-muted-foreground">{description}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function LoadingCards({ count = 4, className }: { count?: number; className?: string }) {
  return (
    <div className={cn("grid gap-4 sm:grid-cols-2 xl:grid-cols-4", className)}>
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i}>
          <CardContent className="space-y-3 p-5">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-7 w-16" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function LoadingRows({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-9 w-full" />
      ))}
    </div>
  );
}

const TONE_TEXT: Record<Tone, string> = {
  success: "text-success",
  warning: "text-warning",
  destructive: "text-destructive",
  muted: "text-muted-foreground",
};

export function toneText(tone: Tone) {
  return TONE_TEXT[tone];
}

export function ScoreBadge({
  score,
  threshold,
  digits = 2,
  className,
}: {
  score: number | null | undefined;
  threshold?: number | null;
  digits?: number;
  className?: string;
}) {
  const tone = scoreTone(score, threshold ?? null);
  return (
    <Badge variant={tone} className={cn("font-mono tabular-nums", className)}>
      {fmtScore(score, digits)}
    </Badge>
  );
}

export function GateBadge({ passed, status }: { passed: boolean | null | undefined; status?: RunStatus }) {
  if (status === "running")
    return (
      <Badge variant="secondary">
        <Loader2 className="h-3 w-3 animate-spin" /> Running
      </Badge>
    );
  if (status === "failed")
    return (
      <Badge variant="destructive">
        <XCircle className="h-3 w-3" /> Errored
      </Badge>
    );
  if (passed === null || passed === undefined) return <Badge variant="muted">n/a</Badge>;
  return passed ? (
    <Badge variant="success">
      <CheckCircle2 className="h-3 w-3" /> PASS
    </Badge>
  ) : (
    <Badge variant="destructive">
      <XCircle className="h-3 w-3" /> FAIL
    </Badge>
  );
}

const DOC_STATUS_TONE: Record<DocumentStatus, Tone | "secondary"> = {
  queued: "secondary",
  parsing: "secondary",
  chunking: "secondary",
  embedding: "secondary",
  indexed: "success",
  failed: "destructive",
};

export function DocStatusBadge({ status }: { status: DocumentStatus }) {
  const busy = status !== "indexed" && status !== "failed";
  return (
    <Badge variant={DOC_STATUS_TONE[status] ?? "muted"}>
      {busy ? <CircleDashed className="h-3 w-3 animate-spin" /> : null}
      {status}
    </Badge>
  );
}

const GUARD_TONE: Record<GuardrailAction, Tone> = {
  allow: "success",
  anonymize: "warning",
  flag: "warning",
  block: "destructive",
};

export function GuardrailBadge({ action }: { action: GuardrailAction }) {
  return <Badge variant={GUARD_TONE[action] ?? "muted"}>{action}</Badge>;
}

export function StatTile({
  label,
  value,
  hint,
  icon,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
          <span>{label}</span>
          {icon}
        </div>
        <div className="mt-2 truncate text-2xl font-semibold tabular-nums">{value}</div>
        {hint && <div className="mt-1 truncate text-xs text-muted-foreground">{hint}</div>}
      </CardContent>
    </Card>
  );
}
