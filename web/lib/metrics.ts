// Metric keys, labels and display helpers shared by dashboards.

export const METRIC_KEYS = [
  "faithfulness",
  "contextual_recall",
  "answer_relevancy",
  "contextual_precision",
  "doc_recall",
] as const;

export type MetricKey = (typeof METRIC_KEYS)[number];

/** LLM-judged metrics (doc_recall is deterministic and per-case boolean). */
export const JUDGED_METRICS = ["faithfulness", "contextual_recall", "answer_relevancy", "contextual_precision"] as const;

export const METRIC_LABELS: Record<string, string> = {
  faithfulness: "Faithfulness",
  contextual_recall: "Context recall",
  answer_relevancy: "Answer relevancy",
  contextual_precision: "Context precision",
  doc_recall: "Doc recall",
};

export const METRIC_DESCRIPTIONS: Record<string, string> = {
  faithfulness: "Share of answer claims supported by the retrieved context (hallucination guard).",
  contextual_recall: "Share of the expected answer covered by the retrieved context.",
  answer_relevancy: "How directly the answer addresses the question.",
  contextual_precision: "Whether relevant chunks are ranked above irrelevant ones.",
  doc_recall: "Deterministic: every expected source document appears in the final context.",
};

/** CSS variable per metric so charts follow light/dark theme. */
export const METRIC_COLORS: Record<string, string> = {
  faithfulness: "hsl(var(--chart-1))",
  contextual_recall: "hsl(var(--chart-2))",
  answer_relevancy: "hsl(var(--chart-3))",
  contextual_precision: "hsl(var(--chart-4))",
  doc_recall: "hsl(var(--chart-5))",
};

export function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? key.replace(/_/g, " ");
}

export type Tone = "success" | "warning" | "destructive" | "muted";

/** Green at/above threshold, amber within 0.05 below, red otherwise. */
export function scoreTone(score: number | null | undefined, threshold: number | null | undefined): Tone {
  if (score === null || score === undefined) return "muted";
  if (threshold === null || threshold === undefined) return "muted";
  if (score >= threshold) return "success";
  if (score >= threshold - 0.05) return "warning";
  return "destructive";
}

export function deltaTone(delta: number | null | undefined, epsilon = 0.005): Tone {
  if (delta === null || delta === undefined) return "muted";
  if (delta > epsilon) return "success";
  if (delta < -epsilon) return "destructive";
  return "muted";
}

export function orderedMetricKeys(keys: Iterable<string>): string[] {
  const set = new Set(keys);
  const known = METRIC_KEYS.filter((k) => set.has(k));
  const rest = Array.from(set).filter((k) => !(METRIC_KEYS as readonly string[]).includes(k)).sort();
  return [...known, ...rest];
}
