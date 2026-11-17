"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fmtScore } from "@/lib/format";
import { METRIC_COLORS, metricLabel } from "@/lib/metrics";

const AXIS = { fontSize: 11, fill: "hsl(var(--muted-foreground))" };
const GRID = "hsl(var(--chart-grid))";
const TOOLTIP_STYLE: React.CSSProperties = {
  background: "hsl(var(--card))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 8,
  fontSize: 12,
  color: "hsl(var(--card-foreground))",
  boxShadow: "0 4px 16px rgb(0 0 0 / 0.08)",
};

function tooltipValue(v: unknown) {
  return typeof v === "number" ? fmtScore(v, 3) : String(v ?? "-");
}

export interface TrendPoint {
  name: string;
  [metric: string]: number | string | null;
}

export function TrendChart({
  data,
  metrics,
  thresholds,
  height = 260,
}: {
  data: TrendPoint[];
  metrics: string[];
  thresholds: Record<string, number>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} interval="preserveStartEnd" />
        <YAxis domain={[0, 1]} tick={AXIS} tickLine={false} axisLine={false} tickCount={6} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v, n) => [tooltipValue(v), metricLabel(String(n))]} />
        <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => metricLabel(String(v))} />
        {metrics.map((m) =>
          thresholds[m] !== undefined ? (
            <ReferenceLine
              key={`t-${m}`}
              y={thresholds[m]}
              stroke={METRIC_COLORS[m] ?? "hsl(var(--muted-foreground))"}
              strokeDasharray="4 4"
              strokeOpacity={0.55}
              ifOverflow="extendDomain"
            />
          ) : null,
        )}
        {metrics.map((m) => (
          <Line
            key={m}
            type="monotone"
            dataKey={m}
            stroke={METRIC_COLORS[m] ?? "hsl(var(--chart-1))"}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function Histogram({
  values,
  metric,
  threshold,
  height = 220,
}: {
  values: number[];
  metric: string;
  threshold?: number;
  height?: number;
}) {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    bin: `${(i / 10).toFixed(1)}`,
    lo: i / 10,
    count: 0,
  }));
  for (const v of values) {
    const idx = Math.min(9, Math.max(0, Math.floor(v * 10)));
    bins[idx].count += 1;
  }
  const thresholdBin = threshold !== undefined ? `${(Math.min(9, Math.floor(threshold * 10)) / 10).toFixed(1)}` : undefined;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={bins} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="bin" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis allowDecimals={false} tick={AXIS} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          cursor={{ fill: "hsl(var(--muted))" }}
          labelFormatter={(l) => `score ${l} - ${(Number(l) + 0.1).toFixed(1)}`}
          formatter={(v) => [String(v), "cases"]}
        />
        {thresholdBin && (
          <ReferenceLine x={thresholdBin} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" label={{ value: "gate", fontSize: 10, fill: "hsl(var(--muted-foreground))", position: "top" }} />
        )}
        <Bar dataKey="count" fill={METRIC_COLORS[metric] ?? "hsl(var(--chart-1))"} radius={[4, 4, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function GroupedBarChart({
  data,
  categoryKey,
  series,
  height = 260,
  domain = [0, 1],
}: {
  data: Record<string, number | string | null>[];
  categoryKey: string;
  series: { key: string; label: string; color: string }[];
  height?: number;
  domain?: [number, number];
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey={categoryKey} tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} interval={0} />
        <YAxis domain={domain} tick={AXIS} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "hsl(var(--muted))" }} formatter={(v, n) => [tooltipValue(v), String(n)]} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[4, 4, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
