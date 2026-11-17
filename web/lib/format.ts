/** "p. 4" / "pp. 4-6" for PDF chunks, null when the source has no pages. */
export function pageLabel(start: number | null | undefined, end?: number | null): string | null {
  if (start === null || start === undefined) return null;
  return end === null || end === undefined || end === start ? `p. ${start}` : `pp. ${start}-${end}`;
}

export function fmtScore(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined || Number.isNaN(v) ? "-" : v.toFixed(digits);
}

export function fmtDelta(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : s;
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined) return "-";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "-";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  return `${m}m ${Math.round(seconds % 60)}s`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function fmtBytes(n: number | null | undefined): string {
  if (!n) return "-";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function shortSha(sha: string | null | undefined, n = 7): string {
  return sha ? sha.slice(0, n) : "-";
}

export function runName(run: { label: string; run_id: string }): string {
  return run.label || run.run_id.slice(0, 12);
}
