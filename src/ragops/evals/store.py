"""Persistence and comparison of evaluation runs.

* Local / UI runs:   ``data/state/eval_runs/<run_id>.json``
* CI baseline:       ``evals/baselines/baseline.json`` (committed; updated deliberately
  with ``ragops eval --save-baseline`` when a PR legitimately moves the bar)
"""

from __future__ import annotations

from pathlib import Path

from ragops.config import EVALS_DIR, get_eval_gates, get_settings
from ragops.evals.models import CaseDelta, EvalComparison, EvalRun, EvalRunSummary, MetricDelta

BASELINE_PATH = EVALS_DIR / "baselines" / "baseline.json"
# A per-case drop this large on a blocking metric is reported as a case regression.
CASE_REGRESSION_DROP = 0.25


def runs_dir() -> Path:
    path = get_settings().state_dir / "eval_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(run: EvalRun, path: Path | None = None) -> Path:
    path = path or runs_dir() / f"{run.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run(run_id: str) -> EvalRun | None:
    if run_id == "baseline":
        return load_baseline()
    path = runs_dir() / f"{Path(run_id).name}.json"
    return EvalRun.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


def list_runs() -> list[EvalRunSummary]:
    runs = []
    for path in runs_dir().glob("*.json"):
        try:
            runs.append(EvalRun.model_validate_json(path.read_text(encoding="utf-8")).summary())
        except Exception:
            continue
    return sorted(runs, key=lambda r: r.created_at, reverse=True)


def load_baseline() -> EvalRun | None:
    if not BASELINE_PATH.exists():
        return None
    return EvalRun.model_validate_json(BASELINE_PATH.read_text(encoding="utf-8"))


def compare_runs(base: EvalRun, head: EvalRun) -> EvalComparison:
    names = sorted(set(base.metrics) | set(head.metrics))
    metric_deltas = {}
    for name in names:
        b = base.metrics.get(name).mean if name in base.metrics else None
        h = head.metrics.get(name).mean if name in head.metrics else None
        metric_deltas[name] = MetricDelta(base=b, head=h, delta=round(h - b, 4) if b is not None and h is not None else None)

    blocking = {n for n, g in get_eval_gates().metrics.items() if g.blocking}
    base_cases = {c.case_id: c for c in base.cases}
    cases, regressions, improvements = [], [], []
    for hc in head.cases:
        bc = base_cases.get(hc.case_id)
        if bc is None:
            continue
        b_scores = {n: s.score for n, s in bc.metrics.items()} | {"doc_recall": float(bc.doc_recall)}
        h_scores = {n: s.score for n, s in hc.metrics.items()} | {"doc_recall": float(hc.doc_recall)}
        delta = {
            n: round(h_scores[n] - b_scores[n], 4) if h_scores.get(n) is not None and b_scores.get(n) is not None else None
            for n in sorted(set(b_scores) | set(h_scores))
        }
        cases.append(
            CaseDelta(
                case_id=hc.case_id,
                question=hc.question,
                base=b_scores,
                head=h_scores,
                delta=delta,
                base_answer=bc.answer,
                head_answer=hc.answer,
            )
        )
        watched = blocking | {"doc_recall"}
        if any((delta.get(n) or 0) <= -CASE_REGRESSION_DROP for n in watched):
            regressions.append(hc.case_id)
        elif any((delta.get(n) or 0) >= CASE_REGRESSION_DROP for n in watched):
            improvements.append(hc.case_id)

    cases.sort(key=lambda c: min([v for v in c.delta.values() if v is not None] or [0]))
    return EvalComparison(
        base=base.summary(),
        head=head.summary(),
        metric_deltas=metric_deltas,
        cases=cases,
        regressions=regressions,
        improvements=improvements,
    )
