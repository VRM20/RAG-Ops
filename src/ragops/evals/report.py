"""Markdown report for the PR comment and the GitHub Actions job summary."""

from __future__ import annotations

from ragops.evals.models import EvalRun
from ragops.evals.runner import METRIC_ORDER

COMMENT_MARKER = "<!-- ragops-eval-report -->"


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _delta(v: float | None) -> str:
    if v is None:
        return ""
    arrow = "▲" if v > 0.0005 else "▼" if v < -0.0005 else "■"
    return f"{arrow} {v:+.3f}"


def render_markdown(run: EvalRun, worst_n: int = 5) -> str:
    status = "✅ PASSED" if run.gate_passed else "❌ FAILED"
    c = run.config
    lines = [
        COMMENT_MARKER,
        f"## RAG Eval Gate: {status}",
        "",
        f"**{run.num_cases} cases** · {run.duration_s or 0:.0f}s · synthesis `{c.synthesis_model}` · "
        f"judge `{c.judge_model}` · prompt `{c.prompt_name}@{c.prompt_sha}` · index `{c.index_name}`",
        "",
        "| Metric | Mean | Threshold | Baseline | Δ vs baseline | Gate |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for name in METRIC_ORDER:
        m = run.metrics.get(name)
        if not m:
            continue
        gate = "✅" if m.passed and not m.regressed else ("❌" if m.blocking else "⚠️")
        kind = "**blocking**" if m.blocking else "advisory"
        lines.append(
            f"| `{name}` ({kind}) | {_fmt(m.mean)} | {m.threshold:.2f} | {_fmt(m.baseline)} | {_delta(m.delta)} | {gate} |"
        )

    acl = run.acl
    lines += [
        "",
        f"**ACL leakage:** {acl.leaked_chunks} forbidden chunk(s) across {acl.cases} permission cases "
        f"{'✅' if acl.passed else '❌'}",
    ]
    if run.gate_failures:
        lines += ["", "**Blocking failures**", *[f"- {f}" for f in run.gate_failures]]

    scored = [x for x in run.cases if "faithfulness" in x.metrics and x.metrics["faithfulness"].score is not None]
    worst = sorted(
        scored,
        key=lambda x: min(x.metrics["faithfulness"].score, (x.metrics.get("contextual_recall") or x.metrics["faithfulness"]).score or 0),
    )[:worst_n]
    if worst:
        lines += [
            "",
            f"<details><summary>Lowest-scoring {len(worst)} cases</summary>",
            "",
            "| Case | Question | Faithfulness | Ctx recall | Reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for x in worst:
            f = x.metrics["faithfulness"]
            r = x.metrics.get("contextual_recall")
            reason = (f.reason or "").replace("|", "\\|").replace("\n", " ")[:160]
            q = x.question.replace("|", "\\|")[:90]
            lines.append(f"| `{x.case_id}` | {q} | {_fmt(f.score)} | {_fmt(r.score if r else None)} | {reason} |")
        lines += ["", "</details>"]

    errored = [x for x in run.cases if x.error]
    if errored:
        lines += ["", f"⚠️ {len(errored)} case(s) errored, e.g. `{errored[0].case_id}`: {errored[0].error[:200]}"]
    lines += ["", f"<sub>run `{run.run_id}` · commit `{(c.git_sha or 'local')[:8]}`</sub>"]
    return "\n".join(lines) + "\n"
