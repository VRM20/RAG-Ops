"""CI/CD RAG quality gate (DeepEval under pytest).

    pytest evals/test_rag_quality.py -m eval

Runs the full evaluation once per session (golden dataset -> RAG pipeline -> DeepEval
judge), writes the Markdown report and run JSON for the PR comment / artifacts, then
asserts the gates from config/eval_gates.yaml. Gates are asserted on dataset means and
against the committed baseline, not per case: single judged cases are noisy, and a
merge should be blocked by a systematic drop, not by one unlucky sample.

Requires real cloud credentials (it costs money). Not collected by the unit-test run.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from ragops.config import ROOT_DIR, get_eval_gates
from ragops.evals.models import EvalRun
from ragops.evals.report import render_markdown
from ragops.evals.runner import load_dataset, run_evaluation
from ragops.evals.store import save_run

pytestmark = pytest.mark.eval

REPORT_DIR = Path(os.environ.get("RAGOPS_REPORT_DIR", ROOT_DIR / "evals" / "reports"))


@pytest.fixture(scope="session")
def eval_run() -> EvalRun:
    max_cases = int(os.environ["EVAL_MAX_CASES"]) if os.environ.get("EVAL_MAX_CASES") else None
    cases = load_dataset(max_cases=max_cases)
    assert cases, "golden dataset is empty"
    run = asyncio.run(run_evaluation(label=os.environ.get("EVAL_LABEL", "ci"), cases=cases, persist=False))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.md").write_text(render_markdown(run), encoding="utf-8")
    save_run(run, REPORT_DIR / "run.json")
    assert run.status == "completed", run.error
    return run


def _blocking() -> list[str]:
    return [name for name, gate in get_eval_gates().metrics.items() if gate.blocking]


@pytest.mark.parametrize("metric", _blocking())
def test_blocking_metric_meets_threshold(eval_run: EvalRun, metric: str):
    m = eval_run.metrics[metric]
    assert m.mean is not None, f"{metric}: no scores were produced"
    assert m.passed, f"{metric} mean {m.mean:.3f} is below the gate {m.threshold}"


@pytest.mark.parametrize("metric", _blocking())
def test_no_regression_vs_baseline(eval_run: EvalRun, metric: str):
    m = eval_run.metrics[metric]
    if m.baseline is None:
        pytest.skip("no committed baseline yet")
    assert not m.regressed, f"{metric} regressed {m.delta:+.3f} vs baseline {m.baseline:.3f}"


def test_no_acl_leakage(eval_run: EvalRun):
    assert eval_run.acl.passed, f"{eval_run.acl.leaked_chunks} forbidden chunk(s) retrievable: {eval_run.acl.details}"


def test_error_rate_below_ten_percent(eval_run: EvalRun):
    errors = [c for c in eval_run.cases if c.error]
    assert len(errors) <= 0.1 * len(eval_run.cases), f"{len(errors)} cases errored, e.g. {errors[0].error}"


def test_advisory_metrics(eval_run: EvalRun):
    """Advisory gates never fail the build; they are surfaced as warnings in the report."""
    for name, m in eval_run.metrics.items():
        if not m.blocking and not m.passed:
            import warnings

            warnings.warn(f"advisory metric {name} = {m.mean} below {m.threshold}", stacklevel=1)
