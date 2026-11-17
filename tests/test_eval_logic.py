"""Gate logic, regression detection, run comparison and the PR report (no LLM calls)."""

from ragops.config import get_eval_gates
from ragops.evals.models import AclSummary, EvalCaseResult, EvalRun, MetricScore
from ragops.evals.report import COMMENT_MARKER, render_markdown
from ragops.evals.runner import load_dataset, summarize
from ragops.evals.store import compare_runs


def _case(cid: str, faith: float, recall: float, doc_ok: bool = True) -> EvalCaseResult:
    return EvalCaseResult(
        case_id=cid,
        question=f"question {cid}",
        expected_answer="x",
        question_type="factual",
        answer="a",
        doc_recall=doc_ok,
        metrics={
            "faithfulness": MetricScore(score=faith, reason="ok"),
            "contextual_recall": MetricScore(score=recall),
            "answer_relevancy": MetricScore(score=0.9),
            "contextual_precision": MetricScore(score=0.8),
        },
    )


def _run(run_id: str, cases: list[EvalCaseResult], baseline: EvalRun | None = None) -> EvalRun:
    run = EvalRun(run_id=run_id, created_at="2026-11-09T00:00:00+00:00", num_cases=len(cases), cases=cases, status="completed")
    run.acl = AclSummary(cases=6, leaked_chunks=0, passed=True)
    summarize(run, get_eval_gates(), baseline)
    return run


def test_gates_pass_when_blocking_metrics_clear_thresholds():
    run = _run("r1", [_case("a", 0.95, 0.9), _case("b", 0.9, 1.0)])
    assert run.gate_passed, run.gate_failures
    assert run.metrics["faithfulness"].mean == 0.925
    assert run.metrics["doc_recall"].mean == 1.0


def test_gates_fail_below_threshold():
    run = _run("r2", [_case("a", 0.6, 0.9), _case("b", 0.7, 0.9)])
    assert not run.gate_passed
    assert any(f.startswith("faithfulness") for f in run.gate_failures)


def test_regression_vs_baseline_fails_even_above_threshold():
    baseline = _run("base", [_case("a", 1.0, 1.0), _case("b", 1.0, 1.0)])
    head = _run("head", [_case("a", 0.9, 1.0), _case("b", 0.88, 1.0)], baseline=baseline)
    m = head.metrics["faithfulness"]
    assert m.passed and m.regressed and m.delta == -0.11
    assert not head.gate_passed and any("regressed" in f for f in head.gate_failures)


def test_acl_leak_fails_the_gate():
    run = _run("r3", [_case("a", 1.0, 1.0)])
    run.acl = AclSummary(cases=6, leaked_chunks=2, passed=False)
    summarize(run, get_eval_gates(), None)
    assert not run.gate_passed and any("ACL" in f for f in run.gate_failures)


def test_compare_flags_case_regressions():
    base = _run("base", [_case("a", 1.0, 1.0), _case("b", 1.0, 1.0)])
    head = _run("head", [_case("a", 0.5, 1.0), _case("b", 1.0, 1.0, doc_ok=True)])
    cmp = compare_runs(base, head)
    assert cmp.regressions == ["a"]
    assert cmp.cases[0].case_id == "a"  # worst first
    assert cmp.metric_deltas["faithfulness"].delta == -0.25


def test_markdown_report_contains_gate_table():
    md = render_markdown(_run("r4", [_case("a", 0.95, 0.9)]))
    assert md.startswith(COMMENT_MARKER)
    assert "RAG Eval Gate: ✅ PASSED" in md
    assert "`faithfulness` (**blocking**)" in md and "ACL leakage" in md


def test_curated_dataset_is_well_formed():
    cases = load_dataset(dataset="curated")
    assert len(cases) >= 25
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.source_doc_ids and c.expected_answer for c in cases)
    sample = load_dataset(dataset="curated", max_cases=5)
    assert len(sample) == 5 and sample == load_dataset(dataset="curated", max_cases=5)  # seeded
