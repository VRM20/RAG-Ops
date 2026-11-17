"""Evaluation runner: golden dataset -> RAG pipeline -> DeepEval scores -> gates.

Used by three entry points with identical semantics:
* CI quality gate      (``evals/test_rag_quality.py`` under pytest)
* CLI                  (``ragops eval``)
* UI / prompt sandbox  (``POST /api/evals/runs``, optionally with overrides)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from ragops.config import ROOT_DIR, EvalGatesConfig, get_eval_gates, get_rag_config, get_settings
from ragops.evals.acl import run_acl_checks
from ragops.evals.judge import RouterJudge
from ragops.evals.metrics import LLM_METRICS, doc_recall, score_answer
from ragops.evals.models import (
    CaseContext,
    EvalCaseResult,
    EvalRun,
    GoldenCase,
    MetricSummary,
    RunConfig,
)
from ragops.evals.store import load_baseline, save_run
from ragops.generation.pipeline import RAGPipeline
from ragops.retrieval.store import collection_name, effective_embedding_model
from ragops.security.acl import resolve_user
from ragops.types import PipelineOverrides

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


# ---- dataset ---------------------------------------------------------------------------
def load_dataset(
    gates: EvalGatesConfig | None = None, dataset: str = "all", max_cases: int | None = None
) -> list[GoldenCase]:
    gates = gates or get_eval_gates()
    cases: list[GoldenCase] = []
    for rel in gates.dataset_files:
        path = ROOT_DIR / rel
        if not path.exists():
            continue
        origin = "synthetic" if "synthetic" in path.name else "curated"
        if dataset != "all" and dataset != origin:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(GoldenCase.model_validate({"origin": origin, **json.loads(line)}))
    limit = max_cases or gates.max_cases
    if limit and len(cases) > limit:
        cases = sorted(random.Random(gates.seed).sample(cases, limit), key=lambda c: c.id)
    return cases


# ---- provenance -----------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def run_config(pipeline: RAGPipeline, judge: RouterJudge, overrides: PipelineOverrides | None) -> RunConfig:
    cfg = pipeline.cfg
    o = overrides or PipelineOverrides()
    prompt = pipeline.prompt(o)
    route = o.route or cfg.generation.route
    rerank_on = cfg.retrieval.rerank.enabled if o.rerank_enabled is None else o.rerank_enabled
    return RunConfig(
        index_name=collection_name(cfg),
        embedding_model=effective_embedding_model(cfg),
        chunking=cfg.ingestion.chunking.model_dump(),
        prompt_name=prompt.name,
        prompt_sha=prompt.sha,
        synthesis_route=route,
        synthesis_model=(pipeline.router.models_for(route) or ["unavailable"])[0],
        judge_model=judge.get_model_name(),
        rerank_model=pipeline.retriever.reranker.model if rerank_on else None,
        git_sha=os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD"),
        git_branch=os.environ.get("GITHUB_HEAD_REF") or _git("rev-parse", "--abbrev-ref", "HEAD"),
        offline=get_settings().ragops_offline,
        overrides=overrides,
    )


# ---- gates ----------------------------------------------------------------------------
def summarize(run: EvalRun, gates: EvalGatesConfig, baseline: EvalRun | None) -> None:
    failures: list[str] = []
    for name, gate in gates.metrics.items():
        if name == "doc_recall":
            values = [1.0 if c.doc_recall else 0.0 for c in run.cases if not c.error]
        else:
            values = [c.metrics[name].score for c in run.cases if name in c.metrics and c.metrics[name].score is not None]
        mean = round(statistics.fmean(values), 4) if values else None
        base = baseline.metrics.get(name).mean if baseline and name in baseline.metrics else None
        delta = round(mean - base, 4) if mean is not None and base is not None else None
        passed = mean is not None and mean >= gate.threshold
        regressed = delta is not None and delta < -gates.regression_max_drop
        run.metrics[name] = MetricSummary(
            mean=mean, threshold=gate.threshold, blocking=gate.blocking, passed=passed,
            n=len(values), baseline=base, delta=delta, regressed=regressed,
        )
        if gate.blocking and not passed:
            failures.append(f"{name} {mean if mean is not None else 'n/a'} < {gate.threshold}")
        if gate.blocking and regressed:
            failures.append(f"{name} regressed {delta:+.3f} vs baseline (max drop {gates.regression_max_drop})")
    if not run.acl.passed:
        failures.append(f"ACL leakage: {run.acl.leaked_chunks} forbidden chunk(s) retrievable")
    errors = sum(1 for c in run.cases if c.error)
    if run.cases and errors / len(run.cases) > 0.1:
        failures.append(f"{errors}/{len(run.cases)} cases errored")
    if run.cases and run.rerank_fallbacks / len(run.cases) > 0.1:
        failures.append(f"reranker failed for {run.rerank_fallbacks} case(s); scores do not reflect the configured pipeline")
    run.gate_failures = failures
    run.gate_passed = not failures


# ---- execution ------------------------------------------------------------------------
async def _evaluate_case(
    case: GoldenCase,
    pipeline: RAGPipeline,
    judge: RouterJudge,
    overrides: PipelineOverrides,
    thresholds: dict[str, float],
) -> EvalCaseResult:
    result = EvalCaseResult(
        case_id=case.id,
        question=case.question,
        expected_answer=case.expected_answer,
        question_type=case.question_type,
        origin=case.origin,
        source_doc_ids=case.source_doc_ids,
    )
    try:
        start = time.perf_counter()
        resp = await pipeline.answer(case.question, resolve_user(case.user_id), overrides)
        result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        result.answer = resp.answer
        result.contexts = [
            CaseContext(chunk_id=c.chunk.chunk_id, doc_id=c.chunk.doc_id, section=c.chunk.section, text=c.chunk.text, rerank_score=c.rerank_score)
            for c in resp.contexts
        ]
        result.retrieved_doc_ids = sorted({c.doc_id for c in result.contexts})
        result.doc_recall = doc_recall(case.source_doc_ids, result.retrieved_doc_ids)
        if not get_settings().ragops_offline:
            result.metrics = await score_answer(
                question=case.question,
                answer=resp.answer,
                contexts=[c.text for c in result.contexts],
                expected_answer=case.expected_answer,
                judge=judge,
                thresholds=thresholds,
            )
    except Exception as exc:
        log.exception("Case %s failed", case.id)
        result.error = f"{type(exc).__name__}: {exc}"[:500]
    return result


async def run_evaluation(
    *,
    label: str = "",
    cases: list[GoldenCase] | None = None,
    overrides: PipelineOverrides | None = None,
    run_id: str | None = None,
    include_acl: bool = True,
    persist: bool = True,
    on_progress: ProgressFn | None = None,
) -> EvalRun:
    gates = get_eval_gates()
    cases = cases if cases is not None else load_dataset(gates)
    # Eval measures the RAG core; guardrails are tested separately (deterministically).
    o = (overrides or PipelineOverrides()).model_copy()
    if o.guardrails_enabled is None:
        o.guardrails_enabled = False

    pipeline = RAGPipeline(get_rag_config())
    judge = RouterJudge(gates.judge_route)
    run = EvalRun(
        run_id=run_id or new_run_id(),
        label=label,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        num_cases=len(cases),
        config=run_config(pipeline, judge, overrides),
    )
    if persist:
        save_run(run)

    started = time.perf_counter()
    try:
        thresholds = {n: g.threshold for n, g in gates.metrics.items()}
        sem = asyncio.Semaphore(gates.judge_concurrency)

        async def bounded(case: GoldenCase) -> EvalCaseResult:
            async with sem:
                res = await _evaluate_case(case, pipeline, judge, o, thresholds)
            # Results are appended as they finish and persisted periodically, so the UI
            # can show live progress (cases done / num_cases) while the run is going.
            run.cases.append(res)
            if on_progress:
                on_progress(len(run.cases), len(cases))
            if persist and len(run.cases) % 5 == 0:
                save_run(run)
            return res

        await asyncio.gather(*(bounded(c) for c in cases))
        run.cases.sort(key=lambda c: c.case_id)
        run.rerank_fallbacks = pipeline.retriever.reranker.fallback_count
        if include_acl:
            run.acl = await run_acl_checks(pipeline.retriever, gates.acl_max_leaked_chunks)
        summarize(run, gates, load_baseline())
        run.status = "completed"
    except Exception as exc:
        log.exception("Evaluation run failed")
        run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
    run.completed_at = datetime.now(UTC).isoformat(timespec="seconds")
    run.duration_s = round(time.perf_counter() - started, 1)
    if persist:
        save_run(run)
    return run


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


# Metric keys surfaced to the UI in a stable order.
METRIC_ORDER = [*LLM_METRICS, "doc_recall"]
