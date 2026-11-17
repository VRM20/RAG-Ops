from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ragops.api.deps import require_admin
from ragops.api.dispatch import dispatch_eval
from ragops.api.schemas import ScoreAnswerRequest, ScoreAnswerResponse, StartEvalRequest
from ragops.config import get_eval_gates, get_settings
from ragops.evals.models import EvalComparison, EvalRun, EvalRunSummary, GoldenCase
from ragops.evals.store import compare_runs, list_runs, load_baseline, load_run, save_run

router = APIRouter(prefix="/api/evals", tags=["evals"])


@router.get("/runs", response_model=list[EvalRunSummary])
async def runs() -> list[EvalRunSummary]:
    return list_runs()


@router.post("/runs", response_model=EvalRunSummary, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_admin)])
async def start_run(req: StartEvalRequest, background: BackgroundTasks) -> EvalRunSummary:
    from ragops.evals.runner import load_dataset, new_run_id

    cases = load_dataset(dataset=req.dataset, max_cases=req.max_cases)
    if not cases:
        raise HTTPException(400, f"No golden cases found for dataset '{req.dataset}'")
    placeholder = EvalRun(
        run_id=new_run_id(),
        label=req.label or ("sandbox" if req.overrides else "manual"),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        num_cases=len(cases),
    )
    save_run(placeholder)
    dispatch_eval(background, placeholder.run_id, placeholder.label, req.dataset, req.max_cases, req.overrides)
    return placeholder.summary()


@router.get("/runs/{run_id}", response_model=EvalRun)
async def get_run(run_id: str) -> EvalRun:
    run = load_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run


@router.get("/baseline", response_model=EvalRun)
async def baseline() -> EvalRun:
    run = load_baseline()
    if run is None:
        raise HTTPException(404, "No baseline committed yet (run `ragops eval --save-baseline`)")
    return run


@router.get("/compare", response_model=EvalComparison)
async def compare(head: str, base: str = "baseline") -> EvalComparison:
    base_run, head_run = load_run(base), load_run(head)
    if base_run is None or head_run is None:
        raise HTTPException(404, f"Run not found: {base if base_run is None else head}")
    return compare_runs(base_run, head_run)


@router.get("/dataset", response_model=list[GoldenCase])
async def dataset() -> list[GoldenCase]:
    from ragops.evals.runner import load_dataset

    return load_dataset()


@router.post("/score", response_model=ScoreAnswerResponse)
async def score(req: ScoreAnswerRequest) -> ScoreAnswerResponse:
    if get_settings().ragops_offline:
        raise HTTPException(503, "Live scoring needs a judge LLM; it is unavailable in offline mode.")
    from ragops.evals.judge import RouterJudge
    from ragops.evals.metrics import score_answer

    gates = get_eval_gates()
    judge = RouterJudge(gates.judge_route)
    thresholds = {n: g.threshold for n, g in gates.metrics.items()}
    scores = await score_answer(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        expected_answer=req.expected_answer,
        judge=judge,
        thresholds=thresholds,
    )
    return ScoreAnswerResponse(
        metrics={
            name: {
                "score": s.score,
                "reason": s.reason,
                "error": s.error,
                "threshold": thresholds.get(name),
                "passed": s.score is not None and s.score >= thresholds.get(name, 0),
            }
            for name, s in scores.items()
        },
        judge_model=judge.get_model_name(),
    )
