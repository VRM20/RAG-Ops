"""Run heavy work (ingestion, eval runs) inline or on Celery workers.

``TASK_QUEUE=inline`` (default): FastAPI background tasks in the API process. No Redis
needed, and required for embedded Qdrant (which allows a single process only).
``TASK_QUEUE=celery``: enqueue on Redis; workers scale independently of the API
(docker compose, with Qdrant in server mode).
"""

from __future__ import annotations

import logging

from fastapi import BackgroundTasks

from ragops.config import get_settings
from ragops.types import PipelineOverrides

log = logging.getLogger(__name__)


async def _ingest_inline(doc_ids: list[str]) -> None:
    from ragops.api.deps import get_ingestion

    service = get_ingestion()
    for doc_id in doc_ids:
        await service.process(doc_id)


async def _eval_inline(run_id: str, label: str, dataset: str, max_cases: int | None, overrides: dict | None) -> None:
    from ragops.evals.runner import load_dataset, run_evaluation

    await run_evaluation(
        run_id=run_id,
        label=label,
        cases=load_dataset(dataset=dataset, max_cases=max_cases),
        overrides=PipelineOverrides.model_validate(overrides) if overrides else None,
    )


def dispatch_ingest(background: BackgroundTasks, doc_ids: list[str]) -> None:
    if get_settings().task_queue == "celery":
        from ragops.worker.tasks import ingest_document

        for doc_id in doc_ids:
            ingest_document.delay(doc_id)
    else:
        background.add_task(_ingest_inline, doc_ids)


def dispatch_eval(
    background: BackgroundTasks, run_id: str, label: str, dataset: str, max_cases: int | None, overrides: PipelineOverrides | None
) -> None:
    payload = overrides.model_dump(exclude_none=True) if overrides else None
    if get_settings().task_queue == "celery":
        from ragops.worker.tasks import run_eval

        run_eval.delay(run_id, label, dataset, max_cases, payload)
    else:
        background.add_task(_eval_inline, run_id, label, dataset, max_cases, payload)
