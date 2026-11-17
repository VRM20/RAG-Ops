from __future__ import annotations

import asyncio
import logging

from ragops.observability.tracing import setup_tracing
from ragops.types import PipelineOverrides
from ragops.worker.celery_app import celery_app

log = logging.getLogger(__name__)
setup_tracing("ragops-worker")


@celery_app.task(name="ragops.worker.tasks.ingest_document", bind=True, max_retries=2, default_retry_delay=30)
def ingest_document(self, doc_id: str) -> dict:
    from ragops.ingestion.pipeline import IngestionService

    rec = asyncio.run(IngestionService().process(doc_id))
    return {"doc_id": doc_id, "status": rec.status, "chunks": rec.num_chunks}


@celery_app.task(name="ragops.worker.tasks.run_eval")
def run_eval(run_id: str, label: str, dataset: str, max_cases: int | None, overrides: dict | None) -> dict:
    from ragops.evals.runner import load_dataset, run_evaluation

    run = asyncio.run(
        run_evaluation(
            run_id=run_id,
            label=label,
            cases=load_dataset(dataset=dataset, max_cases=max_cases),
            overrides=PipelineOverrides.model_validate(overrides) if overrides else None,
        )
    )
    return {"run_id": run.run_id, "status": run.status, "gate_passed": run.gate_passed}
