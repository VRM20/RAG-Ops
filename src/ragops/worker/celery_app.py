"""Celery application (``celery -A ragops.worker.celery_app worker``).

Heavy, bursty work runs here so the API stays responsive: PDF parsing (multimodal LLM
calls), bulk embedding, and full evaluation runs (hundreds of judge calls).
Requires Redis and Qdrant in *server* mode (embedded Qdrant is single-process).
"""

from __future__ import annotations

from celery import Celery

from ragops.config import get_settings

settings = get_settings()

celery_app = Celery("ragops", broker=settings.redis_url, backend=settings.redis_url, include=["ragops.worker.tasks"])
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,              # re-deliver if a worker dies mid-task
    worker_prefetch_multiplier=1,     # long tasks: don't hoard messages
    task_time_limit=60 * 60,
    task_soft_time_limit=55 * 60,
    task_routes={"ragops.worker.tasks.run_eval": {"queue": "evals"}, "ragops.worker.tasks.ingest_document": {"queue": "ingest"}},
    broker_connection_retry_on_startup=True,
)
