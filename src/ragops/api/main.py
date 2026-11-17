"""FastAPI application: ``uvicorn ragops.api.main:app``."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ragops import __version__
from ragops.api.routes import chat, documents, evals, system
from ragops.config import get_settings
from ragops.observability.tracing import setup_tracing
from ragops.providers.router import RouteUnavailableError
from ragops.security.acl import UnknownUserError

log = logging.getLogger("ragops")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    setup_tracing("ragops-api")
    from ragops.api.deps import get_ingestion

    svc = get_ingestion()
    await svc.store.ensure_collection()
    log.info("RAG-Ops API %s | offline=%s | index=%s", __version__, s.ragops_offline, svc.index_name)
    if os.environ.get("RAGOPS_AUTO_INGEST", "").lower() in {"1", "true", "yes"} and await svc.store.count() == 0:
        log.info("Index is empty: auto-ingesting the demo corpus")
        await svc.ingest_corpus()
    yield


app = FastAPI(
    title="RAG-Ops Enterprise Knowledge Engine",
    version=__version__,
    description="Hybrid-search RAG on Vertex AI + AWS Bedrock with CI/CD evaluation gates.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RouteUnavailableError)
async def _route_unavailable(_: Request, exc: RouteUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(UnknownUserError)
async def _unknown_user(_: Request, exc: UnknownUserError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"Unknown user '{exc}'"})


for r in (system.router, documents.router, chat.router, evals.router):
    app.include_router(r)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"service": "ragops", "version": __version__, "docs": "/docs"}
