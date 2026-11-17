from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ragops.api.deps import get_pipeline, user_context
from ragops.api.schemas import ChatRequest, RetrieveResponse
from ragops.types import RAGResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=RAGResponse)
async def chat(req: ChatRequest) -> RAGResponse:
    return await get_pipeline().answer(req.query, user_context(req.user_id), req.overrides)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    user = user_context(req.user_id)  # resolve before streaming so unknown users get a 404

    async def events() -> AsyncIterator[str]:
        try:
            async for event, data in get_pipeline().stream(req.query, user, req.overrides):
                yield _sse(event, data)
        except Exception as exc:  # surface errors to the client as a terminal event
            log.exception("stream failed")
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: ChatRequest) -> RetrieveResponse:
    user = user_context(req.user_id)
    result = await get_pipeline().retriever.retrieve(req.query, user, req.overrides)
    return RetrieveResponse(
        query=req.query,
        user_id=user.user_id,
        groups=user.groups,
        contexts=result.contexts,
        candidates=result.candidates,
        timings_ms=result.timings_ms,
    )
