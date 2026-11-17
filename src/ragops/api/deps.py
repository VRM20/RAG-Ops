"""Shared FastAPI dependencies and singletons."""

from __future__ import annotations

import hmac
from functools import lru_cache

from fastapi import Header, HTTPException, status

from ragops.config import get_settings
from ragops.generation.pipeline import RAGPipeline
from ragops.ingestion.pipeline import IngestionService
from ragops.security.acl import UnknownUserError, resolve_user
from ragops.types import UserContext


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@lru_cache
def get_ingestion() -> IngestionService:
    return IngestionService()


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Protect write endpoints when ADMIN_API_KEY is configured."""
    expected = get_settings().admin_api_key
    if expected is None:
        return
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-Admin-Key")


def user_context(user_id: str) -> UserContext:
    try:
        return resolve_user(user_id)
    except UnknownUserError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown user '{user_id}'") from None
