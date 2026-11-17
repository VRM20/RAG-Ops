from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ragops import __version__
from ragops.api.deps import get_ingestion, get_pipeline
from ragops.api.schemas import (
    ConfigResponse,
    HealthResponse,
    IndexStatus,
    PromptResponse,
    ProviderStatus,
    RouteInfo,
    UserResponse,
)
from ragops.config import get_eval_gates, get_models_config, get_rag_config, get_settings, get_users
from ragops.generation.prompts import list_prompts, load_prompt
from ragops.observability.tracing import tracing_enabled
from ragops.security.acl import all_known_groups

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    pipeline = get_pipeline()
    ingestion = get_ingestion()
    return HealthResponse(
        version=__version__,
        offline=s.ragops_offline,
        providers=ProviderStatus(
            vertex=s.has_gcp,
            bedrock=s.has_aws,
            cohere=s.cohere_api_key is not None,
            guardrail=pipeline.guardrails.provider,  # type: ignore[arg-type]
            qdrant="server" if s.qdrant_url else "embedded",
            task_queue=s.task_queue,
            tracing=tracing_enabled(),
        ),
        index=IndexStatus(
            name=ingestion.index_name,
            chunks=await ingestion.store.count(),
            documents=len(ingestion.registry.list(ingestion.index_name)),
        ),
    )


@router.get("/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    rag = get_rag_config()
    models = get_models_config()
    router_ = get_pipeline().router
    routes = {
        name: RouteInfo(
            primary=r.primary,
            fallbacks=r.fallbacks,
            models=router_.models_for(name) or [models.models[m].litellm_model for m in [r.primary, *r.fallbacks]],
        )
        for name, r in models.routes.items()
    }
    prompt = load_prompt(rag.generation.prompt)
    return ConfigResponse(
        rag=rag.model_dump(),
        routes=routes,
        gates=get_eval_gates().model_dump(),
        prompt={"name": prompt.name, "sha": prompt.sha},
    )


@router.get("/users", response_model=list[UserResponse])
async def users() -> list[UserResponse]:
    return [UserResponse(**u.model_dump()) for u in get_users().values()]


@router.get("/acl-groups", response_model=list[str])
async def acl_groups() -> list[str]:
    return all_known_groups()


@router.get("/prompts", response_model=list[str])
async def prompts() -> list[str]:
    return list_prompts()


@router.get("/prompts/{name}", response_model=PromptResponse)
async def prompt(name: str) -> PromptResponse:
    try:
        p = load_prompt(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Prompt '{name}' not found") from None
    return PromptResponse(name=p.name, description=p.description, system=p.system, user_template=p.user_template, sha=p.sha)
