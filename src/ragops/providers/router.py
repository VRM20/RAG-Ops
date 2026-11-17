"""Multi-provider LLM router (GCP Vertex AI Gemini <-> AWS Bedrock Claude).

Application code asks for a *route* ("synthesis", "pdf_parsing", "judge", ...), never
for a provider. Each route from ``config/models.yaml`` becomes a LiteLLM Router model
group with ordered fallback groups:

    synthesis            -> claude-haiku-vertex  (Claude on Vertex AI)
    synthesis__fallback1 -> gemini-flash         (Gemini on Vertex AI)

(Illustrative chain; config/models.yaml currently runs every route on gemini-flash alone.)

LiteLLM retries transient errors, puts failing deployments in cooldown and walks the
fallback chain, so a throttling event or outage on one model degrades to the next
instead of failing the request. Providers without configured credentials are skipped
when the router is built, so the system runs with only one cloud configured.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ragops.config import ModelsConfig, ModelSpec, Settings, get_models_config, get_settings
from ragops.types import Usage

log = logging.getLogger(__name__)


class RouteUnavailableError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    route: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


def _usage_from(resp: Any) -> Usage:
    u = getattr(resp, "usage", None)
    if not u:
        return Usage()
    usage = Usage(
        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        total_tokens=getattr(u, "total_tokens", 0) or 0,
    )
    try:
        import litellm

        usage.cost_usd = round(float(litellm.completion_cost(completion_response=resp)), 6)
    except Exception:  # pricing unknown for very new model ids - not fatal
        usage.cost_usd = None
    return usage


class LLMRouter:
    def __init__(self, models_cfg: ModelsConfig, settings: Settings):
        self.cfg = models_cfg
        self.settings = settings
        self.offline = settings.ragops_offline
        self.route_models: dict[str, list[str]] = {}  # route -> resolved litellm model ids
        self._router = None if self.offline else self._build()

    # ---- construction --------------------------------------------------------------
    def _provider_available(self, spec: ModelSpec) -> bool:
        if spec.provider == "vertex":
            return self.settings.has_gcp
        if spec.provider == "bedrock":
            return self.settings.has_aws
        return True

    @staticmethod
    def _credentials(spec: ModelSpec) -> dict:
        if spec.provider == "vertex":
            from ragops.providers.gcp import litellm_vertex_params

            return litellm_vertex_params()
        if spec.provider == "bedrock":
            from ragops.providers.aws import litellm_bedrock_params

            return litellm_bedrock_params()
        return {}

    def _build(self):
        import litellm
        from litellm import Router

        litellm.drop_params = True  # silently drop params a provider does not support
        litellm.suppress_debug_info = True

        model_list: list[dict] = []
        fallbacks: list[dict] = []
        for route_name, route in self.cfg.routes.items():
            chain = [m for m in [route.primary, *route.fallbacks] if self._provider_available(self.cfg.models[m])]
            if not chain:
                log.warning("Route '%s' has no model with configured credentials; it is disabled", route_name)
                continue
            groups = []
            for i, model_name in enumerate(chain):
                spec = self.cfg.models[model_name]
                group = route_name if i == 0 else f"{route_name}__fallback{i}"
                model_list.append(
                    {
                        "model_name": group,
                        "litellm_params": {"model": spec.litellm_model, **spec.params, **self._credentials(spec)},
                        "model_info": {"id": f"{group}:{model_name}"},
                    }
                )
                groups.append(group)
            self.route_models[route_name] = [self.cfg.models[m].litellm_model for m in chain]
            if len(groups) > 1:
                fallbacks.append({groups[0]: groups[1:]})

        if not model_list:
            log.warning("No LLM provider credentials configured; all LLM routes are disabled")
            return None
        rs = self.cfg.router
        return Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=rs.num_retries,
            timeout=rs.timeout_s,
            allowed_fails=rs.allowed_fails,
            cooldown_time=rs.cooldown_s,
        )

    def models_for(self, route: str) -> list[str]:
        if self.offline:
            return ["offline/extractive"]
        return self.route_models.get(route, [])

    def _require(self, route: str):
        if self._router is None or route not in self.route_models:
            raise RouteUnavailableError(
                f"LLM route '{route}' is unavailable: configure Vertex AI and/or Bedrock credentials "
                "in .env (see docs/SETUP_CREDENTIALS.md) or set RAGOPS_OFFLINE=true."
            )
        return self._router

    # ---- calls ---------------------------------------------------------------------
    async def complete(
        self,
        route: str,
        messages: list[dict],
        max_tokens: int = 2048,
        span_name: str | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        start = time.perf_counter()
        if self.offline:
            return self._offline_complete(route, messages, start)
        router = self._require(route)
        resp = await router.acompletion(model=route, messages=messages, max_tokens=max_tokens, **kwargs)
        text = resp.choices[0].message.content or ""
        return LLMResult(
            text=text,
            route=route,
            model=getattr(resp, "model", route) or route,
            usage=_usage_from(resp),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    async def stream(
        self, route: str, messages: list[dict], max_tokens: int = 2048, **kwargs: Any
    ) -> AsyncIterator[str | LLMResult]:
        """Yield text deltas, then a final :class:`LLMResult` with the full text."""
        start = time.perf_counter()
        if self.offline:
            result = self._offline_complete(route, messages, start)
            words = result.text.split(" ")
            for i, word in enumerate(words):
                yield word if i == len(words) - 1 else word + " "
            yield result
            return
        router = self._require(route)
        resp = await router.acompletion(
            model=route,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        parts: list[str] = []
        model = route
        usage = Usage()
        async for chunk in resp:
            model = getattr(chunk, "model", None) or model
            if getattr(chunk, "usage", None):
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    yield delta
        yield LLMResult(
            text="".join(parts),
            route=route,
            model=model,
            usage=usage,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    def _offline_complete(self, route: str, messages: list[dict], start: float) -> LLMResult:
        from ragops.providers.offline import extractive_answer

        if route not in {"synthesis", "fast"}:
            raise RouteUnavailableError(f"Route '{route}' needs a real LLM; it is not available in offline mode.")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = extractive_answer(user if isinstance(user, str) else "")
        return LLMResult(
            text=text,
            route=route,
            model="offline/extractive",
            usage=Usage(prompt_tokens=len(str(user).split()), completion_tokens=len(text.split())),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )


@lru_cache
def get_router() -> LLMRouter:
    return LLMRouter(get_models_config(), get_settings())
