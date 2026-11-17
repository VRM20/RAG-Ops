"""LLM-as-judge adapter: DeepEval metrics -> the multi-provider router (route ``judge``).

DeepEval's metrics are model-agnostic: they call ``generate(prompt, schema)`` on a
``DeepEvalBaseLLM``. Routing that through :class:`LLMRouter` means the judge gets the
same retries, cross-provider fallbacks and OpenTelemetry traces as production calls.

The judge defaults to Gemini on Vertex AI while answers are synthesised by Claude on
Bedrock: using a different model family for judging reduces self-preference bias.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

from pydantic import BaseModel

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

from deepeval.models import DeepEvalBaseLLM  # noqa: E402

from ragops.providers.router import LLMRouter, get_router  # noqa: E402

_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _parse(text: str, schema: type[BaseModel]) -> BaseModel:
    try:
        return schema.model_validate_json(text)
    except Exception:
        match = _JSON_RE.search(text)
        if not match:
            raise
        return schema.model_validate(json.loads(match.group(0)))


class RouterJudge(DeepEvalBaseLLM):
    def __init__(self, route: str = "judge", router: LLMRouter | None = None, max_tokens: int = 8192):
        self.route = route
        self.router = router or get_router()
        self.max_tokens = max_tokens
        super().__init__(model=route)

    def load_model(self):
        return self.router

    def get_model_name(self) -> str:
        models = self.router.models_for(self.route)
        return models[0] if models else self.route

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: object):
        kwargs: dict = {}
        if schema is not None:
            kwargs["response_format"] = schema
        # Sampling params (temperature) come from config/models.yaml per model: Claude 5.x
        # rejects non-default temperature, so they must never be hard-coded here.
        result = await self.router.complete(
            self.route, [{"role": "user", "content": prompt}], max_tokens=self.max_tokens, **kwargs
        )
        return _parse(result.text, schema) if schema is not None else result.text

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: object):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.a_generate(prompt, schema))
        raise RuntimeError("RouterJudge.generate() called inside an event loop; use async metrics")
