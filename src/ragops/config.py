"""Central configuration.

Two kinds of configuration, deliberately kept apart:

* **Secrets and infrastructure** (API keys, project IDs, endpoints) come from the
  environment / `.env` and are modelled by :class:`Settings`. They differ per machine
  and are never committed.
* **System behaviour** (chunking, retrieval depth, prompts, model routing, eval gates)
  lives in versioned YAML under ``config/``. Changing behaviour is a code change that
  goes through a PR and therefore through the evaluation gate.

YAML values may reference environment variables as ``${NAME:-default}``.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _root_dir() -> Path:
    """Project root holding config/, data/ and evals/.

    RAGOPS_HOME wins (Docker images); otherwise the source checkout this file lives in;
    otherwise the current directory (non-editable installs).
    """
    if home := os.environ.get("RAGOPS_HOME"):
        return Path(home).resolve()
    source_root = Path(__file__).resolve().parents[2]
    return source_root if (source_root / "config").is_dir() else Path.cwd()


ROOT_DIR = _root_dir()
CONFIG_DIR = ROOT_DIR / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"
DATA_DIR = ROOT_DIR / "data"
EVALS_DIR = ROOT_DIR / "evals"

# Export .env into os.environ so that YAML interpolation, LiteLLM (e.g. COHERE_API_KEY)
# and boto3 all see the same values. Real environment variables win over .env.
load_dotenv(ROOT_DIR / ".env", override=False)


# --------------------------------------------------------------------------------------
# Secrets & infrastructure (.env)
# --------------------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    ragops_env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"
    # Offline mode swaps every cloud dependency for a deterministic local stand-in
    # (hash embeddings, extractive "LLM", lexical reranker). Used by unit tests and
    # for exploring the UI before credentials are configured.
    ragops_offline: bool = False

    # ---- GCP / Vertex AI -------------------------------------------------------------
    gcp_project_id: str | None = None
    # Gemini 3.x (incl. previews) is served from the `global` location; embedding
    # models are regional, hence a separate location for them.
    vertex_location: str = "global"
    vertex_embedding_location: str = "us-central1"
    google_application_credentials: Path | None = None

    # ---- AWS / Bedrock ---------------------------------------------------------------
    aws_region: str = "ap-south-1"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    aws_session_token: SecretStr | None = None
    aws_profile: str | None = None
    bedrock_guardrail_id: str | None = None
    bedrock_guardrail_version: str = "DRAFT"

    # ---- Cohere (reranker) -----------------------------------------------------------
    cohere_api_key: SecretStr | None = None

    # ---- Qdrant ----------------------------------------------------------------------
    # Set QDRANT_URL to use a Qdrant server (Docker / Qdrant Cloud). Leave it empty to
    # use embedded on-disk mode at QDRANT_PATH (single process only).
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = DATA_DIR / "state" / "qdrant"
    qdrant_collection_prefix: str = "ragops"

    # ---- Background tasks --------------------------------------------------------------
    # inline = FastAPI background tasks in the API process (no Redis needed)
    # celery = Celery workers with a Redis broker (docker compose)
    task_queue: Literal["inline", "celery"] = "inline"
    redis_url: str = "redis://localhost:6379/0"

    # ---- State -----------------------------------------------------------------------
    state_dir: Path = DATA_DIR / "state"

    # ---- Observability (OpenTelemetry / OpenInference) ---------------------------------
    # Phoenix (local):  http://localhost:6006/v1/traces
    # Phoenix Cloud:    https://app.phoenix.arize.com/s/<space>/v1/traces  (+ api_key header)
    # Langfuse:         https://cloud.langfuse.com/api/public/otel/v1/traces (+ basic auth)
    otel_exporter_otlp_traces_endpoint: str | None = None
    otel_exporter_otlp_headers: str | None = None
    otel_service_name: str = "ragops"

    # ---- API ---------------------------------------------------------------------------
    cors_origins: str = "http://localhost:3000"
    # Optional shared secret required for write endpoints (upload/delete/eval runs).
    admin_api_key: SecretStr | None = None
    max_upload_mb: int = 100

    # ---- Guardrails --------------------------------------------------------------------
    # Email domains that are organisational content (e.g. support@ / security@ mailboxes
    # quoted by your documents) rather than personal data. Comma-separated.
    guardrail_allowed_email_domains: str = ""

    @property
    def allowed_email_domains(self) -> tuple[str, ...]:
        return tuple(d.strip().lower() for d in self.guardrail_allowed_email_domains.split(",") if d.strip())

    @field_validator(
        "gcp_project_id",
        "google_application_credentials",
        "aws_profile",
        "bedrock_guardrail_id",
        "qdrant_url",
        "otel_exporter_otlp_traces_endpoint",
        "otel_exporter_otlp_headers",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator(
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "cohere_api_key",
        "qdrant_api_key",
        "admin_api_key",
        mode="before",
    )
    @classmethod
    def _blank_secret_to_none(cls, v: Any) -> Any:
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def gcp_key_path(self) -> Path | None:
        p = self.google_application_credentials
        if p is None:
            return None
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def has_gcp(self) -> bool:
        return bool(self.gcp_project_id and self.gcp_key_path and self.gcp_key_path.exists())

    @property
    def has_aws(self) -> bool:
        return bool(
            (self.aws_access_key_id and self.aws_secret_access_key)
            or self.aws_profile
            or os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# --------------------------------------------------------------------------------------
# YAML behaviour config
# --------------------------------------------------------------------------------------
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return _interpolate(yaml.safe_load(fh) or {})


class ChunkingConfig(BaseModel):
    strategy: Literal["semantic", "structural", "fixed"] = "semantic"
    max_tokens: int = Field(350, ge=16, le=4000)
    min_tokens: int = Field(40, ge=0)
    overlap_tokens: int = Field(40, ge=0)
    semantic_breakpoint_percentile: float = Field(85, gt=0, lt=100)
    contextual_header: bool = True


class IngestionConfig(BaseModel):
    pdf_parser: Literal["gemini", "pypdf"] = "gemini"
    # Long PDFs are parsed in page batches: a 300-page document cannot be converted in
    # one call (model output limit), and small batches keep page attribution accurate.
    pdf_pages_per_batch: int = Field(10, ge=1, le=50)
    pdf_parse_concurrency: int = Field(4, ge=1, le=16)
    chunking: ChunkingConfig = ChunkingConfig()


class EmbeddingConfig(BaseModel):
    model: str
    dimensions: int = 768
    batch_size: int = 32


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "cohere/rerank-v3.5"
    top_n: int = 5


class RetrievalConfig(BaseModel):
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    fused_top_k: int = 20
    rerank: RerankConfig = RerankConfig()
    min_relevance_score: float = 0.0


class GenerationConfig(BaseModel):
    route: str = "synthesis"
    prompt: str = "rag_system_v1"
    max_tokens: int = 1024
    max_context_chunks: int = 5


class GuardrailsConfig(BaseModel):
    input: bool = True
    output: bool = True


class RagConfig(BaseModel):
    version: int = 1
    ingestion: IngestionConfig = IngestionConfig()
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()


class ModelSpec(BaseModel):
    litellm_model: str
    provider: Literal["vertex", "bedrock", "other"]
    params: dict[str, Any] = Field(default_factory=dict)


class RouteSpec(BaseModel):
    primary: str
    fallbacks: list[str] = Field(default_factory=list)


class RouterSettings(BaseModel):
    num_retries: int = 2
    timeout_s: float = 90
    allowed_fails: int = 3
    cooldown_s: float = 30


class ModelsConfig(BaseModel):
    models: dict[str, ModelSpec]
    routes: dict[str, RouteSpec]
    router: RouterSettings = RouterSettings()

    def model_post_init(self, __context: Any) -> None:
        for route_name, route in self.routes.items():
            for name in [route.primary, *route.fallbacks]:
                if name not in self.models:
                    raise ValueError(f"route '{route_name}' references unknown model '{name}'")


class MetricGate(BaseModel):
    threshold: float
    blocking: bool = False


class EvalGatesConfig(BaseModel):
    metrics: dict[str, MetricGate]
    regression_max_drop: float = 0.05
    acl_max_leaked_chunks: int = 0
    judge_route: str = "judge"
    judge_concurrency: int = 4
    dataset_files: list[str] = Field(default_factory=list)
    max_cases: int | None = None
    seed: int = 7

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> EvalGatesConfig:
        return cls(
            metrics=data.get("metrics", {}),
            regression_max_drop=data.get("regression", {}).get("max_drop", 0.05),
            acl_max_leaked_chunks=data.get("acl", {}).get("max_leaked_chunks", 0),
            judge_route=data.get("judge", {}).get("route", "judge"),
            judge_concurrency=data.get("judge", {}).get("concurrency", 4),
            dataset_files=data.get("dataset", {}).get("files", []),
            max_cases=data.get("dataset", {}).get("max_cases"),
            seed=data.get("dataset", {}).get("seed", 7),
        )


class UserProfile(BaseModel):
    user_id: str
    name: str
    title: str = ""
    groups: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Cached accessors
# --------------------------------------------------------------------------------------
@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_rag_config() -> RagConfig:
    return RagConfig.model_validate(load_yaml(CONFIG_DIR / "rag.yaml"))


@lru_cache
def get_models_config() -> ModelsConfig:
    return ModelsConfig.model_validate(load_yaml(CONFIG_DIR / "models.yaml"))


@lru_cache
def get_eval_gates() -> EvalGatesConfig:
    return EvalGatesConfig.from_yaml(load_yaml(CONFIG_DIR / "eval_gates.yaml"))


@lru_cache
def get_users() -> dict[str, UserProfile]:
    data = load_yaml(CONFIG_DIR / "users.yaml")
    return {u["user_id"]: UserProfile.model_validate(u) for u in data.get("users", [])}


def reset_config_caches() -> None:
    """Clear cached config (used by tests that patch the environment)."""
    for fn in (get_settings, get_rag_config, get_models_config, get_eval_gates, get_users):
        fn.cache_clear()
