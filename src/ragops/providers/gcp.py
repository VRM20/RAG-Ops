"""GCP Vertex AI authentication via a service-account JSON key.

The same key is used in three places:
1. ``google.oauth2.service_account.Credentials`` - validated and refreshed here, so a
   bad key fails fast at startup with a clear message instead of on the first request.
2. ``vertexai.init(...)`` - initialises the Vertex AI SDK (``google-cloud-aiplatform``)
   for any direct SDK usage.
3. LiteLLM - receives the key JSON as ``vertex_credentials`` for Gemini chat and
   embedding calls routed through :mod:`ragops.providers.router`.

Required IAM role on the service account: ``roles/aiplatform.user`` (Vertex AI User).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from ragops.config import get_settings

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class GCPConfigError(RuntimeError):
    pass


@lru_cache
def service_account_info() -> dict:
    settings = get_settings()
    path = settings.gcp_key_path
    if path is None:
        raise GCPConfigError("GOOGLE_APPLICATION_CREDENTIALS is not set (path to the service-account JSON key).")
    if not path.exists():
        raise GCPConfigError(f"Service-account key not found at {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("type") != "service_account":
        raise GCPConfigError(f"{path} is not a service-account key (type={info.get('type')!r})")
    return info


@lru_cache
def get_credentials():
    """Return refreshed ``google.oauth2.service_account.Credentials``."""
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(service_account_info(), scopes=SCOPES)
    creds.refresh(Request())  # fail fast on a revoked / malformed key
    return creds


@lru_cache
def init_vertex() -> None:
    """Initialise the Vertex AI SDK with the service-account credentials."""
    import vertexai

    settings = get_settings()
    if not settings.gcp_project_id:
        raise GCPConfigError("GCP_PROJECT_ID is not set.")
    vertexai.init(project=settings.gcp_project_id, location=settings.vertex_location, credentials=get_credentials())
    log.info("Vertex AI initialised (project=%s, location=%s)", settings.gcp_project_id, settings.vertex_location)


def litellm_vertex_params(location: str | None = None) -> dict:
    """Credentials block passed to every LiteLLM Vertex AI call."""
    settings = get_settings()
    return {
        "vertex_project": settings.gcp_project_id or service_account_info().get("project_id"),
        "vertex_location": location or settings.vertex_location,
        "vertex_credentials": json.dumps(service_account_info()),
    }
