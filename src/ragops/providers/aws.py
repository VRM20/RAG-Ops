"""AWS Bedrock clients via boto3.

Credential resolution order:
1. Explicit ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` (+ optional session token)
2. ``AWS_PROFILE`` from ~/.aws/credentials
3. The default boto3 chain: IAM role on EC2/ECS/EKS, or GitHub Actions OIDC
   (``aws-actions/configure-aws-credentials``) - preferred in production and CI.

Required IAM permissions: ``bedrock:InvokeModel``, ``bedrock:InvokeModelWithResponseStream``,
``bedrock:ApplyGuardrail``, plus ``bedrock:CreateGuardrail`` /
``bedrock:CreateGuardrailVersion`` for the one-off guardrail setup command.
"""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.config import Config

from ragops.config import get_settings

_BOTO_CONFIG = Config(retries={"max_attempts": 4, "mode": "adaptive"}, connect_timeout=5, read_timeout=60)


@lru_cache
def get_session() -> boto3.Session:
    s = get_settings()
    if s.aws_access_key_id and s.aws_secret_access_key:
        return boto3.Session(
            aws_access_key_id=s.aws_access_key_id.get_secret_value(),
            aws_secret_access_key=s.aws_secret_access_key.get_secret_value(),
            aws_session_token=s.aws_session_token.get_secret_value() if s.aws_session_token else None,
            region_name=s.aws_region,
        )
    if s.aws_profile:
        return boto3.Session(profile_name=s.aws_profile, region_name=s.aws_region)
    return boto3.Session(region_name=s.aws_region)


@lru_cache
def bedrock_runtime():
    """Data plane: Converse / InvokeModel / ApplyGuardrail."""
    return get_session().client("bedrock-runtime", config=_BOTO_CONFIG)


@lru_cache
def bedrock_control():
    """Control plane: CreateGuardrail, ListFoundationModels, ..."""
    return get_session().client("bedrock", config=_BOTO_CONFIG)


def litellm_bedrock_params() -> dict:
    """Credentials block passed to every LiteLLM Bedrock call."""
    s = get_settings()
    params: dict = {"aws_region_name": s.aws_region}
    if s.aws_access_key_id and s.aws_secret_access_key:
        params["aws_access_key_id"] = s.aws_access_key_id.get_secret_value()
        params["aws_secret_access_key"] = s.aws_secret_access_key.get_secret_value()
        if s.aws_session_token:
            params["aws_session_token"] = s.aws_session_token.get_secret_value()
    elif s.aws_profile:
        params["aws_profile_name"] = s.aws_profile
    return params
