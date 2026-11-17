"""Input / output guardrails.

AWS Bedrock Guardrails is called through the standalone ``ApplyGuardrail`` API, so it
protects *any* model - including Gemini - not only Bedrock-hosted ones:

* **Input** (before retrieval): prompt-attack / jailbreak filter, denied topics, and
  PII handling (anonymise emails/phones/Aadhaar/PAN, block card numbers and secrets).
  A blocked query never reaches the retriever or the LLM.
* **Output** (after generation): PII leakage check plus *contextual grounding*: Bedrock
  scores whether the answer is grounded in the retrieved sources and relevant to the
  query. A low grounding score flags a likely hallucination at runtime, complementing
  the offline faithfulness metric used in CI.

Without ``BEDROCK_GUARDRAIL_ID`` (or in offline mode) a local regex guard provides the
same interface so the pipeline behaves consistently in development and tests.
Create the guardrail once with ``ragops create-guardrail``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from functools import lru_cache

from ragops.config import get_settings
from ragops.types import GuardrailVerdict

log = logging.getLogger(__name__)

BLOCKED_INPUT_MESSAGE = (
    "Your request was blocked by the security guardrail (possible prompt injection or "
    "sensitive data). Please rephrase without credentials or personal identifiers."
)

# ---- local fallback -----------------------------------------------------------------
# Generic identifiers plus locale-specific ones (India: PAN / Aadhaar, US: SSN).
_STATIC_PII: list[tuple[str, re.Pattern[str]]] = [
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b")),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # International +CC numbers, or Indian 10-digit mobiles.
    ("PHONE", re.compile(r"(?<![\w+])\+\d{1,3}[\s-]?\d(?:[\s-]?\d){6,12}(?!\d)|(?<!\d)[6-9][0-9]{9}(?!\d)")),
]


@lru_cache
def _pii_patterns() -> list[tuple[str, re.Pattern[str]]]:
    # Mailboxes of the organisation itself (support@, security@ ... quoted by documents)
    # are content, not personal data: GUARDRAIL_ALLOWED_EMAIL_DOMAINS exempts them.
    allowed = get_settings().allowed_email_domains
    exempt = rf"(?!(?:{'|'.join(re.escape(d) for d in allowed)})\b)" if allowed else ""
    email = re.compile(rf"\b[\w.+-]+@{exempt}[\w-]+\.[\w.-]+\b", re.IGNORECASE)
    return [("EMAIL", email), *_STATIC_PII]


_BLOCK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("CREDIT_DEBIT_CARD_NUMBER", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
]
_JAILBREAK_RE = re.compile(
    r"ignore (?:all |any |your |the |previous |prior )*(?:instructions|rules|guidelines|access rules)"
    r"|disregard (?:the |your )?(?:system|previous) (?:prompt|instructions)"
    r"|you are now (?:in )?(?:dan|developer mode|jailbroken)"
    r"|reveal (?:your|the) (?:system prompt|hidden instructions)"
    r"|pretend (?:you have|to have) no (?:restrictions|rules)",
    re.IGNORECASE,
)


def _local_input(text: str) -> GuardrailVerdict:
    reasons: list[str] = []
    if _JAILBREAK_RE.search(text):
        reasons.append("PROMPT_ATTACK")
    for name, pattern in _BLOCK_PATTERNS:
        if pattern.search(text):
            reasons.append(name)
    if reasons:
        return GuardrailVerdict(stage="input", provider="local", action="block", reasons=reasons)
    sanitized, found = _anonymize(text)
    if found:
        return GuardrailVerdict(stage="input", provider="local", action="anonymize", reasons=found, sanitized_text=sanitized)
    return GuardrailVerdict(stage="input", provider="local")


def _anonymize(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for name, pattern in _pii_patterns():
        if pattern.search(text):
            found.append(name)
            text = pattern.sub("{" + name + "}", text)
    return text, found


# ---- Bedrock ------------------------------------------------------------------------
def _parse_bedrock(resp: dict, stage: str) -> GuardrailVerdict:
    reasons: list[str] = []
    blocked = anonymized = False
    grounding = relevance = None
    grounding_failed = False
    for a in resp.get("assessments", []):
        for t in a.get("topicPolicy", {}).get("topics", []):
            if t.get("action") == "BLOCKED":
                blocked = True
                reasons.append(f"TOPIC:{t.get('name')}")
        for f in a.get("contentPolicy", {}).get("filters", []):
            if f.get("action") == "BLOCKED":
                blocked = True
                reasons.append(f"CONTENT:{f.get('type')}")
        for w in a.get("wordPolicy", {}).get("managedWordLists", []) + a.get("wordPolicy", {}).get("customWords", []):
            if w.get("action") == "BLOCKED":
                blocked = True
                reasons.append("WORD")
        sip = a.get("sensitiveInformationPolicy", {})
        for e in sip.get("piiEntities", []) + sip.get("regexes", []):
            label = e.get("type") or e.get("name")
            if e.get("action") == "BLOCKED":
                blocked = True
                reasons.append(f"PII:{label}")
            elif e.get("action") == "ANONYMIZED":
                anonymized = True
                reasons.append(f"PII:{label}")
        for f in a.get("contextualGroundingPolicy", {}).get("filters", []):
            if f.get("type") == "GROUNDING":
                grounding = f.get("score")
            elif f.get("type") == "RELEVANCE":
                relevance = f.get("score")
            if f.get("action") == "BLOCKED":
                grounding_failed = True
                reasons.append(f"{f.get('type')}<{f.get('threshold')}")

    outputs = resp.get("outputs") or []
    sanitized = outputs[0].get("text") if outputs else None
    if blocked:
        action = "block"
    elif anonymized:
        action = "anonymize"
    elif grounding_failed:
        action = "flag"
    else:
        action = "allow"
    return GuardrailVerdict(
        stage=stage,  # type: ignore[arg-type]
        provider="bedrock",
        action=action,  # type: ignore[arg-type]
        reasons=sorted(set(reasons)),
        sanitized_text=sanitized if action == "anonymize" else None,
        grounding_score=grounding,
        relevance_score=relevance,
    )


class Guardrails:
    def __init__(self) -> None:
        s = get_settings()
        self.use_bedrock = bool(s.bedrock_guardrail_id) and not s.ragops_offline
        self.guardrail_id = s.bedrock_guardrail_id
        self.version = s.bedrock_guardrail_version

    @property
    def provider(self) -> str:
        return "bedrock" if self.use_bedrock else "local"

    async def _apply(self, content: list[dict], source: str) -> dict:
        from ragops.providers.aws import bedrock_runtime

        return await asyncio.to_thread(
            bedrock_runtime().apply_guardrail,
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.version,
            source=source,
            content=content,
        )

    async def check_input(self, text: str) -> GuardrailVerdict:
        start = time.perf_counter()
        if self.use_bedrock:
            try:
                resp = await self._apply([{"text": {"text": text}}], "INPUT")
                verdict = _parse_bedrock(resp, "input")
            except Exception as exc:  # fail closed to the local guard, never open
                log.warning("Bedrock ApplyGuardrail failed (%s); using local guard", exc)
                verdict = _local_input(text)
        else:
            verdict = _local_input(text)
        verdict.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return verdict

    async def check_output(self, answer: str, query: str, sources: list[str]) -> GuardrailVerdict:
        start = time.perf_counter()
        if self.use_bedrock and sources:
            try:
                grounding_source = "\n\n".join(sources)[:90_000]
                resp = await self._apply(
                    [
                        {"text": {"text": grounding_source, "qualifiers": ["grounding_source"]}},
                        {"text": {"text": query, "qualifiers": ["query"]}},
                        {"text": {"text": answer, "qualifiers": ["guard_content"]}},
                    ],
                    "OUTPUT",
                )
                verdict = _parse_bedrock(resp, "output")
            except Exception as exc:
                log.warning("Bedrock output guardrail failed (%s); using local guard", exc)
                verdict = self._local_output(answer)
        else:
            verdict = self._local_output(answer)
        verdict.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return verdict

    @staticmethod
    def _local_output(answer: str) -> GuardrailVerdict:
        sanitized, found = _anonymize(answer)
        if found:
            return GuardrailVerdict(stage="output", provider="local", action="anonymize", reasons=found, sanitized_text=sanitized)
        return GuardrailVerdict(stage="output", provider="local")
