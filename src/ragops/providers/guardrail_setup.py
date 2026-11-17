"""One-off provisioning of the AWS Bedrock Guardrail used by the pipeline.

Run ``ragops create-guardrail`` once per AWS account/region, then put the printed
``BEDROCK_GUARDRAIL_ID`` / ``BEDROCK_GUARDRAIL_VERSION`` into ``.env``.

Policy design (per-direction actions, because input and output risks differ):

* Prompt attacks (jailbreak / injection): blocked on input, HIGH strength.
* Harmful content (hate, insults, sexual, violence, misconduct): blocked both ways.
* PII on input: emails/phones anonymised before they reach retrieval and the LLM;
  card numbers, CVVs, passwords and AWS keys blocked outright.
* PII on output: secrets and card data blocked; corporate emails and phone numbers
  that policies legitimately quote are allowed.
* India-specific identifiers (Aadhaar, PAN) have no built-in entity type, so they are
  covered by custom regexes.
* Denied topic: lookups of a specific person's personal or salary data.
* Contextual grounding: answers scoring below 0.75 grounding / 0.5 relevance against
  the retrieved sources are flagged as likely hallucinations.
"""

from __future__ import annotations

from ragops.providers.aws import bedrock_control

GUARDRAIL_NAME = "ragops-enterprise-rag"


def _pii(entity: str, inp: str, out: str) -> dict:
    return {"type": entity, "action": inp, "inputAction": inp, "outputAction": out, "inputEnabled": True, "outputEnabled": out != "NONE"}


def guardrail_definition(name: str = GUARDRAIL_NAME) -> dict:
    harmful = ["HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT"]
    return {
        "name": name,
        "description": "Input/output guardrail for the RAG-Ops enterprise knowledge assistant.",
        "blockedInputMessaging": "Your request was blocked by the security guardrail.",
        "blockedOutputsMessaging": "The response was blocked by the security guardrail.",
        "contentPolicyConfig": {
            "filtersConfig": [
                {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
                *[{"type": t, "inputStrength": "HIGH", "outputStrength": "HIGH"} for t in harmful],
            ]
        },
        "topicPolicyConfig": {
            "topicsConfig": [
                {
                    "name": "PersonalDataLookup",
                    "definition": (
                        "Requests to look up, reveal or compile personal information about a specific, "
                        "named employee or customer, such as their salary, rating, contact details, "
                        "account numbers or identity documents."
                    ),
                    "examples": [
                        "What is Priya Sharma's salary?",
                        "Give me the phone number and address of customer Rahul Verma.",
                        "Which employees got a rating of 1 this year?",
                    ],
                    "type": "DENY",
                }
            ]
        },
        "wordPolicyConfig": {"managedWordListsConfig": [{"type": "PROFANITY"}]},
        "sensitiveInformationPolicyConfig": {
            "piiEntitiesConfig": [
                _pii("EMAIL", "ANONYMIZE", "NONE"),
                _pii("PHONE", "ANONYMIZE", "NONE"),
                _pii("CREDIT_DEBIT_CARD_NUMBER", "BLOCK", "BLOCK"),
                _pii("CREDIT_DEBIT_CARD_CVV", "BLOCK", "BLOCK"),
                _pii("PASSWORD", "BLOCK", "BLOCK"),
                _pii("AWS_ACCESS_KEY", "BLOCK", "BLOCK"),
                _pii("AWS_SECRET_KEY", "BLOCK", "BLOCK"),
                _pii("INTERNATIONAL_BANK_ACCOUNT_NUMBER", "ANONYMIZE", "ANONYMIZE"),
                _pii("US_SOCIAL_SECURITY_NUMBER", "ANONYMIZE", "ANONYMIZE"),
            ],
            "regexesConfig": [
                {
                    "name": "INDIA_AADHAAR",
                    "description": "12-digit Aadhaar number, optionally space-separated in groups of 4",
                    "pattern": r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b",
                    "action": "ANONYMIZE",
                },
                {
                    "name": "INDIA_PAN",
                    "description": "Permanent Account Number (AAAAA9999A)",
                    "pattern": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
                    "action": "ANONYMIZE",
                },
            ],
        },
        "contextualGroundingPolicyConfig": {
            "filtersConfig": [
                {"type": "GROUNDING", "threshold": 0.75, "action": "BLOCK"},
                {"type": "RELEVANCE", "threshold": 0.5, "action": "BLOCK"},
            ]
        },
    }


def create_guardrail(name: str = GUARDRAIL_NAME) -> tuple[str, str]:
    """Create (or reuse) the guardrail and publish an immutable version."""
    client = bedrock_control()
    existing = next((g for g in client.list_guardrails().get("guardrails", []) if g["name"] == name), None)
    if existing:
        guardrail_id = existing["id"]
        client.update_guardrail(guardrailIdentifier=guardrail_id, **{k: v for k, v in guardrail_definition(name).items()})
    else:
        guardrail_id = client.create_guardrail(**guardrail_definition(name))["guardrailId"]
    version = client.create_guardrail_version(guardrailIdentifier=guardrail_id, description="ragops managed version")["version"]
    return guardrail_id, version
