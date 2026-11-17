# Credentials Setup

Everything you need to provide, where to get it, and where it goes. After each section,
run `ragops doctor`: it checks every credential with a real (tiny) API call.

```bash
cp .env.example .env      # then fill in the values below
mkdir -p secrets          # gitignored; holds the GCP key file
```

| # | What | Goes into | Used for |
| --- | --- | --- | --- |
| 1 | GCP service-account JSON key | `secrets/gcp-vertex-key.json` | Gemini (PDF parsing, reasoning, eval judge, synthetic data), embeddings |
| 2 | GCP project id | `.env` `GCP_PROJECT_ID` | Vertex AI |
| 3 | AWS access key pair | `.env` `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Claude on Bedrock (answer synthesis), Bedrock Guardrails |
| 4 | Bedrock Guardrail id + version | `.env` `BEDROCK_GUARDRAIL_ID`, `BEDROCK_GUARDRAIL_VERSION` | Created for you by `ragops create-guardrail` |
| 5 | Cohere API key | `.env` `COHERE_API_KEY` | Rerank 3.5 |
| 6 | (optional) Phoenix Cloud / Langfuse key | `.env` `OTEL_EXPORTER_OTLP_*` | Tracing |

> Want to look around before setting up any cloud account? Set `RAGOPS_OFFLINE=true`
> (or run `make offline`). Everything works with deterministic local stand-ins;
> only evaluation scores are meaningless in that mode.

---

## 1-2. Google Cloud: Vertex AI service account

1. **Project.** In <https://console.cloud.google.com>, select or create a project. Copy its
   **Project ID** (not the name) into `GCP_PROJECT_ID`.
2. **Billing + API.** Make sure billing is enabled, then enable the **Vertex AI API**:
   *APIs & Services > Library > "Vertex AI API" > Enable* (`aiplatform.googleapis.com`).
3. **Service account.** *IAM & Admin > Service Accounts > Create service account*
   - Name: `ragops-vertex`
   - Role: **Vertex AI User** (`roles/aiplatform.user`). That is all it needs.
4. **Key.** Open the service account, then *Keys > Add key > Create new key > JSON*. A file downloads.
   Save it as **`secrets/gcp-vertex-key.json`** in the repo (the folder is gitignored).
5. **Models.** In *Vertex AI > Model Garden*, open **Gemini 3.8 Flash** and accept the terms
   if prompted. It runs every route and is served from the `global` location
   (`VERTEX_LOCATION=global`, the default). Embeddings (`gemini-embedding-001`) use `us-central1`.
   - Optional, parked: **Claude Haiku 5.5** (by Anthropic) for answer synthesis. Partner models
     cannot be enabled on a free-trial billing account ("Choose Different Billing Account"):
     upgrade billing first, click **Enable**, check the model id on its card matches
     `claude-haiku-5-5` (otherwise set `VERTEX_CLAUDE_HAIKU_MODEL`), and request quota if needed
     (*IAM & Admin > Quotas*, filter "anthropic"). Then route `synthesis`/`fast` to it in
     `config/models.yaml`.

`.env`:
```ini
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=secrets/gcp-vertex-key.json
VERTEX_LOCATION=global
VERTEX_EMBEDDING_LOCATION=us-central1
```

> If your organisation blocks key creation (`iam.disableServiceAccountKeyCreation`), use a
> personal GCP project for this portfolio project, or switch CI to Workload Identity
> Federation (`google-github-actions/auth` with OIDC).

## 3. AWS: Bedrock in ap-south-1 (Mumbai)

1. **Model access.** In the AWS console, switch the region to **Asia Pacific (Mumbai) ap-south-1**,
   open *Amazon Bedrock > Model catalog*, and open **Claude Sonnet 5.5** and **Claude Haiku 5.5**.
   If prompted, submit Anthropic's one-time use-case form.
   From Mumbai the app calls Claude through **APAC cross-region inference profiles**
   (`apac.anthropic.claude-sonnet-5-5`, `apac.anthropic.claude-haiku-5-5`), which Bedrock routes
   to APAC regions with capacity.
2. **IAM user.** *IAM > Users > Create user* `ragops-app` (no console access). Attach this
   inline policy:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "InvokeClaude",
         "Effect": "Allow",
         "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
         "Resource": [
           "arn:aws:bedrock:*::foundation-model/anthropic.*",
           "arn:aws:bedrock:*:*:inference-profile/apac.anthropic.*",
           "arn:aws:bedrock:*:*:inference-profile/global.anthropic.*"
         ]
       },
       {
         "Sid": "Guardrails",
         "Effect": "Allow",
         "Action": [
           "bedrock:ApplyGuardrail",
           "bedrock:CreateGuardrail",
           "bedrock:UpdateGuardrail",
           "bedrock:CreateGuardrailVersion",
           "bedrock:GetGuardrail",
           "bedrock:ListGuardrails"
         ],
         "Resource": "*"
       }
     ]
   }
   ```
   Cross-region profiles need `InvokeModel` on the foundation-model ARNs in *every*
   destination region, hence the `*` region in the first resource.
3. **Access key.** *Security credentials > Create access key > "Application running outside AWS"*.

`.env`:
```ini
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

## 4. Bedrock Guardrail (generated, nothing to copy from the console)

```bash
ragops create-guardrail
```
This creates `ragops-enterprise-rag` (prompt-attack filter, PII anonymisation, India-specific
Aadhaar/PAN regexes, a denied topic and contextual-grounding checks; see
`src/ragops/providers/guardrail_setup.py`) and prints:
```ini
BEDROCK_GUARDRAIL_ID=abc123xyz
BEDROCK_GUARDRAIL_VERSION=1
```
Paste both into `.env`. Until then the pipeline uses the built-in local regex guard.

## 5. Cohere Rerank

Sign in at <https://dashboard.cohere.com/api-keys> and copy the trial key into `COHERE_API_KEY`.
Trial keys are rate-limited (about 10 rerank calls per minute). The reranker retries with
backoff, but a full eval run is faster with a production key or with `judge.concurrency: 2`
in `config/eval_gates.yaml`.

## 6. Tracing (optional)

* **Local Phoenix:** `docker compose up phoenix`, then `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:6006/v1/traces`.
  The UI is at <http://localhost:6006>.
* **Phoenix Cloud (free):** create a space at <https://app.phoenix.arize.com>, then
  `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://app.phoenix.arize.com/s/<space>/v1/traces` and
  `OTEL_EXPORTER_OTLP_HEADERS=api_key=<your key>`.
* **Langfuse Cloud:** `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces` and
  `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64 of public_key:secret_key>`.

---

## GitHub Actions secrets (for the eval gate)

*Repository > Settings > Secrets and variables > Actions*

| Type | Name | Value |
| --- | --- | --- |
| Secret | `GCP_SA_KEY` | the **entire contents** of `secrets/gcp-vertex-key.json` |
| Secret | `GCP_PROJECT_ID` | your project id |
| Secret | `AWS_ACCESS_KEY_ID` | as in `.env` |
| Secret | `AWS_SECRET_ACCESS_KEY` | as in `.env` |
| Secret | `COHERE_API_KEY` | as in `.env` |
| Secret (optional) | `OTEL_EXPORTER_OTLP_HEADERS` | tracing auth header |
| Variable | `AWS_REGION` | `ap-south-1` |
| Variable (optional) | `EVAL_MAX_CASES` | e.g. `30` to cap CI cost per PR |
| Variable (optional) | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | to trace CI runs |

### Make the gate block merges

*Settings > Branches > Add branch ruleset* (or classic protection rule) for `main`:
- Require a pull request before merging
- Require status checks to pass: **`RAG Eval Gate`**, **`Python lint + tests`**, **`Frontend build`**

The gate job is skipped (and counts as passing) on PRs that don't touch RAG behaviour
(for example README-only changes), so docs PRs are never blocked.

## Security checklist

- `.env`, `secrets/` and `*service-account*.json` are gitignored. Confirm with `git status`
  before every commit.
- Never paste keys into issues, PRs or the chat UI (the input guardrail blocks AWS keys, but don't rely on it).
- Rotate the AWS key and the GCP key if they are ever exposed. For production, replace static
  keys with IAM roles / GitHub OIDC (`aws-actions/configure-aws-credentials`, `google-github-actions/auth`).
