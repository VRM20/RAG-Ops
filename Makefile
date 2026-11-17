# Developer shortcuts (run from WSL / Linux / macOS).  `make help` lists targets.
.RECIPEPREFIX = >
.DEFAULT_GOAL := help

VENV   ?= ragenv
PY     := $(VENV)/bin/python
RAGOPS := $(VENV)/bin/ragops

.PHONY: help install doctor pdfs ingest api worker web test lint fmt eval eval-quick synth baseline guardrail up down offline

help: ## Show this help
> @grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package + eval + dev extras into $(VENV)
> $(PY) -m pip install -e ".[eval,dev]"

doctor: ## Verify every credential and service
> $(RAGOPS) doctor

pdfs: ## Render the PDF documents of the demo corpus
> $(RAGOPS) build-pdfs

ingest: ## Ingest the demo corpus into Qdrant
> $(RAGOPS) ingest

api: ## Run the FastAPI server with auto-reload (http://localhost:8000/docs)
> $(VENV)/bin/uvicorn ragops.api.main:app --reload --port 8000

worker: ## Run a Celery worker (TASK_QUEUE=celery, needs Redis + Qdrant server)
> $(VENV)/bin/celery -A ragops.worker.celery_app worker -Q ingest,evals,celery --loglevel INFO

web: ## Run the Next.js dev server (http://localhost:3000)
> cd web && npm run dev

offline: ## Explore everything with zero credentials (API in offline mode)
> RAGOPS_OFFLINE=true RAGOPS_AUTO_INGEST=true $(VENV)/bin/uvicorn ragops.api.main:app --reload --port 8000

test: ## Offline unit + integration tests
> $(PY) -m pytest -q

lint: ## Ruff lint
> $(VENV)/bin/ruff check src tests evals scripts

fmt: ## Ruff autofix + format
> $(VENV)/bin/ruff check --fix src tests evals scripts && $(VENV)/bin/ruff format src tests evals scripts

eval: ## Full LLM-judged evaluation (costs money)
> $(RAGOPS) eval --report evals/reports/report.md --fail-on-gate

eval-quick: ## Evaluation on 10 seeded cases
> $(RAGOPS) eval --max-cases 10

synth: ## Generate the synthetic golden dataset with Gemini
> $(RAGOPS) synth

baseline: ## Re-run the evaluation and overwrite the committed baseline
> $(RAGOPS) eval --label baseline --save-baseline

guardrail: ## Create the AWS Bedrock Guardrail (one-off)
> $(RAGOPS) create-guardrail

up: ## Full stack in Docker (qdrant, redis, phoenix, api, worker, web)
> docker compose up --build

down: ## Stop the Docker stack
> docker compose down
