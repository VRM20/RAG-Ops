# API + Celery worker image (same image, different command).
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAGOPS_HOME=/app

WORKDIR /app

# Dependency layer first: only invalidated when pyproject.toml changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[eval]"

COPY config ./config
COPY data/corpus ./data/corpus
COPY evals ./evals

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data/state /app/secrets \
    && chown -R app:app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"

CMD ["uvicorn", "ragops.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
