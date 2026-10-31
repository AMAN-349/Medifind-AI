FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
RUN chown appuser:appuser /app
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
COPY --chown=appuser:appuser src/ ./src/
USER appuser
RUN uv sync --locked --no-dev

COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser services/ ./services/
COPY --chown=appuser:appuser data/knowledge_base/ ./data/knowledge_base/
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml

CMD ["/app/.venv/bin/streamlit", "run", "app.py", "--server.headless=true"]