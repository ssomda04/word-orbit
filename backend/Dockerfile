# syntax=docker/dockerfile:1
#
# Contextle backend image.
# Base: official uv image with Python 3.12 preinstalled (small Debian slim).
# We do NOT install the heavy `embeddings` extra here — the skeleton runs on the
# deterministic mock, keeping the image small and the build fast.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# --- Dependency layer (cached until pyproject.toml / uv.lock change) ---
# Only runtime deps; --no-dev skips pytest/ruff. `uv.lock` is optional: the glob
# copies it if it has been generated (`uv lock`), otherwise uv resolves fresh.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# --- Application layer ---
COPY app ./app

# --- Run as an unprivileged user ---
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness check hitting /health using only the stdlib (no extra packages).
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
