# syntax=docker/dockerfile:1
#
# Contextle backend image.
# Base: official uv image with Python 3.12 preinstalled (small Debian slim).
# We do NOT install the heavy `embeddings` extra here — the deterministic mock
# needs no model, and artifact mode reads precomputed `.npy` files through numpy,
# which is already a base dependency. Neither mode loads an embedding model in
# this image, so it stays small and the build stays fast.
#
# The image is deployment-target agnostic: it binds whatever `PORT` it is given
# and selects its scoring provider from the environment. Nothing about Railway,
# an artifact root, or a frontend origin is baked in — see docs/DEPLOYMENT.md.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# --- Unprivileged user, created before anything is copied ---
# `COPY --chown=` resolves names against this image's /etc/passwd and /etc/group,
# so the user has to exist first — creating it after the COPYs would fail the
# build. The gid is declared explicitly rather than left to useradd's default,
# so `appuser:appuser` means one thing regardless of the base image's login.defs.
RUN groupadd --gid 10001 appuser \
 && useradd --create-home --uid 10001 --gid 10001 appuser

# --- Dependency layer (cached until pyproject.toml / uv.lock change) ---
# Only runtime deps; --no-dev skips pytest/ruff. `uv.lock` is optional: the glob
# copies it if it has been generated (`uv lock`), otherwise uv resolves fresh.
COPY --chown=appuser:appuser pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# --- Application layer ---
COPY --chown=appuser:appuser app ./app

# `uv sync` runs as root, so the environment it built is the one thing left that
# the runtime user does not own. Scoped to /app/.venv rather than a `chown -R`
# over all of /app: everything else arrived with its ownership already set, and a
# recursive chown rewrites metadata for every file it touches, which copies them
# into a new layer.
RUN chown -R appuser:appuser /app/.venv

USER appuser

# Documentation for `docker run -P` and for readers. Railway ignores EXPOSE and
# routes to whatever the process binds, which is why the port below is a variable.
EXPOSE 8000

# Default for local runs and Compose. A platform that injects PORT (Railway does)
# overrides this at runtime, and the CMD and HEALTHCHECK both read that value.
ENV PORT=8000

# Liveness check hitting /health using only the stdlib (no extra packages).
# Python reads PORT out of the environment itself, so this needs no shell
# expansion and stays correct when the platform picks the port.
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/health').status==200 else 1)"

# `sh -c` because an exec-form CMD has no shell, so ${PORT} would reach uvicorn
# as a literal string. `exec` because without it sh keeps PID 1 and a platform's
# SIGTERM never reaches uvicorn, turning every redeploy into a SIGKILL instead of
# a graceful shutdown.
#
# `--workers 1` is load-bearing, not decoration. uvicorn's --workers defaults to
# $WEB_CONCURRENCY, so without this flag a single environment variable silently
# forks the server — and game state lives in a process-wide in-memory repository
# (app/api/deps.py), so a second worker means games created on one are 404 on the
# other. A command-line value overrides the environment variable.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
