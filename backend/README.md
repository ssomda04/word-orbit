# Contextle Backend

FastAPI service for Contextle. This is the **collaboration skeleton**: `/health`,
a swappable embedding interface with a deterministic mock, and a dev-only
similarity endpoint. The real game API is not implemented yet (see
[../docs/ROADMAP.md](../docs/ROADMAP.md)).

## Requirements

- Python **3.12** (pinned in [`../.python-version`](../.python-version))
- [uv](https://docs.astral.sh/uv/) (preferred). Fallback: `python -m venv` + `pip`.

## Quick start (uv)

```bash
cd backend
uv sync                                   # install runtime + dev deps
uv run uvicorn app.main:app --reload      # http://localhost:8000
uv run ruff check .                       # lint
uv run pytest                             # tests
```

Open http://localhost:8000/docs for the interactive API docs, or:

```bash
curl http://localhost:8000/health         # -> {"status":"ok"}
```

### Fallback without uv (pip + venv)

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
pip install -e ".[embeddings]"   # or: pip install fastapi "uvicorn[standard]" pydantic pydantic-settings
pip install pytest httpx ruff
ruff check .
pytest
```

> `uv.lock` is committed once you run `uv lock`; until then `uv sync` resolves fresh.

## Docker

```bash
# from the repo root
docker compose up --build backend
```

The image runs as a non-root user and ships a `/health` `HEALTHCHECK`.

## Environment

Copy [`.env.example`](.env.example) to `.env` and adjust. Key variables:

| Variable             | Default                  | Purpose                                            |
| -------------------- | ------------------------ | -------------------------------------------------- |
| `APP_ENV`            | `development`            | Environment name.                                  |
| `FRONTEND_ORIGIN`    | `http://localhost:3000`  | CORS allow-list (comma-separated).                 |
| `EMBEDDING_PROVIDER` | `mock`                   | `mock` \| `deterministic` \| `sentence-transformers`. |
| `MODEL_NAME`         | *(empty)*                | Real model id (later phases).                      |
| `DATABASE_URL`       | *(empty)*                | Reserved (multiplayer/history).                    |
| `REDIS_URL`          | *(empty)*                | Reserved (multiplayer/history).                    |

## Layout

```
app/
├─ main.py            # create_app(): CORS, error handler, router wiring
├─ api/
│  ├─ deps.py         # dependency providers (DI seams for tests)
│  └─ routes/         # thin HTTP handlers (health, dev)
├─ core/config.py     # Settings (pydantic-settings)
├─ schemas/           # Pydantic wire models (camelCase JSON)
├─ services/embedding # EmbeddingService Protocol + deterministic mock + factory
└─ domain/            # (placeholder) pure game logic — no FastAPI, no model
```

## Conventions

- **Routers stay thin**; logic goes in `services/` or `domain/`.
- **Wire format is camelCase**, Python stays snake_case (`schemas/base.py`).
- The embedding model is accessed only through the `EmbeddingService` Protocol,
  so mock and real implementations are interchangeable.
- The answer word is never returned to the client or logged.

See [../AGENTS.md](../AGENTS.md) for the full backend rules.
