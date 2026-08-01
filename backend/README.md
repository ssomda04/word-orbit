# Contextle Backend

FastAPI service for Contextle: `/health`, a swappable embedding interface with a
deterministic mock, and the **single-player game API** (create game, submit
guess, read state). Multiplayer, persistence, and 3D coordinates are later
phases (see [../docs/ROADMAP.md](../docs/ROADMAP.md)).

## Endpoints

| Method + Path                       | Purpose                                     |
| ----------------------------------- | ------------------------------------------- |
| `GET /health`                       | Liveness.                                   |
| `POST /api/games`                   | Start a game (answer chosen server-side).   |
| `POST /api/games/{gameId}/guesses`  | Score a guess; duplicates are idempotent.   |
| `GET /api/games/{gameId}`           | Game state + guess history.                 |
| `POST /api/dev/similarity`          | Dev harness for raw provider similarity.    |

The contract is [../docs/API_SPEC.md](../docs/API_SPEC.md). Two rules the code
enforces and tests assert: **the answer word never leaves the server while a game
is `playing`** (not in a response, not in a log), and game state is **in-memory
only** — it does not survive a restart.

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

| Variable               | Default                  | Purpose                                            |
| ---------------------- | ------------------------ | -------------------------------------------------- |
| `APP_ENV`              | `development`            | Environment name.                                  |
| `FRONTEND_ORIGIN`      | `http://localhost:3000`  | CORS allow-list (comma-separated).                 |
| `EMBEDDING_PROVIDER`   | `mock`                   | `mock` \| `deterministic` \| `fasttext` \| `sentence-transformers`. |
| `FASTTEXT_MODEL_PATH`  | *(empty)*                | Absolute path to a local FastText `.bin`. Required only for `fasttext`. |
| `MODEL_NAME`           | *(empty)*                | Real model id (later phases).                      |
| `DATABASE_URL`         | *(empty)*                | Reserved (multiplayer/history).                    |
| `REDIS_URL`            | *(empty)*                | Reserved (multiplayer/history).                    |

### Running on the FastText baseline

The default is the mock — nothing below is needed for normal development, tests,
or CI. To score guesses with the real baseline model instead:

```powershell
# 1. Install the FastText-only extra (no torch, no sentence-transformers).
uv sync --extra fasttext

# 2. Point at a model you downloaded yourself. Nothing here downloads one, and
#    model files are never committed (.gitignore blocks *.bin).
$env:EMBEDDING_PROVIDER = "fasttext"
$env:FASTTEXT_MODEL_PATH = "C:/models/cc.ko.300.bin"

uv run --extra fasttext uvicorn app.main:app --reload
```

```bash
# macOS / Linux
uv sync --extra fasttext
export EMBEDDING_PROVIDER=fasttext
export FASTTEXT_MODEL_PATH=/opt/models/cc.ko.300.bin
uv run --extra fasttext uvicorn app.main:app --reload
```

Notes:

- The model is loaded **once, at startup** (`app.main` lifespan), so a bad path
  fails the server immediately instead of surfacing as a 500 on the first guess.
  Expect a pause of roughly 8 s while `cc.ko.300.bin` loads.
- Use an **absolute path**; a relative one resolves against the process working
  directory (`backend/` locally, `/app` in Docker). Prefer forward slashes and an
  ASCII-only path on Windows.
- Out-of-vocabulary guesses are fine — FastText composes character n-grams.
- `rank` and `coordinate` stay `null`: the contract is unchanged, and
  `project_3d` is not implemented for this provider (Phase 2).
- The Docker image does **not** install the extra; running FastText in a
  container needs a Dockerfile change plus a mounted model file.

### Optional FastText tests

`pytest` skips them unless the extra is installed **and** `FASTTEXT_MODEL_PATH`
points at a real file, so the default suite and CI stay model-free:

```powershell
uv sync --extra fasttext
$env:FASTTEXT_MODEL_PATH = "C:/models/cc.ko.300.bin"
uv run --extra fasttext pytest -m fasttext -v
```

## Layout

```
app/
├─ main.py            # create_app(): CORS, error handlers, router wiring
├─ api/
│  ├─ deps.py         # dependency providers (DI seams for tests)
│  └─ routes/         # thin HTTP handlers (health, games, dev)
├─ core/
│  ├─ config.py       # Settings (pydantic-settings)
│  └─ errors.py       # AppError hierarchy -> the standard error envelope
├─ schemas/           # Pydantic wire models (camelCase JSON)
├─ services/
│  ├─ embedding/      # EmbeddingService Protocol + mock + FastText + factory
│  └─ game/           # GameRepository Protocol + in-memory store + GameService
└─ domain/            # pure game logic — no FastAPI, no model
   ├─ game.py         # Game, Guess, GameStatus, word normalization
   └─ words.py        # placeholder answer words + AnswerSelector
```

Request flow: `routes/games.py` → `GameService` → `Game` rules, with the
`EmbeddingService` called only from the service layer. Storage, embeddings, and
answer selection are all injected via [`api/deps.py`](app/api/deps.py), so tests
substitute a fresh store and a pinned answer word.

## Conventions

- **Routers stay thin**; logic goes in `services/` or `domain/`.
- **Wire format is camelCase**, Python stays snake_case (`schemas/base.py`).
- The embedding model is accessed only through the `EmbeddingService` Protocol,
  so mock and real implementations are interchangeable.
- The answer word is never returned to the client or logged.

See [../AGENTS.md](../AGENTS.md) for the full backend rules.
