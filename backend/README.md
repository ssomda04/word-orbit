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

The image runs as a non-root user and ships a `/health` `HEALTHCHECK`. It binds
`$PORT` (default `8000`) and always runs a **single** uvicorn worker, because game
state is in-memory and per-process. Deploying it — the platform contract, the
environment variables, and the variables that must never be set — is
[../docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md).

## Environment

Copy [`.env.example`](.env.example) to `.env` and adjust. Key variables:

| Variable               | Default                  | Purpose                                            |
| ---------------------- | ------------------------ | -------------------------------------------------- |
| `APP_ENV`              | `development`            | Environment name.                                  |
| `FRONTEND_ORIGIN`      | `http://localhost:3000`  | CORS allow-list (comma-separated).                 |
| `EMBEDDING_PROVIDER`   | `mock`                   | `mock` \| `deterministic` \| `fasttext` \| `sentence-transformers`. |
| `FASTTEXT_MODEL_PATH`  | *(empty)*                | Absolute path to a local FastText `.bin`. Required only for `fasttext`. |
| `MODEL_NAME`           | *(empty)*                | Real model id (later phases).                      |
| `VOCABULARY_PATH`      | *(empty)*                | Word list that guess ranks are computed against. Empty ⇒ `rank` is always null. |
| `RANK_CACHE_SIZE`      | `32`                     | Answers whose similarity array stays in memory.    |
| `SCORING_PROVIDER`     | `embedding`              | `embedding` \| `artifact`. Where `similarity` and `rank` come from. |
| `ARTIFACT_ROOT`        | *(empty)*                | Absolute path to a precomputed artifact root. Required only for `artifact`. |
| `ARTIFACT_CACHE_SIZE`  | `64`                     | Answers whose artifact arrays stay in memory. Positive integer. |
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
- `coordinate` stays `null`: `project_3d` is not implemented for this provider
  (Phase 2). For `rank`, see **Guess ranking** below.
- The Docker image does **not** install the extra; running FastText in a
  container needs a Dockerfile change plus a mounted model file.

### Guess ranking

`rank` answers "out of every word we know, where does this guess sit?" — 1 is
the answer itself, 2 is the closest other word. It is computed against a
vocabulary file, so it is `null` until you configure one:

```powershell
$env:VOCABULARY_PATH = "C:/models/game_words.txt"
uv run uvicorn app.main:app --reload
```

The file is UTF-8 (a BOM is tolerated), one word per line. Generate one with
[`ml/scripts/extract_wiktionary_words.py`](../ml/scripts/extract_wiktionary_words.py);
vocabulary files are data, so they are never committed.

Notes:

- The whole vocabulary is embedded **once, at startup**, for the same reason the
  model is: a bad path should fail the server, not the first guess.
- A guess outside the vocabulary gets `rank: null`. That is normal, not an
  error — with a small word list, most guesses will.
- The rank policy matches the ML harness's `build_rank_table` exactly: ties are
  broken by word, ascending, and the answer is always rank 1. See
  [`tests/test_ranking.py`](tests/test_ranking.py) for the parity tests.
- Memory is roughly `vocabulary_size × dimension × 8` bytes for the matrix, plus
  `RANK_CACHE_SIZE × vocabulary_size × 8` for cached answers. Nothing is stored
  per game. Past a few hundred thousand words, switch the matrix to `float32`
  (`VectorRankProvider(dtype=...)`).

### Scoring from a precomputed artifact root

`SCORING_PROVIDER=artifact` reads `similarity` and `rank` out of an **artifact
root** built offline by the ML area instead of computing them from a live model.
It is the production path: no model is loaded, so `EMBEDDING_PROVIDER`,
`FASTTEXT_MODEL_PATH` and `VOCABULARY_PATH` are all unused and the `fasttext`
extra need not be installed.

```powershell
$env:SCORING_PROVIDER = "artifact"
$env:ARTIFACT_ROOT = "C:/models/contextle-artifacts"
uv run uvicorn app.main:app --reload
```

```bash
# macOS / Linux
export SCORING_PROVIDER=artifact
export ARTIFACT_ROOT=/opt/models/contextle-artifacts
uv run uvicorn app.main:app --reload
```

The layout is [../docs/ARTIFACT_FORMAT.md](../docs/ARTIFACT_FORMAT.md). Artifact
roots are data — they are produced by the ML area and never committed.

What changes in this mode:

- **`manifest.answers` is the answer source.** A new game's answer is drawn from
  the answers the loaded root can actually serve, never from the placeholder
  `ANSWER_WORDS` list. There is one source of truth, and it is the root.
- **An out-of-vocabulary guess is `400 INVALID_WORD`.** An artifact holds a row
  for exactly the words in its `vocabulary.txt`; a word outside it cannot be
  scored at all, unlike a live model which composes a vector for anything. The
  scorer reports "no score" and `GameService` applies the rule, so the client
  gets the error code it already handles rather than a 500.
- **`rank` is never null for an accepted guess**, because every word the root
  can score also has a stored rank. The field stays nullable in the contract:
  embedding mode without a vocabulary still returns `null`.
- **No FastText at startup or at request time.** The lifespan warms the artifact
  store *instead of* the embedding service and rank provider, and the guess path
  declares no embedding dependency at all
  ([`tests/test_artifact_runtime.py`](tests/test_artifact_runtime.py) proves it
  rather than asserting it in prose).

How it loads:

- The root is **fully validated at startup** — the manifest, the canonical
  vocabulary and its hash, every answer mapping, and the existence of every file
  they refer to. A wrong `ARTIFACT_ROOT` fails the server, not the first guess.
- **Every answer must be one the game can set.** A manifest key has to be a
  canonical *vocabulary* word, and the vocabulary policy allows entries the
  *guess* policy rejects — a headword containing a space, a word past the
  50-character cap. Such an answer would break `POST /api/games` on the draws
  that happened to pick it, so startup refuses the whole root instead. Nothing is
  filtered: a root with one unusable answer is a root to rebuild. The refusal
  names the `artifact_id` and the rule, never the word.
- The per-answer `similarity.npy` / `rank.npy` arrays are **not** read at
  startup. A root of a few thousand answers is hundreds of megabytes and one
  game touches one answer's worth, so they load on first use and stay in a
  **bounded LRU cache** of `ARTIFACT_CACHE_SIZE` answers. Startup time is
  therefore flat in the number of answers.
- Memory is roughly `ARTIFACT_CACHE_SIZE × vocabulary_size × 6` bytes (a float32
  similarity array plus a uint16/32 rank array). Nothing is stored per game.
- Cache entries are keyed by `artifact_id`, never by the answer word.

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
│  ├─ ranking/        # RankProvider Protocol + null + vectorized + factory
│  ├─ scoring/        # GuessScorer Protocol + provider factory
│  │  ├─ embedding.py # EmbeddingGuessScorer: similarity + rank from a live model
│  │  └─ artifact/    # artifact root reader + bounded store + ArtifactGuessScorer
│  └─ game/           # GameRepository Protocol + in-memory store + GameService
└─ domain/            # pure game logic — no FastAPI, no model
   ├─ game.py         # Game, Guess, GameStatus, word normalization
   ├─ vocabulary.py   # vocabulary normalization + loading (matches ml/)
   └─ words.py        # placeholder answer words + AnswerSelector
```

Request flow: `routes/games.py` → `GameService` → `Game` rules. `GameService`
asks a single `GuessScorer` what a guess is worth and gets back both values, and
which scorer answers is `SCORING_PROVIDER`:

- `embedding` → `EmbeddingGuessScorer` calls the `EmbeddingService` and the
  `RankProvider`, which is why overriding either still changes what a guess
  scores.
- `artifact` → `ArtifactGuessScorer` reads one row of one precomputed answer
  artifact through the `ArtifactStore`, and no model is built at all.

Storage, scoring, and answer selection are all injected via
[`api/deps.py`](app/api/deps.py), so tests substitute a fresh store and a pinned
answer word. The provider is resolved once, in `create_app`, which is also where
the artifact scorer is wired in — a single dependency function cannot both keep
honouring an `embedding_service` override and avoid building the embedding stack
when it is unused, because FastAPI resolves every declared dependency.

The scoring seam is what made that swap a wiring change rather than a rewrite:
both values come from one lookup, so a precomputed source can replace two live
ones without `GameService` or the schemas noticing.

## Conventions

- **Routers stay thin**; logic goes in `services/` or `domain/`.
- **Wire format is camelCase**, Python stays snake_case (`schemas/base.py`).
- The embedding model is accessed only through the `EmbeddingService` Protocol,
  so mock and real implementations are interchangeable.
- The answer word is never returned to the client or logged.

See [../AGENTS.md](../AGENTS.md) for the full backend rules.
