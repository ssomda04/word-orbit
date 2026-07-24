# Roadmap

Phased plan with completion criteria. Scope is deliberately incremental — do not
pull later-phase work forward without a reason.

## Phase 0 — Collaboration environment ✅ (this setup)

Skeleton so the team can build in parallel.

**Done when:**
- [x] Separated `frontend` / `backend` / `ml` / `docs` structure.
- [x] Backend runs (`/health`), CORS from env, `EmbeddingService` + deterministic mock, tests.
- [x] Frontend runs and shows the live `/health` connection.
- [x] Version pins (`.nvmrc`, `.python-version`), `.env.example` files, Dockerfile + Compose (backend).
- [x] CI (lint/build/test) for existing areas; GitHub templates; `AGENTS.md`; README + docs.

## Phase 1 — Single-player MVP 📅

**Scope:** create game, submit guess, validation, embedding, cosine similarity,
win detection, guess history, minimal web UI, locally runnable + deployable.

**Done when:**
- `POST /api/games`, `POST /api/games/{id}/guesses`, `GET /api/games/{id}` implemented per [API_SPEC.md](./API_SPEC.md).
- Guesses validated; answer never exposed pre-win.
- Guess history renders and sorts by similarity.
- Backend tests cover game creation, guessing, win detection; frontend lint+build green.
- The dev-only `/api/dev/similarity` is removed (or explicitly kept + documented).

## Phase 2 — 3D embedding visualization 📅

**Scope:** PCA-based 3D projection, hidden answer position revealed at end,
exploration path connecting the player's guesses.

**Done when:**
- `coordinate` populated in guess responses (Phase 1 nullable → filled).
- A 3D map renders after enough guesses (3D lib chosen via a short comparison:
  React Three Fiber / Three.js / Plotly).
- Reveal shows answer position + path. Projection caveats surfaced in UI copy.

## Phase 3 — Multiplayer 📅

**Scope:** rooms (create/join), WebSocket real-time sync, competitive & co-op
modes, player-state sync, shared 3D map. **Introduces Redis (and Postgres for
history)** — enable the Compose `data` profile.

**Done when:**
- `POST /api/rooms`, `/join`, `GET /api/rooms/{id}`, `WS /ws/rooms/{id}` with the
  documented event types.
- State stays in sync across clients; answers never broadcast pre-reveal.
- Redis wired for room state; Postgres for persisted history.

## Phase 4 — Sentence mode & Attention 📅

**Scope:** sentence-embedding similarity; optional educational attention
visualization (heatmap / token links) for a chosen layer/head.

**Done when:**
- Sentence guesses scored by sentence embeddings (word-overlap-independent).
- Attention view is clearly framed as an aid, **not** a full explanation.

## Phase 5 — Evaluation & fine-tuning 📅

**Scope:** execute [MODEL_EVALUATION.md](./MODEL_EVALUATION.md); fine-tune only
if a pretrained model is measurably insufficient against the baseline.

**Done when:**
- Evaluation set (~20–30 answers) built; candidate models compared and a choice
  recorded with numbers.
- Real `EmbeddingService` implementation swapped in via `EMBEDDING_PROVIDER`.

## Phase 6 — Deployment hardening & user testing 📅

**Scope:** Vercel (frontend) + Render/Railway (backend) stable; managed
Redis/Postgres; monitoring; playtests.

**Done when:**
- Reproducible deploys from `main`; health checks green in prod.
- Feedback loop from a small user test informs the backlog.
