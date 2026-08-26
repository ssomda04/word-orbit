# Deployment Guide

How Contextle is deployed: **frontend → Vercel**, **backend → Railway** (Docker
image). This document is the deployment contract — what the platform provides,
what this repository provides, and which variables must never be set by hand.

> **Status — Phase 1 of 2.** The image is hardened for Railway and production
> currently serves **embedding mode on the deterministic mock**. Artifact mode is
> fully implemented in the backend but **not yet deployed**: no artifact root has
> been produced by the ML area, and this repository deliberately contains none.
> See [Phase 2](#phase-2--artifact-mode-blocked-on-the-ml-artifact-root).

---

## 1. What the image guarantees

[`backend/Dockerfile`](../backend/Dockerfile) is deployment-target agnostic.
Nothing about Railway, an artifact root, or a frontend origin is baked in — the
image reads all of it from the environment. Three properties are guaranteed by
the image itself and must not be overridden:

| Property | How | Why it matters |
| --- | --- | --- |
| Binds `$PORT`, defaults to `8000` | `CMD ["sh","-c","exec uvicorn … --port ${PORT:-8000} …"]` | The platform picks the port. `sh -c` is required because an exec-form `CMD` has no shell and would pass `${PORT}` to uvicorn as a literal string. |
| uvicorn receives `SIGTERM` directly | `exec` in the same `CMD` | Without it `sh` keeps PID 1, the platform's `SIGTERM` never reaches uvicorn, and every redeploy becomes a `SIGKILL` instead of a graceful shutdown. |
| Exactly **one** worker process | `--workers 1` | uvicorn's `--workers` defaults to `$WEB_CONCURRENCY`. See [§4](#4-why-one-process-is-a-correctness-requirement). |

The container runs as the unprivileged user `appuser` (uid/gid `10001`). The
`HEALTHCHECK` reads `PORT` from the environment in Python, so it stays correct
whatever port the platform assigns. Railway does not use Docker's `HEALTHCHECK` —
it probes its own configured path — but keeping the two consistent means a local
`docker run -e PORT=9000` does not report the container as unhealthy.

---

## 2. Railway service configuration

Set once, in the service settings — not as environment variables.

| Setting | Value | Notes |
| --- | --- | --- |
| Root Directory | `backend` | This is also the Docker **build context**. `COPY` cannot reach outside it. |
| Builder | Dockerfile | `backend/Dockerfile`. |
| Healthcheck Path | `/health` | Returns `{"status":"ok"}`. |
| Replicas | **1** | See [§4](#4-why-one-process-is-a-correctness-requirement). |
| Restart Policy | `on-failure` | A failed start is a configuration error worth surfacing, not retrying forever. |
| Public Domain | generated | `https://<service>.up.railway.app` — the value the frontend points at. |

---

## 3. Environment variables

### 3.1 Currently deployed (Phase 1 — embedding / mock)

| Key | Value | Required |
| --- | --- | --- |
| `APP_ENV` | `production` | ✅ |
| `FRONTEND_ORIGIN` | `https://<vercel-production-domain>` | ✅ See [§5](#5-cors-and-the-frontend-origin). |
| `EMBEDDING_PROVIDER` | `mock` | ⬜ Defaults to `mock`. |

`SCORING_PROVIDER` is unset, which means `embedding` — the historical default.
Every guess is scored by the deterministic mock, and `rank` is always `null`
because no `VOCABULARY_PATH` is configured. That is a valid API response
([API_SPEC.md](./API_SPEC.md)), not a fault.

### 3.2 Artifact mode (Phase 2 — not yet applied)

Apply these **only** once a real artifact root has been received and committed.
Setting `SCORING_PROVIDER=artifact` without a valid `ARTIFACT_ROOT` fails startup
by design — the server will not boot.

| Key | Value | Required |
| --- | --- | --- |
| `SCORING_PROVIDER` | `artifact` | ✅ Without it the server silently stays on embedding/mock. |
| `ARTIFACT_ROOT` | `/app/artifacts/smoke` | ✅ Absolute path **inside the container**. Valid only after the artifact ships in the image. |
| `ARTIFACT_CACHE_SIZE` | `64` | ⬜ Defaults to `64`. Roughly `N × vocabulary_size × 6` bytes. |

In artifact mode `EMBEDDING_PROVIDER`, `FASTTEXT_MODEL_PATH`, `VOCABULARY_PATH`
and `RANK_CACHE_SIZE` are all unused — no model is loaded at all. Leave them
unset rather than set-and-ignored, so the configuration says what it does.

### 3.3 Never set these

| Key | Why not |
| --- | --- |
| `PORT` | Railway injects it. Setting it by hand overrides the platform's own routing and is the classic cause of a service that builds, starts, and never receives traffic. |
| `WEB_CONCURRENCY` | uvicorn reads it as the default for `--workers`. See [§4](#4-why-one-process-is-a-correctness-requirement). The image passes `--workers 1` explicitly, which overrides it — but the variable should not exist in the first place. |

### 3.4 Reserved

`MODEL_NAME`, `DATABASE_URL`, `REDIS_URL` — later phases
([ROADMAP.md](./ROADMAP.md)). Leave empty.

---

## 4. Why one process is a correctness requirement

Game state lives in `InMemoryGameRepository`, built once per process by an
`@lru_cache` in [`backend/app/api/deps.py`](../backend/app/api/deps.py). It is
**not** shared between processes.

So a second process — whether a second uvicorn worker or a second Railway
replica — means a game created on one is `404` on the other, intermittently and
at random. In artifact mode it also duplicates the per-answer array cache in
every worker, multiplying memory by the worker count.

Three things keep this at one, and all three are needed:

1. `--workers 1` in the image's `CMD`. A command-line value overrides the
   `WEB_CONCURRENCY` environment variable.
2. `WEB_CONCURRENCY` absent from the Railway environment ([§3.3](#33-never-set-these)).
3. `Replicas = 1` in the service settings ([§2](#2-railway-service-configuration)).

**Verifying it in production.** Railway offers no shell, so the deploy log is the
evidence. uvicorn only starts its multiprocess supervisor when
`reload or workers > 1`, and only that supervisor logs `Started parent process`:

| Deploy log contains | Meaning |
| --- | --- |
| `Started server process [1]` only | ✅ one worker |
| `Started parent process [...]` | ❌ more than one worker — check `WEB_CONCURRENCY` |

Check this after every deploy that changes the image or the environment.

This constraint holds only while game state is in-memory. Scaling out requires
persistence first, not a replica-count change.

---

## 5. CORS and the frontend origin

`FRONTEND_ORIGIN` is a **comma-separated exact-match list**
([`backend/app/core/config.py`](../backend/app/core/config.py)); there is no
wildcard and no regex. Include the scheme, omit any trailing slash — the value is
compared byte-for-byte against the browser's `Origin` header.

```bash
FRONTEND_ORIGIN=https://word-orbit.vercel.app
# multiple origins (e.g. once a custom domain exists):
# FRONTEND_ORIGIN=https://word-orbit.vercel.app,https://wordorbit.app
```

> **Known limitation — Vercel preview deployments.** Every preview gets its own
> hostname (`word-orbit-git-<branch>-<team>.vercel.app`), which a static list
> cannot cover. Supporting them needs `allow_origin_regex`, which the backend does
> not currently wire up. Until then, develop previews against a local backend.

Verify with a preflight rather than a browser:

```bash
BASE=https://<service>.up.railway.app

# allowed origin -> the header echoes exactly that origin
curl -s -i -X OPTIONS $BASE/api/games \
  -H "Origin: https://word-orbit.vercel.app" \
  -H "Access-Control-Request-Method: POST" | grep -i access-control-allow-origin

# disallowed origin -> no such header at all
curl -s -i -X OPTIONS $BASE/api/games \
  -H "Origin: https://evil.example" \
  -H "Access-Control-Request-Method: POST" | grep -i access-control-allow-origin
```

---

## 6. Connecting Vercel to Railway

The two halves point at each other, so deploy the backend first.

1. **Deploy the backend** and copy its public domain
   (`https://<service>.up.railway.app`). Confirm `GET /health` returns
   `{"status":"ok"}`.
2. **In the Vercel project**, set Root Directory to `frontend` and add the
   environment variable — see [`frontend/.env.example`](../frontend/.env.example):

   | Key | Value | Environment |
   | --- | --- | --- |
   | `NEXT_PUBLIC_API_URL` | `https://<service>.up.railway.app` | Production |

   `NEXT_PUBLIC_WS_URL` is for later multiplayer phases; leave it unset for now.
   Only `NEXT_PUBLIC_*` variables reach the browser.
3. **Deploy the frontend** and copy its production domain.
4. **Back in Railway**, set `FRONTEND_ORIGIN` to that domain ([§5](#5-cors-and-the-frontend-origin))
   and redeploy the backend.
5. **Verify** with the preflight commands in [§5](#5-cors-and-the-frontend-origin),
   then load the site and confirm a game starts.

Changing either domain later means updating the other side and redeploying.

---

## 7. Smoke test

Run after every deploy. `BASE` is the Railway public domain.

```bash
BASE=https://<service>.up.railway.app

# liveness and docs
curl -s $BASE/health                                    # {"status":"ok"}
curl -s -o /dev/null -w '%{http_code}\n' $BASE/docs     # 200

# create a game and submit a guess
GAME=$(curl -s -X POST $BASE/api/games \
  | python -c "import sys,json;print(json.load(sys.stdin)['gameId'])")
curl -s -X POST $BASE/api/games/$GAME/guesses \
  -H 'content-type: application/json' -d '{"word":"학생"}'

# one process: every lookup of the same game must succeed
for i in $(seq 1 10); do
  curl -s -o /dev/null -w '%{http_code} ' $BASE/api/games/$GAME
done                                                    # ten 200s; any 404 => §4

# the answer never leaves the server while a game is playing
curl -s $BASE/api/games/$GAME | grep -i answer          # no output
```

Also check the deploy log for the worker signature in [§4](#4-why-one-process-is-a-correctness-requirement).

> `POST /api/dev/similarity` is **not** a deployment check. It is scaffolding that
> reads through the embedding service, so it reports mock numbers regardless of
> which scoring provider is active.

---

## 8. Local verification before deploying

The CI workflow ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs
`ruff` and `pytest`; it does **not** build or run the Docker image. Any change to
[`backend/Dockerfile`](../backend/Dockerfile) must therefore be verified by hand.

```bash
# from the repository root — build context is backend/, exactly as on Railway
docker build -t contextle-backend:deployment ./backend

# runs as the unprivileged user, and owns what it needs
docker run --rm contextle-backend:deployment id          # uid=10001(appuser) gid=10001(appuser)
docker run --rm contextle-backend:deployment ls -ln /app # app/ owned by 10001 10001

# honours an injected PORT
docker run --rm -d --name smoke -p 9000:9000 \
  -e PORT=9000 \
  -e APP_ENV=production \
  -e FRONTEND_ORIGIN=http://localhost:3000 \
  contextle-backend:deployment
curl -s http://localhost:9000/health                     # {"status":"ok"}

# one worker, even with WEB_CONCURRENCY set. Checked from the host: the base
# image has no procps, and it must not gain one just to be testable.
docker top smoke -o pid,args                             # a single uvicorn line
docker logs smoke 2>&1 | grep -E "Started (parent|server) process"
#   -> "Started server process [1]" only; "Started parent process" means failure

docker rm -f smoke

# the Compose path must keep working — this is the regression that matters most
docker compose up --build backend
curl -s http://localhost:8000/health
docker compose down
```

---

## Phase 2 — artifact mode (blocked on the ML artifact root)

Production scoring will move to a precomputed **rank artifact root**
([ARTIFACT_FORMAT.md](./ARTIFACT_FORMAT.md)), which the backend already
implements end to end. What is missing is the data: the ML area produces artifact
roots, and no root has been handed over yet.

Artifact roots are data, not source. This repository contains none, and none is
generated locally — a synthetic root would prove the plumbing works while telling
us nothing about the numbers the game actually serves.

**When a small (10–50 answer) root arrives, Phase 2 is:**

1. Place it at `backend/artifacts/smoke/` — inside the Docker build context, which
   is the only place `COPY` can reach ([§2](#2-railway-service-configuration)).
2. Add `!backend/artifacts/**/*.npy` to the root [`.gitignore`](../.gitignore).
   The global `*.npy` rule blocks the arrays; the negation works here because no
   parent directory of `backend/artifacts/` is itself excluded.
3. Add `COPY --chown=appuser:appuser artifacts ./artifacts` to
   [`backend/Dockerfile`](../backend/Dockerfile), **before** `COPY … app ./app` so
   an application change does not invalidate the artifact layer. The user already
   exists at that point in the file, so the name-based `--chown` resolves.
4. Set the artifact-mode variables from [§3.2](#32-artifact-mode-phase-2--not-yet-applied)
   and redeploy.
5. Extend the smoke test in [§7](#7-smoke-test):
   - Pick the probe words from the shipped `vocabulary.txt`, rather than assuming
     any particular word is in it. Both probes must satisfy the guess rules — no
     internal whitespace, at most 50 characters — because an out-of-vocabulary
     guess and a malformed one both return `400 INVALID_WORD` with the same
     message, so a malformed probe would prove nothing.
   - Compare the returned `similarity` and `rank` against the values stored in the
     root's `.npy` files. Nothing in the request path rounds, so the match is
     exact. `rank` merely being an integer is weak evidence; a value-for-value
     match is what shows the API is serving the stored artifact.

Steps 1–3 are deliberately absent from Phase 1: adding
`COPY artifacts ./artifacts` before the directory exists breaks the build for
everyone.
