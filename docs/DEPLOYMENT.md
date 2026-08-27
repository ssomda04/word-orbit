# Deployment Guide

How Contextle is deployed: **frontend → Vercel**, **backend → Railway** (Docker
image). This document is the deployment contract — what the platform provides,
what this repository provides, and which variables must never be set by hand.

> **Status — Phase 2, verified locally, not yet applied to Railway.** The image
> now *contains* the 10-answer smoke artifact root
> ([§9](#9-the-shipped-smoke-artifact-root)) and has been verified end to end in
> artifact mode against a local container, including a value-for-value match
> between the API's `similarity`/`rank` and the stored `.npy` arrays.
>
> Railway still serves **embedding mode on the deterministic mock**. Applying
> [§3.2](#32-artifact-mode-phase-2) is the one remaining step, and it is a
> configuration change — the image already carries everything it needs.
>
> The smoke root is a *verification* root — **not the final production answer
> pool**, which follows once the corpus work lands
> ([MODEL_EVALUATION.md](./MODEL_EVALUATION.md)).

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

### 3.2 Artifact mode (Phase 2)

The root now ships inside the image ([§9](#9-the-shipped-smoke-artifact-root)),
so these are ready to apply. Setting `SCORING_PROVIDER=artifact` without a valid
`ARTIFACT_ROOT` fails startup by design — the server will not boot.

| Key | Value | Required |
| --- | --- | --- |
| `APP_ENV` | `production` | ✅ Unchanged from [§3.1](#31-currently-deployed-phase-1--embedding--mock). |
| `SCORING_PROVIDER` | `artifact` | ✅ Without it the server silently stays on embedding/mock. |
| `ARTIFACT_ROOT` | `/app/artifacts/smoke` | ✅ Absolute path **inside the container**, not a host path. |
| `ARTIFACT_CACHE_SIZE` | `64` | ⬜ Defaults to `64`. Roughly `N × vocabulary_size × 6` bytes — about 23 MB at this vocabulary, and the root only holds ten answers. |
| `FRONTEND_ORIGIN` | `https://<vercel-production-domain>` | ✅ Unchanged. See [§5](#5-cors-and-the-frontend-origin). |

In artifact mode `EMBEDDING_PROVIDER`, `FASTTEXT_MODEL_PATH`, `VOCABULARY_PATH`
and `RANK_CACHE_SIZE` are all unused — no model is loaded at all. Leave them
unset rather than set-and-ignored, so the configuration says what it does; that
includes **removing** the `EMBEDDING_PROVIDER` from
[§3.1](#31-currently-deployed-phase-1--embedding--mock) when the switch is made.

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

### 8.1 Artifact mode, locally

The same image, with the artifact-mode environment from
[§3.2](#32-artifact-mode-phase-2). `PORT` is set here only because nothing else
assigns one locally; on Railway it stays unset ([§3.3](#33-never-set-these)).

```bash
docker build -t contextle-backend:artifact-smoke ./backend

# the root is inside the image, owned by the runtime user
docker run --rm contextle-backend:artifact-smoke sh -c \
  'ls -ln /app/artifacts/smoke; wc -l < /app/artifacts/smoke/vocabulary.txt;
   find /app/artifacts -name "*.npy" | wc -l'
#   -> files owned by 10001; 59582; 20

# no model ships with it, and none is loaded
docker run --rm contextle-backend:artifact-smoke \
  python -c "import importlib.util; print(importlib.util.find_spec('fasttext'))"
#   -> None

docker run -d --name artifact-smoke -p 9000:9000 \
  -e PORT=9000 \
  -e APP_ENV=production \
  -e SCORING_PROVIDER=artifact \
  -e ARTIFACT_ROOT=/app/artifacts/smoke \
  -e ARTIFACT_CACHE_SIZE=64 \
  -e FRONTEND_ORIGIN=http://localhost:3000 \
  contextle-backend:artifact-smoke

curl -s http://localhost:9000/health                      # {"status":"ok"}
docker logs artifact-smoke 2>&1 | grep -E "Started (parent|server) process"
#   -> "Started server process [1]" only

# a game, an in-vocabulary guess, and an out-of-vocabulary guess (§9.1)
GAME=$(curl -s -X POST http://localhost:9000/api/games \
  | python -c "import sys,json;print(json.load(sys.stdin)['gameId'])")
curl -s -X POST http://localhost:9000/api/games/$GAME/guesses \
  -H 'content-type: application/json' -d '{"word":"가가"}'
#   -> 200 with a float similarity and an integer rank
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  http://localhost:9000/api/games/$GAME/guesses \
  -H 'content-type: application/json' -d '{"word":"없는말뷁"}'
#   -> 400 (INVALID_WORD)

docker rm -f artifact-smoke
```

> **On Git Bash / MSYS (Windows), two things bite:**
>
> - Prefix `docker run` with `MSYS_NO_PATHCONV=1`. Otherwise
>   `-e ARTIFACT_ROOT=/app/artifacts/smoke` is rewritten to a Windows path before
>   Docker sees it, and startup fails with `ARTIFACT_ROOT is not a directory`.
> - An inline `-d '{"word":"가가"}'` can reach curl re-encoded out of UTF-8, and
>   the server answers `{"detail":"There was an error parsing the body"}` — a
>   *body* error that happens to share the `400` status with a genuine
>   `INVALID_WORD`, so it will quietly counterfeit the OOV result. Send the body
>   from a UTF-8 file instead, and read the response envelope rather than the
>   status alone:
>
>   ```bash
>   printf '{"word":"가가"}' > /tmp/in.json
>   curl -s -X POST http://localhost:9000/api/games/$GAME/guesses \
>     -H 'content-type: application/json' --data-binary @/tmp/in.json
>   ```
>
>   A real out-of-vocabulary rejection is `{"code":"INVALID_WORD",…}`.

The check that actually proves artifact mode is serving stored data is
[§9.2](#92-verifying-the-api-serves-the-stored-arrays).

---

## 9. The shipped smoke artifact root

Production scoring reads a precomputed **rank artifact root**
([ARTIFACT_FORMAT.md](./ARTIFACT_FORMAT.md)) instead of a live model. One such
root ships inside the image, at `/app/artifacts/smoke`, built from
`backend/artifacts/smoke/` in the build context.

> **This is not the final production answer pool.** It is a *verification* root:
> ten answers, enough to prove the deployment path end to end and no more. The
> production pool follows once the corpus work lands
> ([MODEL_EVALUATION.md](./MODEL_EVALUATION.md)) and will be far too large to
> ship this way — it needs a real distribution channel, not a `COPY`.

| Property | Value |
| --- | --- |
| Root inside the container | `/app/artifacts/smoke` |
| Root in the repository | [`backend/artifacts/smoke/`](../backend/artifacts/smoke/) |
| `schema_version` | `1.0` |
| Answers | 10 |
| Vocabulary | 59,582 words |
| Vocabulary SHA-256 | `92f4fb52259e7f708610609b2f883c46a7a5edb3d11ab0ac933f336d09734a9d` |
| `similarity_dtype` | `float32` |
| `rank_dtype` | `uint16` |
| `embedding_model` | `fasttext-cc-ko-300` (`cc.ko.300.bin`) — used **offline**, never shipped |
| `.npy` files | 20 (a `similarity.npy` + `rank.npy` per answer) |
| Total files | 22 (20 arrays + `manifest.json` + `vocabulary.txt`) |
| Uncompressed size | ~4.05 MiB (4,249,250 bytes) |

The ten answer words are not repeated here — they are in `manifest.json`, which
is the source of truth, and a deployment document is a poor place to duplicate
game secrets. The root is server-side data: `manifest.json` lists every answer in
plain text, so anyone with the image has them ([ARTIFACT_FORMAT.md §1.1](./ARTIFACT_FORMAT.md#11-artifact_id)).

The arrays are versioned in Git through a narrow exception to the global `*.npy`
rule in the root [`.gitignore`](../.gitignore), scoped to `backend/artifacts/`.
That exception is about *this* 4 MiB root, not a policy that artifact roots
belong in the repository.

### 9.1 Probe words

Both probes are selected **from the shipped `vocabulary.txt`**, never assumed. An
out-of-vocabulary guess and a malformed one both return `400 INVALID_WORD` with
the same message, so a probe that happened to be malformed would prove nothing —
each must satisfy the guess rules in `app/domain/game.py`: non-blank, no internal
whitespace, at most 50 characters, and unchanged by NFKC normalization.

| Probe | Word | Chosen by |
| --- | --- | --- |
| IN | `가가` | First word in canonical `vocabulary.txt` order that is at least two characters, is a legal guess, and is **not** one of the ten answers. Vocabulary index `1`. |
| OOV | `없는말뷁` | First generated candidate absent from `vocabulary.txt` that is still a legal guess. |

Re-select them rather than trusting this table if the root is ever replaced: both
are facts about *this* vocabulary, and the IN probe must keep being a non-answer.

### 9.2 Verifying the API serves the stored arrays

`rank` merely coming back as an integer is weak evidence. Nothing in the request
path rounds — `AnswerArtifact.score_at` returns `float(similarity[i])` and
`int(rank[i])` — so the API value must equal the stored value **exactly**, with
no tolerance.

The check that does not require knowing the answer: submit the IN probe, then
read `similarity[<probe index>]` and `rank[<probe index>]` out of each of the ten
answers' `.npy` files. The response must equal exactly one of those ten
candidate pairs — which both proves the value is the stored one and identifies
which answer the game drew, without the answer ever being logged.

Guessing that answer must then return `similarity == 1.0`, `rank == 1`, and a
`won` game, matching that answer's own self-row in its arrays.

---

## 10. Applying artifact mode to Railway

The image already carries the root and has been verified locally
([§8.1](#81-artifact-mode-locally), [§9.2](#92-verifying-the-api-serves-the-stored-arrays)),
so what remains is configuration:

1. Set the [§3.2](#32-artifact-mode-phase-2) variables and remove
   `EMBEDDING_PROVIDER`.
2. Redeploy, then check the deploy log for the worker signature
   ([§4](#4-why-one-process-is-a-correctness-requirement)) and for the absence of
   any artifact error — a bad `ARTIFACT_ROOT` stops the process at startup rather
   than degrading to mock, so a service that is up is a service that validated
   its root.
3. Run the smoke test in [§7](#7-smoke-test) with the [§9.1](#91-probe-words)
   probes: the IN probe must return an integer `rank`, and the OOV probe must
   return `400 INVALID_WORD`. In embedding mode the OOV probe would have scored
   normally, so that `400` is the signal that the switch actually took effect.

Do not add `PORT`, `WEB_CONCURRENCY`, or `FASTTEXT_MODEL_PATH`
([§3.3](#33-never-set-these)) — the last one has nothing to load in this image.
