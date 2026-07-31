# API Specification

Status legend: ✅ implemented · 🚧 in progress · 📅 planned

This is the **contract** frontend and backend build against in parallel. Changing
it requires updating both `backend/app/schemas/*` and `frontend/src/types/api.ts`
in the same PR, and flagging the change to the team (see
[COLLABORATION.md](./COLLABORATION.md)).

## Conventions

- **JSON field naming: `camelCase`.** Chosen because the frontend is
  TypeScript-first and it keeps the client free of case conversion. The backend
  keeps idiomatic snake_case internally and serializes camelCase via a Pydantic
  alias generator (`backend/app/schemas/base.py::CamelModel`).
- **Times** are ISO-8601 strings in UTC (e.g. `2026-07-24T11:22:57Z`).
- **IDs** are opaque strings; clients must not parse them. `gameId` is a UUID
  (`0117…-…-bdc1`) so games cannot be enumerated by a third party; `guessId` is
  sequential *within* one game (`guess-001`). Neither format is part of the contract.
- **Base URL** comes from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).
- **The answer word is never sent to the client while a game is in progress** —
  not in any response, not in any log. It is revealed only after the game leaves
  `playing`, through `answer` on `GET /api/games/{gameId}` (see that endpoint).

## Error format

All handled errors return HTTP 4xx/5xx with this envelope:

```json
{
  "code": "INVALID_WORD",
  "message": "입력한 단어를 처리할 수 없습니다.",
  "details": null
}
```

| Field     | Type              | Notes                                             |
| --------- | ----------------- | ------------------------------------------------- |
| `code`    | string            | Stable machine-readable code (SCREAMING_SNAKE).   |
| `message` | string            | Human-readable, may be localized.                 |
| `details` | object \| array \| null | Optional context (e.g. validation errors).  |

Current codes: `INVALID_INPUT` (validation, 422) ✅ · `INVALID_WORD` (400) ✅ ·
`GAME_NOT_FOUND` (404) ✅ · `GAME_ALREADY_FINISHED` (409) ✅ ·
`ROOM_NOT_FOUND` 📅 · `INTERNAL_ERROR` 📅.

`INVALID_INPUT` vs. `INVALID_WORD`: the former is schema validation (missing
field, empty/whitespace-only `word`), the latter is a word the game cannot
process (too long, contains whitespace; out-of-vocabulary once a real model with
a vocabulary is wired — the deterministic mock accepts any string).

---

## Endpoints

### ✅ `GET /health`

Liveness check.

**200**

```json
{ "status": "ok" }
```

### ✅ `POST /api/dev/similarity` — dev harness only

> ⚠️ **Not a game endpoint.** Scaffolding to verify the embedding wiring.
> `POST /api/games/{gameId}/guesses` has landed, so this endpoint is now
> **explicitly kept and documented** rather than removed — the alternative
> [ROADMAP.md](./ROADMAP.md) Phase 1 allows. Rationale: it is the only way to
> check a provider's raw similarity output without creating a game, which stays
> useful while swapping in a real model. Nothing in the frontend calls it;
> removal can happen in a separate change.

**Request**

```json
{ "first": "학생", "second": "선생" }
```

**200**

```json
{ "first": "학생", "second": "선생", "similarity": -0.0657, "provider": "mock" }
```

Both fields are required and non-blank, else `422 INVALID_INPUT`.

---

### ✅ `POST /api/games` — create a single-player game

No request body.

**200** (not 201 — kept deliberately, clients depend on this)

```json
{
  "gameId": "01174974-fb2b-4ba9-9ad7-3f16e42bbdc1",
  "status": "playing",
  "createdAt": "2026-07-30T10:30:33Z"
}
```

`status` ∈ `"playing" | "won" | "abandoned"`. The answer word is chosen
server-side and never returned by this endpoint — the response has no `answer`
field at all.

> `abandoned` is reserved: no endpoint currently abandons a game, and there is no
> attempt limit, so today a game only ever moves `playing` → `won`.

### ✅ `POST /api/games/{gameId}/guesses` — submit a guess

**Request**

```json
{ "word": "학생" }
```

**200**

```json
{
  "guessId": "guess-001",
  "word": "학생",
  "similarity": 0.8732,
  "rank": null,
  "isAnswer": false,
  "coordinate": null
}
```

| Field        | Type            | Nullable | Notes                                                        |
| ------------ | --------------- | -------- | ------------------------------------------------------------ |
| `guessId`    | string          | no       |                                                              |
| `word`       | string          | no       | Normalized (trimmed) form of the guess.                      |
| `similarity` | number          | no       | Cosine similarity, range **[-1.0, 1.0]**.                    |
| `rank`       | integer \| null | **yes**  | `null` until precomputed nearest-neighbor ranks exist (Phase 1+). |
| `isAnswer`   | boolean         | no       | `true` when the guess equals the answer.                     |
| `coordinate` | object \| null  | **yes**  | `{x,y,z}` for 3D map; `null` until projection exists (Phase 2). |

**Word normalization.** The server trims the word and applies Unicode NFKC before
scoring; `word` in the response is that normalized form. Guesses are single words:
internal whitespace is rejected (sentence mode is Phase 4).

**Duplicate guesses are idempotent.** Re-submitting a word already guessed in this
game returns the **stored result unchanged** — same `guessId`, same `similarity` —
and does not create a new guess or increase `guessCount`. Duplicates are matched on
the normalized word, so `" 학생 "` and `"학생"` are the same guess.

**A finished game accepts nothing.** Once `status` is not `playing`, every guess is
rejected with `409 GAME_ALREADY_FINISHED` — including a replay of a word already
guessed. The finished check runs *before* the duplicate check, so idempotency
applies only while a game is in progress.

Errors: `404 GAME_NOT_FOUND`, `409 GAME_ALREADY_FINISHED`,
`422 INVALID_INPUT` (missing / empty / whitespace-only `word`),
`400 INVALID_WORD` (unprocessable: too long, contains whitespace; OOV later).

### ✅ `GET /api/games/{gameId}` — fetch game state

**200**

```json
{
  "gameId": "01174974-fb2b-4ba9-9ad7-3f16e42bbdc1",
  "status": "playing",
  "createdAt": "2026-07-30T10:30:33Z",
  "guessCount": 12,
  "guesses": [
    { "guessId": "guess-001", "word": "학생", "similarity": 0.8732, "rank": null, "isAnswer": false, "coordinate": null }
  ],
  "answer": null
}
```

**`guesses` ordering: submission order, oldest first.** The server returns the raw
history and never reorders it — the Phase 2 exploration path depends on that order.
Sorting by similarity for display is the client's job.

**`answer` reveal condition.** `answer` is `null` for **every** request while
`status === "playing"`, and is populated **only** once the game has reached `won`
or `abandoned` (reveal phase). There is no other endpoint or field through which
the answer word can reach a client.

`guessCount` equals `guesses.length` (duplicates are not counted twice).

Errors: `404 GAME_NOT_FOUND`.

---

## 📅 Multiplayer (Phase 3 — candidates)

REST:

| Method + Path                 | Purpose                    |
| ----------------------------- | -------------------------- |
| `POST /api/rooms`             | Create a room, returns `roomId` + join `code`. |
| `POST /api/rooms/{roomId}/join` | Join with a player name/code.               |
| `GET /api/rooms/{roomId}`     | Room + players + game status.                |

WebSocket: `WS /ws/rooms/{roomId}`. Messages use an envelope:

```json
{ "type": "PLAYER_GUESSED", "payload": { "...": "..." } }
```

Candidate event types:

| Type                 | Direction | Meaning                                    |
| -------------------- | --------- | ------------------------------------------ |
| `PLAYER_JOINED`      | server→   | A player joined the room.                  |
| `PLAYER_LEFT`        | server→   | A player left.                             |
| `GAME_STARTED`       | server→   | Round started (shared answer chosen).      |
| `PLAYER_GUESSED`     | both      | A guess was made (competitive: best score/attempts only; co-op: the word). |
| `GAME_FINISHED`      | server→   | Round ended; reveal payload.               |
| `ROOM_STATE_UPDATED` | server→   | Full room-state resync.                    |
| `ERROR`              | server→   | Uses the standard error envelope as payload. |

> Competitive vs. co-op differ in what `PLAYER_GUESSED` exposes — see
> [PRODUCT.md](./PRODUCT.md). Answer words are never broadcast before reveal.

---

## Change process

1. Discuss the contract change with the team first.
2. Update this file (source of truth).
3. Update `backend/app/schemas/*` and `frontend/src/types/api.ts` together.
4. Keep the [PR checklist](../.github/pull_request_template.md) item "API/docs
   updated" checked.
