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
- **IDs** are opaque strings (`game-001`, `guess-001`, …); clients must not parse them.
- **Base URL** comes from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).
- **The answer word is never sent to the client** in any response or log.

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

Current codes: `INVALID_INPUT` (validation, 422) ✅ · `INVALID_WORD` 📅 ·
`GAME_NOT_FOUND` 📅 · `ROOM_NOT_FOUND` 📅 · `INTERNAL_ERROR` 📅.

---

## Endpoints

### ✅ `GET /health`

Liveness check.

**200**

```json
{ "status": "ok" }
```

### ✅ `POST /api/dev/similarity` — dev harness only

> ⚠️ **Not a game endpoint.** Scaffolding to verify the embedding wiring before
> the game API exists. **Slated for removal** once
> `POST /api/games/{gameId}/guesses` lands (see [ROADMAP.md](./ROADMAP.md)).

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

### 📅 `POST /api/games` — create a single-player game

**200**

```json
{
  "gameId": "game-001",
  "status": "playing",
  "createdAt": "2026-07-24T11:22:57Z"
}
```

`status` ∈ `"playing" | "won" | "abandoned"`. The answer word is chosen
server-side and never returned.

### 📅 `POST /api/games/{gameId}/guesses` — submit a guess

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
  "rank": 24,
  "isAnswer": false,
  "coordinate": { "x": 0.42, "y": -1.03, "z": 0.77 }
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

Errors: `404 GAME_NOT_FOUND`, `422 INVALID_INPUT`, `400 INVALID_WORD` (OOV / unprocessable).

### 📅 `GET /api/games/{gameId}` — fetch game state

**200**

```json
{
  "gameId": "game-001",
  "status": "playing",
  "createdAt": "2026-07-24T11:22:57Z",
  "guessCount": 12,
  "guesses": [
    { "guessId": "guess-001", "word": "학생", "similarity": 0.8732, "rank": 24, "isAnswer": false, "coordinate": null }
  ],
  "answer": null
}
```

`answer` is `null` while `status === "playing"` and is only populated after the
game is `won`/`abandoned` (reveal phase).

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
