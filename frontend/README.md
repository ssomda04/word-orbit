# Contextle Frontend

Next.js (App Router) + TypeScript + Tailwind CSS. This is the **Phase 0
skeleton**: it renders a landing page and verifies the connection to the
backend `/health` endpoint. The game UI is not built yet — see
[../docs/ROADMAP.md](../docs/ROADMAP.md).

> ⚠️ This project uses **Next.js 16** (React 19, React Compiler). Some APIs
> differ from older Next.js. Read [`AGENTS.md`](./AGENTS.md) and the bundled
> docs in `node_modules/next/dist/docs/` before writing framework code.

## Requirements

- Node.js **≥ 20.9** (pinned to **22** in [`../.nvmrc`](../.nvmrc); `nvm use`)
- Package manager: **npm** (this repo uses `package-lock.json` — do not add
  other lockfiles).

## Getting started

```bash
cd frontend
npm install
cp .env.example .env.local     # PowerShell: Copy-Item .env.example .env.local
npm run dev                    # http://localhost:3000
```

The landing page shows **“Frontend is running”** and a live **Backend /health**
indicator. For it to turn green, start the backend first (see
[../backend/README.md](../backend/README.md) or `docker compose up backend`).

## Scripts

| Command         | What it does                        |
| --------------- | ----------------------------------- |
| `npm run dev`   | Start the dev server (HMR).         |
| `npm run lint`  | ESLint (`eslint-config-next`).      |
| `npm run build` | Production build (run before PRs).  |
| `npm run start` | Serve the production build.         |

## Environment

Only `NEXT_PUBLIC_*` variables reach the browser. See [`.env.example`](.env.example):

| Variable              | Default                 | Purpose                          |
| --------------------- | ----------------------- | -------------------------------- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend base URL.                |
| `NEXT_PUBLIC_WS_URL`  | `ws://localhost:8000`   | WebSocket base URL (later phase). |

## Code boundaries

```
src/
├─ app/           # routes, layouts (App Router)
├─ components/    # reusable presentational UI (no fetch, no model logic)
├─ features/      # feature-scoped UI (e.g. health/HealthStatus.tsx)
├─ lib/api/       # API client + endpoint functions (the ONLY place fetch lives)
└─ types/         # shared wire types, kept in sync with the API contract
```

- UI components never call `fetch` directly — use `@/lib/api`.
- Response types are declared in `@/types/api` and mirror
  [../docs/API_SPEC.md](../docs/API_SPEC.md).
- Keep `'use client'` scoped to components that truly need interactivity.

See [../AGENTS.md](../AGENTS.md) for the full frontend rules.
