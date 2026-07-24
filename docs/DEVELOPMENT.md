# Development Guide

How to set up, run, and verify Contextle locally. Commands are given for both
**Windows PowerShell** and **macOS/Linux** where they differ.

## Prerequisites

| Tool    | Version                     | Pinned in            | Notes                          |
| ------- | --------------------------- | -------------------- | ------------------------------ |
| Node.js | ≥ 20.9 (repo pins **22**)   | [`.nvmrc`](../.nvmrc)          | `nvm use` to match.  |
| npm     | bundled with Node           | —                    | Lockfile = `package-lock.json`. |
| Python  | **3.12**                    | [`.python-version`](../.python-version) | For backend + ml.   |
| uv      | latest (recommended)        | —                    | Python deps; fallback pip.     |
| Docker  | optional                    | —                    | For the backend container.     |

Install uv: <https://docs.astral.sh/uv/getting-started/installation/>.

## First-time setup

```bash
# Frontend
cd frontend
npm install
# copy env example:
#   PowerShell:      Copy-Item .env.example .env.local
#   macOS/Linux:     cp .env.example .env.local

# Backend
cd ../backend
uv sync
#   PowerShell:      Copy-Item .env.example .env
#   macOS/Linux:     cp .env.example .env
```

## Running locally (recommended: two terminals)

**Terminal 1 — backend** (Docker):

```bash
docker compose up --build backend
# API on http://localhost:8000  (docs at /docs)
```

…or backend **without Docker**:

```bash
cd backend
uv run uvicorn app.main:app --reload
```

**Terminal 2 — frontend**:

```bash
cd frontend
npm run dev
# App on http://localhost:3000
```

Open <http://localhost:3000>. The **Backend /health** indicator turns green when
the connection works.

## Lint / test / build

**Frontend**

```bash
cd frontend
npm run lint
npm run build
```

**Backend**

```bash
cd backend
uv run ruff check .
uv run pytest
```

> No uv? Use a venv:
> ```bash
> python -m venv .venv
> # PowerShell:   .venv\Scripts\Activate.ps1
> # macOS/Linux:  source .venv/bin/activate
> pip install fastapi "uvicorn[standard]" pydantic pydantic-settings pytest httpx ruff
> ruff check .
> pytest
> ```

## Docker commands

```bash
docker compose config            # validate compose file
docker compose build backend     # build image
docker compose up backend        # run (http://localhost:8000/health)
docker compose --profile data up # ALSO start redis + postgres (later phases)
```

## Environment variables

Never commit real `.env` files (git-ignored). Copy the examples:

- Frontend: [`frontend/.env.example`](../frontend/.env.example) →
  `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_WS_URL`.
- Backend: [`backend/.env.example`](../backend/.env.example) → `APP_ENV`,
  `FRONTEND_ORIGIN`, `EMBEDDING_PROVIDER`, `MODEL_NAME`, `DATABASE_URL`, `REDIS_URL`.

## Troubleshooting

| Symptom                                   | Fix                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------- |
| Health dot is red                         | Start the backend; check `NEXT_PUBLIC_API_URL` matches its port.    |
| CORS error in browser console             | Ensure `FRONTEND_ORIGIN` includes `http://localhost:3000`.          |
| `uv: command not found`                   | Install uv, or use the pip/venv fallback above.                     |
| Wrong Node version / build oddities       | `nvm use` (reads `.nvmrc`), then reinstall: delete `node_modules`.  |
| `ModuleNotFoundError: app` in pytest      | Run pytest from `backend/` (pytest `pythonpath=["."]` handles imports). |
| Stale frontend build / weird cache        | Delete `frontend/.next` and rebuild.                                |
| Stale Python caches                       | Delete `.pytest_cache`, `.ruff_cache`, `__pycache__`.               |
| Port already in use (8000/3000)           | Stop the other process or change the port (`--port`, `PORT`).       |

### Clearing caches quickly

```bash
# Frontend
rm -rf frontend/.next frontend/node_modules   # PowerShell: Remove-Item -Recurse -Force
# Backend
rm -rf backend/.pytest_cache backend/.ruff_cache
```
