# Contributing to Contextle

Thanks for helping build Contextle! This is the quick version; the full guide is
[docs/COLLABORATION.md](./docs/COLLABORATION.md).

## Before you start

- Read [README.md](./README.md) and [docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md)
  to get running locally.
- Skim [docs/API_SPEC.md](./docs/API_SPEC.md) — it's the contract between
  frontend and backend.
- AI agents: also read [AGENTS.md](./AGENTS.md).

## Workflow (short)

1. **Open an issue** (Feature / Bug / Technical task template).
2. Branch from the latest `main`:
   - `feature/<issue>-<short-name>` · `fix/<issue>-<short-name>` ·
     `docs/<short-name>` · `chore/<short-name>`
3. Make **small** commits using Conventional Commits:
   `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
4. Run the checks for your area:
   - Frontend: `npm run lint && npm run build` (in `frontend/`)
   - Backend: `uv run ruff check . && uv run pytest` (in `backend/`)
5. Open a **Pull Request** and fill in the template.
6. Get **≥ 1 approval** and a **green CI** run.
7. **Squash merge**, then delete the branch.

## Ground rules

- Keep PRs scoped to **one area** where possible (frontend / backend / ml).
- Don't commit secrets or `.env` files, model weights, datasets, or build output.
- Changing the API contract? Update `docs/API_SPEC.md`,
  `backend/app/schemas/*`, and `frontend/src/types/api.ts` **together**, and give
  the team a heads-up first.
- Don't submit PRs that only reformat large areas.

## Definition of Done

Runs locally · lint passes · tests pass · docs updated · no secrets · PR
describes how to verify.
