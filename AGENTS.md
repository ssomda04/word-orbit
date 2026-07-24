# AGENTS.md

Rules for AI coding agents (Claude Code, Codex, etc.) working in this repo.
Human contributors: see [CONTRIBUTING.md](./CONTRIBUTING.md) and
[docs/COLLABORATION.md](./docs/COLLABORATION.md).

> There is also a **frontend-specific** [`frontend/AGENTS.md`](./frontend/AGENTS.md):
> this project uses **Next.js 16** whose APIs may differ from your training data.
> Read the bundled docs in `frontend/node_modules/next/dist/docs/` before writing
> framework code.

## Project principles

- **Do not** perform large, unrequested refactors.
- **Read first:** open the relevant files and docs before changing anything.
- **Respect area boundaries** — frontend, backend, ML each own their layer
  (see [docs/COLLABORATION.md](./docs/COLLABORATION.md)).
- **Never put embedding/model logic in React components or API routers.**
- **Keep the mock↔real swap intact:** everything depends on the
  `EmbeddingService` Protocol, selected by `EMBEDDING_PROVIDER`.
- **Contract changes travel together:** update
  [docs/API_SPEC.md](./docs/API_SPEC.md), `backend/app/schemas/*`, and
  `frontend/src/types/api.ts` in the same change.
- **Never commit** secrets, `.env` files, heavy model files, datasets,
  checkpoints, build output, or caches.
- **Add dependencies sparingly** — justify each one; keep the core minimal.
- **Run the relevant `lint` / `build` / `test` before declaring done.**
- **Report failing checks** — never hide or suppress them.
- **No git commits/pushes and no GitHub-settings changes** unless explicitly asked.

## Frontend rules

- Keep TypeScript **strict**; avoid `any`.
- API calls live only in `src/lib/api/*` — components never call `fetch` directly.
- Declare response types in `src/types/api.ts`, mirroring the API contract.
- Always handle **loading / error / empty** states.
- No model math in the UI.
- Scope `'use client'` to components that genuinely need interactivity.

## Backend rules

- Routers stay thin: validate input, call a service, shape the response.
- Business logic lives in `app/services/` or `app/domain/`.
- Validate all input with Pydantic schemas.
- Load the embedding model **once** at startup (via the cached factory / DI).
- **Never expose the answer word** in a response or a log.
- Choose async vs. sync deliberately and be able to justify it.
- Prefer dependency injection so handlers/services are testable.

## ML rules

- Experiment code stays under `ml/`; never import `ml/` from the backend.
- Record **seed, model name, library versions, and data version** for every run.
- Document model-comparison results in
  [docs/MODEL_EVALUATION.md](./docs/MODEL_EVALUATION.md).
- Do **not** claim 3D-projection distances equal true semantic distance.
- Do **not** present attention weights as a complete explanation of a decision.
- Define the **evaluation set and baseline before** any training.
- Fine-tune **only** when a pretrained model is measurably insufficient.

## Task-completion report format

When finishing a task, report:

1. **Files created or changed.**
2. **Key work performed.**
3. **Verification commands run.**
4. **Verification results** (state failures honestly).
5. **Open issues / limitations.**
6. **What a teammate should do next.**
