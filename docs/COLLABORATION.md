# Collaboration Guide

Small team, simple flow. This document is the source of truth for how we work
together; [../CONTRIBUTING.md](../CONTRIBUTING.md) is the short version.

## Branch strategy

We do **not** use Git Flow. `main` is always runnable and deployable.

- `main` — protected, always green.
- `feature/<issue>-<short-name>` — new functionality.
- `fix/<issue>-<short-name>` — bug fixes.
- `docs/<short-name>` — docs-only.
- `chore/<short-name>` — tooling/config.

Examples:

```
feature/12-guess-api
feature/18-game-ui
feature/21-model-evaluation
fix/27-duplicate-guess
```

## Workflow

1. **Open an Issue** (use a template: Feature / Bug / Technical task).
2. **Assign** an owner.
3. Branch from the **latest `main`**.
4. Commit in **small** steps.
5. Open a **Pull Request** (fill in the template).
6. **≥ 1 review** required.
7. **CI must pass.**
8. **Squash merge** (preferred).
9. **Delete the branch** after merge.

## Commit messages (Conventional Commits, light)

```
feat: add game creation endpoint
fix: prevent duplicate word submission
docs: document local setup
test: add health endpoint test
refactor: separate embedding service
chore: configure backend linting
```

Types: `feat` · `fix` · `docs` · `test` · `refactor` · `chore`.

**No AI attribution.** Commits and PRs must **not** include a Claude/AI
co-author (`Co-Authored-By: Claude …`), "Generated with Claude Code" text, or
Claude listed as a contributor. Author = the human. (See [../AGENTS.md](../AGENTS.md).)

## Code review checklist (reviewer)

- Changes stay within the **requested scope** (no drive-by edits).
- The [API contract](./API_SPEC.md) is not broken (or is updated on both sides).
- **No secrets** committed.
- Input validation and error handling are present.
- Tests or a stated verification method are included.
- README/docs updated if behavior or usage changed.
- **Model logic is not coupled** into routers or UI components.

## Avoiding conflicts

- Don't touch frontend + backend + ml broadly in one PR — split by area.
- **No PRs that only reformat** large areas.
- Announce changes to **shared types / the API contract** before making them.
- If several people must edit the same file, prefer **small sequential PRs**.

## Ownership boundaries

| Area     | Owns                                                                 |
| -------- | ------------------------------------------------------------------- |
| Frontend | UI, game-state rendering, API client, TS wire types, mock data, 3D later |
| Backend  | FastAPI app, game/session/room APIs, validation, WebSocket, storage, embedding calls |
| ML       | Model comparison, eval sets, embedding interface, cosine/PCA/UMAP experiments, quality eval, fine-tuning later |

Keep experiment-only ML code in `ml/`; only validated code moves into
`backend/app/services/embedding/`.

## Branch protection (configure in GitHub UI)

> ⚠️ These are **manual** GitHub settings. This repo/agent does not and cannot
> apply them for you. Set them under **Settings → Branches → Add rule** for `main`:

- Require a pull request before merging.
- Require **≥ 1 approval**.
- Require status checks (CI) to pass before merging.
- Block direct pushes to `main`.
- Disallow **force pushes** to `main`.
- Consider **dismiss stale approvals** when new commits are pushed.

## Definition of Done

A task is done when:

- It runs locally.
- Lint passes.
- Relevant tests pass.
- Docs are updated.
- No secrets are included.
- The verification method is described in the PR.
