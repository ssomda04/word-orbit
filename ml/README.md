# Contextle ML & Data Experiments

This area is for **experiments**: comparing Korean embedding models, building
evaluation sets, and prototyping dimensionality reduction (PCA → later UMAP).
It is intentionally separate from the production backend.

> ⚠️ **Boundary rule.** Notebooks and scripts here are exploratory. Only code
> that has been validated gets promoted into the backend as an implementation of
> the `EmbeddingService` Protocol (`backend/app/services/embedding/`). Do not
> import from `ml/` in the backend.

## Directory layout

```
ml/
├─ notebooks/    # Jupyter notebooks — exploration only, never imported by services
├─ scripts/      # reproducible CLI scripts (data prep, batch evaluation)
├─ evaluation/   # evaluation sets + metrics for model comparison
├─ data/         # local datasets — NOT committed (see data/README.md)
└─ README.md
```

## What lives here (and what does not)

| Do here                                             | Do NOT here                                  |
| --------------------------------------------------- | -------------------------------------------- |
| Compare candidate models, record metrics            | Serve HTTP / game logic (that's `backend/`)  |
| Build the word-similarity evaluation set            | Commit model weights, checkpoints, datasets  |
| Prototype PCA/UMAP projections                      | Hard-code experiment-only code into services |
| Draft the `EmbeddingService` real implementation    | Fine-tune before a baseline exists           |

## Reproducibility rules (see ../AGENTS.md → ML rules)

Every experiment must record:

- **Seed(s)** used.
- **Model name + version** and library versions (`sentence-transformers`, `torch`, …).
- **Data version** (which evaluation set / snapshot).
- The **metric definitions** and results, written up in
  [../docs/MODEL_EVALUATION.md](../docs/MODEL_EVALUATION.md).

## Getting started (later phases)

The heavy libraries are **not** installed by default. When you start model work:

```bash
# Option A — reuse the backend project's optional extra:
cd backend
uv sync --extra embeddings          # sentence-transformers, torch, scikit-learn, numpy

# Option B — a standalone venv for notebooks:
python -m venv .venv
# activate, then:  pip install sentence-transformers scikit-learn jupyter matplotlib
```

Keep notebooks runnable top-to-bottom and prefer moving stable logic into
`scripts/` so it can run in CI-free batch jobs.

## FastText baseline harness (dataset v0.2)

The versioned dataset separates four relationships: `veryClose` (same or very
direct concept), `related` (different concepts sharing a normal context),
`unrelated` (no meaningful or ordinary contextual connection), and `surfaceTrap`
(form-only resemblance). Do not interpret `surfaceTrap` as an expected semantic
ordering group.

The harness calculates raw cosine similarity and five strict pairwise accuracies:
`veryClose > unrelated`, `related > unrelated`, `veryClose > related`,
`veryClose > surfaceTrap`, and `related > surfaceTrap`. Precision@k and Recall@k
use only `veryClose` as relevant; separate counts show related words and surface
traps entering the candidate-only top-k. This ranking is not a full-vocabulary
game rank.

From the repository root, run a local, already-downloaded model on Windows:

```powershell
uv run --project backend --extra embeddings python .\ml\scripts\evaluate_fasttext.py `
  --model-path "C:\models\cc.ko.300.bin" `
  --dataset ".\ml\evaluation\word_similarity_eval.json" `
  --output-dir ".\ml\evaluation\results\fasttext-v0.2" `
  --top-k 10 `
  --seed 42
```

For macOS/Linux:

```bash
uv run --project backend --extra embeddings python ml/scripts/evaluate_fasttext.py \
  --model-path "/path/to/cc.ko.300.bin" \
  --dataset "ml/evaluation/word_similarity_eval.json" \
  --output-dir "ml/evaluation/results/fasttext-v0.2" \
  --top-k 10 \
  --seed 42
```

The output directory must be new or empty. Existing `results/fasttext/` files
belong to the v0.1 criteria and are intentionally preserved. Increment the JSON
dataset version whenever candidate membership or relationship definitions change.

## Honest-interpretation reminders

- A **3D projection is a lossy view**, not the true high-dimensional semantic
  distance. Never present projected distances as exact similarity.
- If/when Attention visualization is added (Phase 4), **attention weights are
  not a complete explanation** of a model's decision.
