# Model Evaluation Plan

> **No model is selected yet.** The skeleton runs on
> `DeterministicEmbeddingService` (a hash-based mock). This document is the
> **plan** for choosing a real Korean embedding model — not a decision.

## Goal

Pick a Korean word/sentence embedding model that makes Contextle's similarity
scores feel *natural* to players, while being cheap enough to run (ideally CPU)
and license-clean for deployment.

## Baseline first

Before any comparison or fine-tuning:

1. Freeze an **evaluation set** (see below and `ml/evaluation/`).
2. Define **metrics** and a **baseline** (the deterministic mock is the trivial
   floor; a first off-the-shelf model is the real baseline).
3. Only consider fine-tuning **if** a pretrained model is measurably insufficient
   against this baseline.

## Candidate models (to compare — not endorsements)

Evaluate a spread of Korean-capable sentence/word encoders, e.g.:

- `jhgan/ko-sroberta-multitask`
- `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- `BM-K/KoSimCSE-roberta`
- A multilingual baseline (e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)

(Confirm current availability and licenses at evaluation time.)

## Comparison criteria

| Criterion                         | Why it matters                                   |
| --------------------------------- | ------------------------------------------------ |
| Korean word-similarity naturalness | Core gameplay feel.                             |
| Sentence-embedding quality        | Needed for Phase 4 sentence mode.                |
| Inference speed                   | Per-guess latency.                               |
| Memory usage                      | Fits the deploy target.                          |
| Model size                        | Cold-start / image size.                         |
| Commercial / deployment license   | Must allow our hosting.                          |
| CPU viability                     | Avoid mandatory GPU cost.                        |
| OOV / rare-word handling          | Players will try unusual words.                  |
| Homonyms & context handling       | Same spelling, different meaning.                |

## Evaluation-set draft

Target **~20–30 answer words**, each with related words grouped by expected
closeness. Template and format live in
[`../ml/evaluation/word_similarity_eval.example.json`](../ml/evaluation/word_similarity_eval.example.json):

- `veryClose` — near-synonyms / tightly related.
- `medium` — loosely related, same domain.
- `unrelated` — should score low.
- `confusable` — surface-similar but semantically different (traps).

**Pass rule (per answer):** the model ranks `veryClose > medium > unrelated`
on average. Report per-model pass rate plus the criteria table above.

## Dimensionality reduction (for the 3D map)

- **PCA first** (simple, fast, deterministic; `scikit-learn`).
- **UMAP later** as a follow-up experiment — document the trade-offs
  (local vs. global structure, nondeterminism, extra dependency).
- ⚠️ Always note that a 3D projection is a **lossy view**; projected distance is
  not the true embedding similarity.

## Reproducibility

Record for every run (see `ml/README.md` and [../AGENTS.md](../AGENTS.md)):
seed(s), model name + version, library versions, data version, metric
definitions, and results. Commit the write-up here; never commit weights/data.

## Outcome

When a model is chosen, record the decision and numbers in this file, then add a
`SentenceTransformerEmbeddingService` conforming to the `EmbeddingService`
Protocol and switch `EMBEDDING_PROVIDER` — no caller changes required.
