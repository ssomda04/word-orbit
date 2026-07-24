# ml/evaluation

Evaluation sets and metrics for comparing candidate Korean embedding models.

## Evaluation-set format

See [`word_similarity_eval.example.json`](./word_similarity_eval.example.json)
for the starter template. Each answer word lists related words grouped by how
close they should be, so we can check whether a model ranks them sensibly:

- `veryClose` — near-synonyms / tightly related
- `medium` — loosely related, same domain
- `unrelated` — should score low
- `confusable` — surface-similar but semantically different (traps)

## How it's used

A model "passes" for an answer when it ranks `veryClose > medium > unrelated`
on average. Record per-model results (naturalness, speed, memory, size, license,
CPU-viability, OOV handling, homonyms) in
[../../docs/MODEL_EVALUATION.md](../../docs/MODEL_EVALUATION.md).

The target is ~20–30 answer words. The example file holds only a few entries as
a template — expanding it is the first ML issue (see the root README).
