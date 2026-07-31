# ml/evaluation

Evaluation sets and metrics for comparing candidate Korean embedding models.

## Evaluation-set format

[`word_similarity_eval.json`](./word_similarity_eval.json) is the versioned input
for real runs. [`word_similarity_eval.example.json`](./word_similarity_eval.example.json)
is schema guidance only. Each answer has four equally sized candidate groups:

- `veryClose`: synonyms, expressions for the same referent, or a very direct
  hypernym/hyponym relation.
- `related`: different referents that commonly share a topic or situation.
- `unrelated`: neither semantically related nor likely to co-occur in ordinary
  context.
- `surfaceTrap`: spelling, syllable, pronunciation, or morphology looks similar,
  but meaning is nearly unrelated. This is a form-bias diagnostic, not a
  similarity target.

## Metrics and interpretation

A run reports group statistics and these strict pairwise accuracies:

- `veryClose > unrelated`
- `related > unrelated`
- `veryClose > related`
- `veryClose > surfaceTrap`
- `related > surfaceTrap`

Top-k Precision and Recall treat only `veryClose` as relevant. The report also
counts `veryCloseInTopK`, `relatedInTopK`, and `surfaceTrapInTopK`. Rankings cover
only the candidates listed for each answer, not the future game vocabulary.

The v0.2 dataset has 20 answers and four candidates per group. Record model
results and broader operational criteria in
[../../docs/MODEL_EVALUATION.md](../../docs/MODEL_EVALUATION.md).

## Result versioning

Files under `results/fasttext/` were produced with the v0.1 relationship criteria.
Preserve them as historical output. New v0.2 runs must use a separate empty
directory such as `results/fasttext-v0.2/`; the reporter rejects a non-empty
output directory rather than overwriting it.
