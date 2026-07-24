# ml/scripts

Reproducible command-line scripts (data preparation, batch model evaluation).
Prefer promoting stable notebook logic into a script here so it can run
unattended and be diffed in review.

Each script should:

- accept a `--seed` and log it,
- print the model + library versions it used,
- write results/metrics somewhere referenced by
  [../../docs/MODEL_EVALUATION.md](../../docs/MODEL_EVALUATION.md).

(No scripts yet — this is the Phase 0 skeleton. See the "임베딩 모델 비교 및
평가셋 작성" issue in the README's *Next steps*.)
