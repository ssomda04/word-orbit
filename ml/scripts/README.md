# ml/scripts

Reproducible command-line scripts (data preparation, batch model evaluation).
Prefer promoting stable notebook logic into a script here so it can run
unattended and be diffed in review.

Each experiment script should:

- accept a `--seed` and log it,
- print the model + library versions it used,
- write results/metrics somewhere referenced by
  [../../docs/MODEL_EVALUATION.md](../../docs/MODEL_EVALUATION.md).

Data preparation scripts are deterministic and do not need a meaningless seed.
`extract_wiktionary_words.py` streams the official Korean Wiktionary bz2 dump;
see [`../data/README.md`](../data/README.md) for its invocation and output policy.
