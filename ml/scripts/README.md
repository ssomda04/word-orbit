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
`build_answer_candidates.py` reuses that dump and vocabulary to create an ignored
CSV for human answer-pool review; it does not create a final answer pool.
`analyze_answer_frequency.py` joins a caller-supplied real corpus-frequency CSV,
writes an ignored analysis CSV, and prints distribution and coverage statistics;
it never imputes missing frequency or selects an answer-pool cutoff.
