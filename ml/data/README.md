# ml/data

Local datasets and snapshots live here. **Nothing in this folder is committed**
(except this README and `.gitkeep`) — see the root `.gitignore`.

Do not commit corpora, embeddings, model outputs, or `.npy`/`.bin` files. Keep a
note in [../../docs/MODEL_EVALUATION.md](../../docs/MODEL_EVALUATION.md) of where
each dataset came from and how to regenerate it.

## Korean Wiktionary game words

`game_words.txt` is generated from the official Korean Wiktionary
`pages-articles.xml.bz2` dump. Both the dump and generated word list are local
artifacts covered by the root `.gitignore`; do not commit either one.

From the repository root:

```powershell
uv run --project backend python .\ml\scripts\extract_wiktionary_words.py `
  --dump-path "C:\data\wiktionary\kowiktionary-latest-pages-articles.xml.bz2" `
  --output ".\ml\data\game_words.txt"
```

The extractor streams the bz2 XML, considers only explicit Korean language
sections with a major lexical part-of-speech heading, normalizes titles with
Unicode NFKC, retains only predicate lemmas ending in `다`, and prints exclusion
counts.

## Reviewable answer candidates

`answer_candidates.csv` is a generated, ignored review artifact. It keeps the
game vocabulary unchanged, excludes proper nouns and non-lexical entries from
answer candidacy, and marks explicit Wiktionary usage/domain labels plus words
of eight or more Hangul syllables for human review.

```powershell
uv run --project backend python .\ml\scripts\build_answer_candidates.py `
  --dump-path "C:\data\wiktionary\kowiktionary-latest-pages-articles.xml.bz2" `
  --vocabulary ".\ml\data\game_words.txt" `
  --output ".\ml\data\answer_candidates.csv"
```

No corpus-frequency data is used, so the CSV does not infer rarity. Only an
explicit Wiktionary rare-usage label can produce `explicit_rare_label`.

## Corpus-frequency analysis

Place a real, UTF-8 corpus-frequency CSV at an ignored local path such as
`ml/data/frequency/korean_frequency.csv`. Required columns are `word` and one
of `count` or `frequency`. Optional columns are `document_frequency`,
`frequency_rank`, and `source`.

```csv
word,count,document_frequency,frequency_rank,source
사람,123456,42000,17,corpus-name
```

Words are joined after Unicode NFKC normalization and trimming. Blank words are
ignored. Numeric values must be finite and non-negative. Duplicate normalized
words use `--duplicate-policy sum` by default, which is valid only for additive
counts or rows on the same frequency scale; use `--duplicate-policy error` when
that assumption does not hold. Input ranks are preserved as
`source_frequency_rank`, while `frequency_rank` is recomputed among matched
answer candidates with competition ties (`1, 1, 3`). Missing frequency remains
blank with `frequency_found=false` and is never treated as zero.

```powershell
uv run --project backend python .\ml\scripts\analyze_answer_frequency.py `
  --frequency ".\ml\data\frequency\korean_frequency.csv" `
  --candidates ".\ml\data\answer_candidates.csv" `
  --output ".\ml\data\answer_candidates_with_frequency.csv"
```

The command writes the joined CSV and prints coverage, distributions, rank
cutoffs, deterministic samples, and review-reason comparisons as JSON. It does
not choose a cutoff or create `answer_pool.txt`.
