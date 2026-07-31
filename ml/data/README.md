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
