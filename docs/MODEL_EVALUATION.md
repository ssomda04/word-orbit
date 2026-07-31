# Model Evaluation Plan

> **No final model is selected yet.** FastText has been evaluated and accepted
> as the first real baseline, and it is now wired into the backend as the
> `fasttext` provider alongside the default `DeterministicEmbeddingService`
> (a hash-based mock). Transformer models remain comparison candidates. This
> document is the plan for choosing the *final* Korean embedding model; the
> FastText wiring is a baseline, not that decision.

## Goal

Pick a Korean word/sentence embedding model that makes Contextle's similarity
scores feel natural to players, while being cheap enough to run (ideally CPU)
and license-clean for deployment.

## Baseline first

Before any comparison or fine-tuning:

1. Freeze an evaluation set and increment its version when membership changes.
2. Define metrics and a baseline. The deterministic mock is the trivial floor;
   the first off-the-shelf model is the real baseline.
3. Only consider fine-tuning if a pretrained model is measurably insufficient.

FastText with the Common Crawl/Wikipedia Korean binary is the first static-word
baseline to evaluate. This is a baseline choice, not a quality conclusion. It is
useful for CPU-oriented inference and subword OOV behavior, but a static vector
cannot distinguish different senses of the same spelling from sentence context.
Transformer contextual encoders remain comparison candidates.

## Candidate models (not endorsements)

Evaluate a spread of Korean-capable sentence/word encoders, for example:

- Korean FastText Common Crawl/Wikipedia static vectors
- `jhgan/ko-sroberta-multitask`
- `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- `BM-K/KoSimCSE-roberta`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Confirm current availability and licenses at evaluation time.

## Comparison criteria

| Criterion | Why it matters |
| --- | --- |
| Korean word-similarity naturalness | Core gameplay feel. |
| Sentence-embedding quality | Needed for Phase 4 sentence mode. |
| Inference speed | Per-guess latency. |
| Memory usage | Must fit the deployment target. |
| Model size | Cold-start and image size. |
| Commercial/deployment license | Must permit hosting. |
| CPU viability | Avoid mandatory GPU cost. |
| OOV/rare-word handling | Players will try unusual words. |
| Homonyms and context handling | Static and contextual models differ here. |

## Evaluation set (v0.2.0)

The versioned baseline contains 20 answer words with four candidates in each
group. The schema example is
[`../ml/evaluation/word_similarity_eval.example.json`](../ml/evaluation/word_similarity_eval.example.json),
while actual runs use `word_similarity_eval.json`.

- `veryClose`: synonyms, the same referent, or a very direct hypernym/hyponym.
- `related`: different referents that normally share a topic or situation.
- `unrelated`: no semantic or ordinary contextual connection.
- `surfaceTrap`: spelling, syllable, pronunciation, or morphology resembles the
  answer but meaning is nearly unrelated. This diagnoses form bias; it is not an
  expected similarity group.

Every run reports mean, population standard deviation, minimum, and maximum for
each group. It also reports strict pairwise accuracy, per answer and overall:

- `veryClose > unrelated`
- `related > unrelated`
- `veryClose > related`
- `veryClose > surfaceTrap`
- `related > surfaceTrap`

Precision@k and Recall@k use only `veryClose` as relevant. Counts for
`veryCloseInTopK`, `relatedInTopK`, and `surfaceTrapInTopK` make candidate-pool
behavior explicit. These top-k metrics rank only the 16 listed candidates for an
answer and are not equivalent to rank over the future full game vocabulary.

## FastText execution and result versions

The harness loads a local Facebook FastText `.bin` file and never downloads a
model. It uses raw cosine similarity in `[-1, 1]`. The binary model retains
character-subword OOV behavior; direct-vocabulary and subword-generated vectors
are counted separately when the library exposes membership.

Evaluation result directories under `ml/evaluation/results/` are **local
artifacts and are never committed** — a fresh clone has none of them. Whoever
runs the harness keeps their own output; the numbers that matter are transcribed
into this document. Earlier `results/fasttext/` output used the v0.1
relationship criteria, so where it still exists locally it must not be compared
against v0.2 runs. New runs must use a new empty directory, for example:

```powershell
uv run --project backend --extra embeddings python .\ml\scripts\evaluate_fasttext.py `
  --model-path "C:\models\cc.ko.300.bin" `
  --dataset ".\ml\evaluation\word_similarity_eval.json" `
  --output-dir ".\ml\evaluation\results\fasttext-v0.2" `
  --top-k 10 `
  --seed 42
```

The reporter refuses a non-empty output directory to protect previous results.

## Dimensionality reduction (future 3D map)

- Evaluate PCA first because it is simple, fast, and deterministic.
- Consider UMAP later and document local/global structure, nondeterminism, and
  its extra dependency.
- A 3D projection is a lossy view; projected distance is not true embedding
  similarity.

## Reproducibility

Record seed, model name/version, library versions, dataset version, metric
definitions, and results for every run. Commit the write-up, but never model
weights, local data, cached vectors, or generated evaluation result files.

## Outcome

FastText has been accepted as the first real baseline model, but no final model
has been selected yet.

The baseline result shows strong separation between related and unrelated words,
but weak separation between `veryClose` and `related`. The next step is to
compare contextual Transformer models using the same versioned evaluation set.

### Wiring the baseline into the backend

FastText is connected to the backend **as a baseline provider, ahead of the
Transformer comparison**, deliberately:

- The purpose of the wiring is to verify **service integration and
  runnability** — that a real model loads once at startup, scores guesses
  through the existing `EmbeddingService` seam, and survives deployment
  constraints. It is not a statement about answer quality.
- **The final model decision still requires** comparing Transformer candidates
  against this same versioned evaluation set. Nothing below changes because the
  provider now exists.
- The measured limits stay exactly as recorded — in particular
  `veryClose > related` at 51.56%, close to random.
- Selecting the final model is a separate change: add another
  `EmbeddingService` implementation and switch `EMBEDDING_PROVIDER`. No caller
  edits are needed, which is the point of the Protocol.

The default provider remains `mock`; tests and CI never load a model. See
[`../backend/README.md`](../backend/README.md) for `EMBEDDING_PROVIDER=fasttext`
and `FASTTEXT_MODEL_PATH`.

## FastText baseline result

### Run configuration

- Model: `cc.ko.300.bin`
- Library: `fasttext-wheel==0.9.2`
- Vector dimension: 300
- Dataset: `contextle-ko-word-similarity`
- Dataset version: `0.2.1`
- Answer words: 20
- Candidates per answer: 16
- Top-k: 10
- Seed: 42
- Environment: Windows, Python 3.12, CPU

### Group similarity

| Group | Count | Mean | Standard deviation | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| veryClose | 80 | 0.3991 | 0.1020 | 0.1746 | 0.6352 |
| related | 80 | 0.3935 | 0.0949 | 0.1318 | 0.5901 |
| unrelated | 80 | 0.1664 | 0.0802 | 0.0168 | 0.4192 |
| surfaceTrap | 80 | 0.2070 | 0.0794 | -0.0103 | 0.4028 |

### Pairwise accuracy

| Comparison | Accuracy |
| --- | ---: |
| veryClose > unrelated | 97.50% |
| related > unrelated | 98.44% |
| veryClose > related | 51.56% |
| veryClose > surfaceTrap | 95.00% |
| related > surfaceTrap | 96.56% |

### Top-k results

- Mean Precision@10: 0.39
- Mean Recall@10: 0.975
- veryClose in Top-10: 78 / 80
- related in Top-10: 79 / 80
- surfaceTrap in Top-10: 32 / 80

These Top-k values rank only the 16 candidates listed for each answer. They do
not represent rank over the future full game vocabulary.

### Timing

- Model load time: 8.54 seconds
- Evaluation excluding model load: 0.081 seconds
- Total run time: 8.63 seconds
- Approximate mean vector lookup time: 0.124 ms

The model should be loaded once at server startup and reused for user guesses.
Per-word vector lookup is fast enough for CPU-based interactive use.

### Vocabulary handling

- Total words: 320
- Direct vocabulary: 318
- Subword-generated: 2
- Failed: 0
- Vector failures: none

The binary FastText model successfully generated vectors for all evaluated
words, including two words handled through subword composition.

### Interpretation

FastText clearly separates semantically related words from unrelated words.
Both `veryClose > unrelated` and `related > unrelated` exceeded 97% accuracy.

It also separates meaningfully related words from surface-form traps with high
accuracy. `veryClose > surfaceTrap` reached 95.00%, and
`related > surfaceTrap` reached 96.56%.

However, FastText did not reliably distinguish `veryClose` from `related`.
The mean similarities were 0.3991 and 0.3935, and pairwise accuracy was only
51.56%. This indicates that the static embedding strongly reflects contextual
co-occurrence and topical association, not only synonym-level semantic
closeness.

For example, words such as `교사` and `학교` may rank above synonym-like words
for the answer `학생`, because they frequently occur in the same contexts.

### Strengths

- Fast CPU inference
- Pretrained Korean vocabulary
- Character-subword OOV handling
- High related-vs-unrelated separation
- High related-vs-surfaceTrap separation
- Simple deployment compared with Transformer encoders

### Limitations

- Weak distinction between synonym-level similarity and topical relatedness
- One fixed vector per word
- Cannot resolve homonyms using sentence context
- Sensitive to corpus frequency and co-occurrence patterns
- Current Top-k evaluation uses only a small candidate pool
- Full game-vocabulary ranking has not yet been measured

### Current decision

FastText is accepted as the first real baseline model for Contextle.

It is suitable for the initial word-guessing mode because it is fast, CPU-friendly,
and robust to many OOV words. It is not yet selected as the final model because
its `veryClose > related` performance is close to random.

Transformer-based contextual models should be evaluated with the same dataset
before the final model decision.

### Next steps

1. Build the full game vocabulary.
2. Measure rank over the full vocabulary.
3. Compare FastText with Korean sentence-transformer models.
4. Evaluate homonyms and context-sensitive inputs separately.
5. Decide the final model from that comparison, then add it as a further
   `EmbeddingService` implementation. The FastText baseline provider already in
   the backend does not pre-empt this choice.