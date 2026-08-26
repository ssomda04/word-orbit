# Rank Artifact Format

A **rank artifact root** holds, for every answer the game may choose, that
answer's similarity and rank against every word of one canonical vocabulary —
computed once, offline, by a model the game server never loads.

It is a seam between two areas, like [API_SPEC.md](./API_SPEC.md):

- **ML writes it.** `ml/src/contextle_eval/rank_artifact.py`, via
  `ml/scripts/build_rank_artifacts.py`.
- **The backend reads it.** `backend/app/services/scoring/artifact/`.

The backend never imports `ml` ([AGENTS.md](../AGENTS.md)), so these are two
independent implementations of one format. This document describes that format
and, separately, what the backend additionally requires before it will serve
from a root. **It is not yet a single validated source of truth for both sides** —
see [Compatibility notes](#compatibility-notes).

> **Status.** The format is implemented on both sides. No production root exists
> yet: the answer pool is provisional and the final one waits on corpus work
> (see [MODEL_EVALUATION.md](./MODEL_EVALUATION.md)). The backend reader is not
> wired into the game — there is no provider selection, no cache, and no scorer.

---

## 1. On-disk format

What `write_artifact_root()` produces.

```text
<artifact root>/
├── manifest.json
├── vocabulary.txt
└── artifacts/
    └── <artifact_id[:2]>/
        └── <artifact_id>/
            ├── similarity.npy
            └── rank.npy
```

There is no per-answer metadata file: everything about an answer lives in
`manifest.json`.

### 1.1 `artifact_id`

```text
artifact_id = sha256( NFKC(answer).strip().encode("utf-8") ).hexdigest()
```

A deterministic identifier, used as the directory name and recorded in the
manifest. The two-character shard directory keeps a listing usable at a few
thousand answers and means nothing else.

> **What the hash is for.** It keeps a plaintext answer out of filesystem paths,
> and therefore out of anything that echoes a path — a stack trace, a log line,
> a directory listing. **It is not cryptographic secrecy.** The manifest in the
> same directory lists every answer in plain text, and the answer pool is a
> known, enumerable word list, so anyone holding the root can reverse an id
> trivially. Treat the root as server-side data; treat the hash as hygiene.
>
> It also covers paths only. A library that quotes a file's *contents* back into
> an error message can still surface an answer from a path that never named one
> — see [2.5](#25-secrecy).

### 1.2 `vocabulary.txt`

The canonical word list that every stored array is indexed by. `similarity[i]`
and `rank[i]` describe `vocabulary.txt` line `i` (0-based), so **line order is
part of the contract**.

Canonical form, exactly:

- UTF-8, **no** byte-order mark;
- one word per line, `\n` separated, **with** a trailing newline;
- each word already NFKC-normalized and stripped;
- no blank lines, no duplicates after normalization.

Equivalently: the file is byte-identical to
`"".join(f"{word}\n" for word in words).encode("utf-8")` for its own normalized
contents. `manifest.vocabulary.sha256` is the SHA-256 of those exact bytes.

### 1.3 `manifest.json`

```json
{
  "schema_version": "1.0",
  "artifact_id_algorithm": "sha256-nfkc-utf8",
  "embedding_model": { "name": "fasttext-cc-ko-300", "source": "cc.ko.300.bin" },
  "vocabulary": { "path": "vocabulary.txt", "size": 59582, "sha256": "604fcf…" },
  "similarity_dtype": "float32",
  "rank_dtype": "uint16",
  "ranking_policy": {
    "metric": "cosine",
    "answer_rank": 1,
    "order": "similarity_desc",
    "tie_break": "lexical"
  },
  "answers": {
    "바다": {
      "artifact_id": "1ea5c356…",
      "answer_vocab_index": 17,
      "similarity_path": "artifacts/1e/1ea5c356…/similarity.npy",
      "rank_path": "artifacts/1e/1ea5c356…/rank.npy"
    }
  }
}
```

| Field | Meaning |
| --- | --- |
| `schema_version` | Root layout version. `"1.0"`. |
| `artifact_id_algorithm` | How `artifact_id` is derived. `"sha256-nfkc-utf8"`. |
| `embedding_model.name` / `.source` | Which model produced the numbers, for auditing. |
| `vocabulary.path` | Always `"vocabulary.txt"`; the vocabulary travels inside the root. |
| `vocabulary.size` / `.sha256` | Must match `vocabulary.txt`. |
| `similarity_dtype` | `float32` or `float16`. |
| `rank_dtype` | Narrowest unsigned type that reaches the vocabulary size: `uint16`, `uint32`, or `uint64`. |
| `ranking_policy` | The ordering the ranks were produced by. |
| `answers` | Answer word → entry. Keys are canonical vocabulary words. |

All answers in one root share `similarity_dtype` and `rank_dtype`.

#### `answers` keys

An `answers` key is a **canonical vocabulary entry**: NFKC-normalized, stripped,
and present in `vocabulary.txt`. It is not a display form and not an arbitrary
label.

This matters because the key is the runtime lookup key. The server normalizes a
guess with the same NFKC + strip rule before scoring it, and finds the answer's
entry by that normalized string. A key in any other form — with a stray space,
in a compatibility form, or absent from the vocabulary — describes an answer the
server can never look up, and the root would carry an artifact that no game
could ever use.

### 1.4 Arrays

Both are one-dimensional, length `vocabulary.size`, saved with
`allow_pickle=False`.

| File | dtype | Contents |
| --- | --- | --- |
| `similarity.npy` | `similarity_dtype` | Cosine similarity to the answer, in `[-1.0, 1.0]`. The answer's own slot is exactly `1.0`. |
| `rank.npy` | `rank_dtype` | A permutation of `1..N`. The answer's own slot is `1`. |

### 1.5 Ranking policy

For answer `a` over vocabulary `V`:

```text
ranked = [a] + sorted(V \ {a}, key=lambda w: (-similarity(w), w))
rank(w) = position of w in ranked, 1-based
```

Ranks are **dense and unique**: words with equal similarity do not share a rank,
they are separated by the word itself in ascending code-point order. The answer
is rank 1 by construction, not by score. This is the same policy
`docs/API_SPEC.md` documents for the `rank` field, and the same one
`ml/src/contextle_eval/rank_table.py` implements directly.

### 1.6 Size

At the current game vocabulary of 59,582 words:

| | Bytes | |
| --- | ---: | --- |
| `similarity.npy` (float32) | 238,328 | `59,582 × 4` |
| `rank.npy` (uint16) | 119,164 | `59,582 × 2` |
| **per answer** | **≈ 349 KB** | |
| 1,949 provisional answers | ≈ 664 MB | |

`float16` similarities would make it ≈ 233 KB per answer (≈ 443 MB total), at the
cost of similarity precision. Ranks are stored independently, so they stay exact
either way.

---

## 2. Backend consumption requirements

The backend reads a root as **untrusted input**. It cannot re-derive these
numbers — it has no model — so anything it does not check, it assumes. Beyond
parsing the format above, it enforces the following, and refuses the root
otherwise.

### 2.1 At load time (whole root)

| Check | Why |
| --- | --- |
| `schema_version` in a supported set | A new layout may reorder or reinterpret the arrays. |
| `artifact_id_algorithm` exact match | A different rule addresses different files. |
| `ranking_policy` **exact equality**, including no extra keys | Ranks mean something else under a different policy. |
| `embedding_model.name` / `.source` non-empty strings | Auditability. |
| `vocabulary.path == "vocabulary.txt"` | The vocabulary is never a pointer elsewhere. |
| `vocabulary.txt` is canonical, **BOM rejected** | See [2.3](#23-why-the-vocabulary-reader-is-strict). |
| `vocabulary.size` matches the file | |
| `vocabulary.sha256` matches the file | The strongest single statement that the arrays and the word list belong together. |
| `similarity_dtype` ∈ `{float32, float16}` | Enumerated, not "any float": `float64` would silently double every artifact. |
| `rank_dtype` ∈ `{uint16, uint32, uint64}` | `uint8` cannot address a real game vocabulary. |
| `rank_dtype` reaches `vocabulary.size` | A root relabelled from a smaller vocabulary would wrap. |
| `answers` is an object and **not empty** | A root that can serve nothing fails at startup, not at the first game. |
| each key is a canonical vocabulary word | The key is the runtime lookup key; a key a normalized guess cannot equal is an unreachable answer. |
| `artifact_id` equals sha256 of its key | |
| `answer_vocab_index` is an `int` (not `bool`) and matches the vocabulary | JSON `true` must not pass as index 1. |
| `similarity_path` / `rank_path` equal the paths recomputed from `artifact_id` | See [2.2](#22-paths-are-derived-never-accepted). |
| both referenced files exist | Missing files fail at startup, not mid-game. |

**Array contents are not read at load time.** A production root is hundreds of
megabytes; reading it all would add seconds to startup for data most of which no
game will use.

### 2.2 Paths are derived, never accepted

The manifest records two paths per answer, but the backend recomputes them from
`artifact_id` and compares for exact equality. **The manifest cannot choose an
arbitrary artifact path**: absolute paths, `..` traversal, and alternate
spellings of the same path all fail that one comparison, with no rule of their
own. The path the backend opens is always the canonical hash-based relative one.

That is a statement about the *path string*, not about where the bytes finally
come from. If the canonical location is itself a symlink, `is_file()` and
`np.load` follow it like any other program would, and the data can live outside
the root. Resolving links and confining them to the root is not implemented, and
is out of scope for the current threat model: an artifact root is server-side
data an operator installs, in the same trust position as `FASTTEXT_MODEL_PATH`.
The check exists to stop a *manifest* from redirecting the server, not to
sandbox a root against whoever wrote it.

### 2.3 Why the vocabulary reader is strict

The backend has a forgiving vocabulary loader for the operator-supplied
`VOCABULARY_PATH` (`app/domain/vocabulary.py`): it reads `utf-8-sig`, and drops
blanks and duplicates. That is right for a hand-maintained file.

It is wrong here. `vocabulary.txt` is *part of an artifact* — the exact bytes
whose hash the manifest records and whose line order every array is indexed by.
Anything a reader silently repairs becomes a difference between what the arrays
were built against and what the server believes they were built against.

A BOM is the sharpest case. `utf-8-sig` strips it; strict UTF-8 decodes it to
U+FEFF, which is not whitespace, so it survives `strip()` and merges into the
first word as `'\ufeff<first word>'` — an entry no guess can ever match, and one
that shifts nothing else, so the corruption is invisible. The backend rejects a
leading BOM outright. The *normalization policy* is shared with
`app/domain/vocabulary.py`; only the file reading differs.

### 2.4 When an answer is actually loaded

Ordinary `np.load(..., allow_pickle=False)`, fully into memory. **Not** `mmap`:
at ~349 KB an artifact is smaller than the mapping overhead, validation touches
every element anyway, and a mapped file cannot be replaced on Windows while the
server runs.

| Check | Why |
| --- | --- |
| shape `(vocabulary.size,)`, dtype matches the manifest | The arrays belong to this root. |
| every similarity finite | A NaN sorts below everything without looking wrong. |
| every similarity within `[-1.0, 1.0]` | That range is the published API contract for `similarity`, not merely a property of cosine. |
| `similarity[answer_vocab_index] == 1.0` **exactly** | Otherwise the winning guess would not report a perfect score. |
| `rank[answer_vocab_index] == 1` | |
| ranks are a permutation of `1..N` | The strongest cheap statement that the file is intact. |

The answer similarity is compared **exactly**, with no tolerance: the writer
assigns a literal `1.0` before casting, and `1.0` is exactly representable in
both `float32` and `float16`. A tolerance here would only widen what counts as a
valid artifact.

### 2.5 Secrecy

`manifest.json` holds answers in plain text. That is fine as server-side data,
and it is why the rest of this section exists: the answer of a game in progress
must never reach a client **or the server log**
([AGENTS.md](../AGENTS.md), [API_SPEC.md](./API_SPEC.md)).

The backend reader therefore:

- **never interpolates an answer** into an exception message — messages identify
  a failure by `artifact_id`, or by the entry's position when the id itself is
  what is wrong;
- has **no "answer not found" exception**: a missing answer is reported by
  returning `None`, which cannot carry a word;
- **sanitizes the `np.load` boundary**: neither numpy's message nor numpy's
  exception is carried out of it (below).

This is the same policy the rest of the backend follows for the answer word, and
it is covered by the same style of regression test: every rejection is checked
against the answer in raw, `repr`, and `unicode_escape` form.

#### The `np.load` boundary

Hash-only paths keep an answer out of a *filename*. They say nothing about a
file's *contents*, and numpy quotes contents: a `.npy` whose header will not
parse comes back as `Cannot parse header: {!r}`, `Header is not a dictionary:
{!r}`, `Header does not contain the correct keys: {!r}`, or `descr is not a
valid dtype descriptor: {!r}` — the file, verbatim, inside the message.

So a broken array that happens to hold an answer would reach the server log
twice: once through an interpolated message, and once through the rendered
`__cause__`, since a load failure mid-game escapes into `app.main`'s
`INTERNAL_ERROR` handler and is logged with its traceback. The backend therefore
raises `Could not load artifact <artifact_id> (<ExceptionType>).` and chains
`from None`, keeping the two things an operator needs — which artifact, and what
kind of failure — and dropping everything numpy chose to say. It is the same
treatment `embedding/fasttext_service.py` gives the native FastText boundary.

The catch is deliberately broad, for the same reason it exists: numpy picks its
own exception types, and at least one of them (`tokenize.TokenError`, from a
header the tokenizer cannot finish reading) is outside
`(OSError, ValueError, EOFError)` and would otherwise escape unsanitized.

### 2.6 What the backend does not validate

The list in [2.1](#21-at-load-time-whole-root) and [2.4](#24-when-an-answer-is-actually-loaded)
is exhaustive. One limit is worth stating plainly, so that "the backend
validated the root" is not read as more than it is.

**`ranking_policy` is checked as a declaration, not as a property of the data.**
The backend requires the manifest's `ranking_policy` to match the policy it
serves, exactly. It does **not** re-sort the similarities to confirm that
`rank.npy` was actually produced by that policy — no `O(N log N)` recomputation
happens on the load path, at startup or per answer.

So the arrays are checked for *integrity*, not for *agreement with each other*:

| Checked | Not checked |
| --- | --- |
| similarity shape, dtype, finiteness, range `[-1, 1]` | that rank order follows descending similarity |
| `similarity[answer] == 1.0` | that ties are broken lexically |
| rank shape and dtype | that `ranking_policy` describes how `rank.npy` was built |
| `rank[answer] == 1` | |
| ranks are a permutation of `1..N` | |

A root whose ranks were sorted the wrong way round would pass every backend
check, because each array is individually well-formed. What the checks do catch
is a truncated, corrupted, mismatched, or mis-typed file — the failures that
actually occur when data is copied, regenerated, or partially written.

Full semantic consistency between similarities and ranks is the **producer's**
responsibility, and is the natural job of the planned golden
cross-implementation fixture: a committed root that both `ml/tests` and
`backend/tests` read, where the expected ranks can be asserted directly against
known similarities without either side recomputing them at runtime.

---

## Compatibility notes

Both sides implement §1 and agree on it. The backend additionally enforces the
checks below, which the ML reader does not as of `37ca3d7`. They are recorded
here rather than hidden, and each is a candidate for promotion into §1 once both
sides check it.

| Check | ML reader | Backend reader |
| --- | --- | --- |
| `manifest.answers` empty object | accepted | **rejected** |
| similarity outside `[-1, 1]` | accepted (clipped when written, not re-checked on read) | **rejected** |
| `similarity[answer] != 1.0` | accepted (assigned when written, not re-checked on read) | **rejected** |
| `vocabulary.txt` with a leading BOM | accepted; U+FEFF merges into the first word | **rejected** |
| `similarity_dtype` / `rank_dtype` | any float / any unsigned kind | **enumerated sets** |
| answer word in a failure message | present (`load_artifact_root_answer`) | **never** |

None of these blocks the backend: it is the consuming side, and a consumer that
trusts a producer's invariants has no invariants of its own.

**Planned.** A small **golden micro fixture** — a real, few-kilobyte root
committed to the repository and read by both `ml/tests` and `backend/tests` — is
agreed as a separate joint change after the ML validation fixes land. Because
the backend cannot import `ml`, a shared file that both sides parse is the only
way to catch format drift automatically; until it exists, this document and the
two test suites are what keep the implementations aligned.

---

## Changing the format

1. Discuss it — this is a cross-area contract.
2. Update this document first.
3. Update `ml/src/contextle_eval/rank_artifact.py` and
   `backend/app/services/scoring/artifact/` together, and bump
   `schema_version` if the change is not backward compatible.
4. A root is immutable once written: a new version is a new directory, never an
   edit in place.
