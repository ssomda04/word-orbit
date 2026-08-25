"""Rank a guess by comparing it to every word in the vocabulary.

Rank policy (identical to ``ml/src/contextle_eval/rank_table.py``)
-----------------------------------------------------------------
Given answer ``a`` and vocabulary ``V``, the ML harness builds the full ordering

    ranked = [a] + sorted(V \\ {a}, key=lambda w: (-similarity(w), w))

and assigns ``rank = position + 1``. Three consequences drive this module:

1. **Ranks are dense and unique.** ``enumerate`` over a sorted list gives every
   word its own rank, so tied similarities do *not* share a rank — they are
   separated by the word itself, ascending by code point.
2. **The answer is always rank 1**, by construction rather than by score. It
   holds even when the answer ties with other words, and even when its vector is
   zero and its own similarity is therefore ``0.0`` rather than ``1.0``.
3. **Whether ``a`` is in ``V`` changes nothing for other words.** ``a`` is
   excluded from the sorted tail either way, so this module never needs to add
   the answer to the vocabulary.

Computing rank without materializing the ordering
-------------------------------------------------
Building that list per game would cost a sort over the whole vocabulary and a
per-game dictionary of every word's rank — memory that grows with the number of
games played. Instead, ``rank_of`` counts the words that would sort ahead:

    rank(w) = 2 + |{v ∈ V\\{a} : sim(v) > sim(w)}|
                + |{v ∈ V\\{a} : sim(v) = sim(w) ∧ v < w}|

The leading 2 is the answer's slot plus the shift from 0-based to 1-based. This
is exact, not an approximation: because words are unique after deduplication,
``(-similarity, word)`` is a *strict total order*, so a word's position in the
tail is precisely the number of words that compare less than it.

Note what this is not: ``|{v : sim(v) > sim(w)}| + 1`` is competition ranking,
which gives tied words the same rank and then skips. That is a different policy
and would not match the ML harness.

The second term is the tie-break. Comparing strings for every vocabulary entry
on every guess would be slow, so the lexicographic order is precomputed once as
``_lex``: an integer array where ``_lex[i] < _lex[j]`` exactly when
``words[i] < words[j]``. It is derived with Python's own ``sorted``, so the
comparison semantics are the language's, not NumPy's.

Cost and memory
---------------
Built once: a unit-normalized ``(N, D)`` matrix, an ``N`` integer array, and a
word→index dict. Per guess: two vectorized scans over ``N``. Per *answer* (not
per game): one cached similarity array, bounded by ``cache_size``. Nothing is
stored per game, so memory does not grow as games accumulate.
"""

import threading
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from app.domain.vocabulary import normalize_vocabulary_word
from app.services.embedding.base import EmbeddingService
from app.services.ranking.base import NonFiniteEmbeddingError, RankingError

# float64 keeps the arithmetic close to the ML harness, which works in Python
# floats. See `dtype` in __init__ for the trade-off at large vocabularies.
DEFAULT_DTYPE = np.float64

# How many answers' similarity arrays to keep. Each costs N * itemsize bytes.
DEFAULT_CACHE_SIZE = 32


class VectorRankProvider:
    """Ranks a guess against a fixed vocabulary using an ``EmbeddingService``."""

    def __init__(
        self,
        words: Sequence[str],
        embedder: EmbeddingService,
        *,
        cache_size: int = DEFAULT_CACHE_SIZE,
        dtype: np.dtype | type = DEFAULT_DTYPE,
    ) -> None:
        """Embed the whole vocabulary once and precompute the tie-break order.

        Args:
            words: An already-normalized, deduplicated vocabulary
                (``app.domain.vocabulary.normalize_vocabulary``). Passing raw
                words would produce ranks over a different word set than the ML
                harness computes.
            embedder: Used for the vocabulary now and for each answer later. The
                same instance must be used for both, or similarities would come
                from two different vector spaces.
            cache_size: Number of answers whose similarity array is retained.
            dtype: Element type of the vector matrix. ``float64`` matches the ML
                harness's Python-float arithmetic most closely. ``float32``
                halves the memory — worthwhile past a few hundred thousand words
                — at the cost of exact agreement on coincidentally-equal
                similarities; see the module docstring in ``tests/test_ranking.py``.

        Raises:
            RankingError: the vocabulary is empty, a word is not a string, a
                vector is empty, or vector lengths differ.
            NonFiniteEmbeddingError: a vector or norm is NaN or infinity.
        """
        if cache_size < 1:
            raise RankingError("cache_size must be at least 1.")
        if not words:
            raise RankingError("Vocabulary must not be empty.")

        self._words: tuple[str, ...] = tuple(words)
        self._index: dict[str, int] = {word: i for i, word in enumerate(self._words)}
        if len(self._index) != len(self._words):
            raise RankingError(
                "Vocabulary contains duplicates; normalize it with "
                "app.domain.vocabulary.normalize_vocabulary first."
            )

        self._dtype = np.dtype(dtype)
        self._unit = self._build_unit_matrix(embedder)
        self._dimension = int(self._unit.shape[1])
        self._lex = self._build_lexicographic_order(self._words)

        self._embedder = embedder
        self._cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        # Sync route handlers run in a thread pool, so two guesses can reach the
        # cache at once — same reasoning as InMemoryGameRepository.
        self._lock = threading.Lock()

    # --- Construction -------------------------------------------------------

    def _build_unit_matrix(self, embedder: EmbeddingService) -> np.ndarray:
        """Embed every word and scale each row to unit length.

        Rows are pre-divided so a later similarity is a plain dot product. A
        zero-length row is left as zeros rather than divided (which would give
        NaN), and its dot product is then exactly ``0.0`` — the same value the ML
        harness returns from its explicit zero-vector branch.
        """
        vectors: list[Sequence[float]] = []
        dimension: int | None = None
        for word in self._words:
            try:
                vector = embedder.encode(word)
            except Exception as exc:
                raise RankingError(
                    f"Embedding model could not produce a vector for {word!r}: {exc}"
                ) from exc
            if len(vector) == 0:
                raise RankingError(f"Embedding model returned an empty vector for {word!r}.")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise RankingError(
                    f"Embedding vector dimensions differ: first={dimension}, "
                    f"second={len(vector)} (at {word!r})."
                )
            vectors.append(vector)

        matrix = np.asarray(vectors, dtype=self._dtype)
        if not np.isfinite(matrix).all():
            raise NonFiniteEmbeddingError("Embedding vector contains NaN or infinity.")

        norms = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(norms).all():
            raise NonFiniteEmbeddingError("Embedding norm is NaN or infinity.")

        unit = np.zeros_like(matrix)
        non_zero = norms > 0.0
        unit[non_zero] = matrix[non_zero] / norms[non_zero, None]
        return unit

    @staticmethod
    def _build_lexicographic_order(words: tuple[str, ...]) -> np.ndarray:
        """Map each word index to its position in lexicographic order.

        Uses Python's ``sorted`` on ``str`` so the comparison is the same one the
        ML harness's sort key performs. Only the relative order is used, so the
        exact positions never leak into a rank.
        """
        by_word = sorted(range(len(words)), key=lambda i: words[i])
        order = np.empty(len(words), dtype=np.int64)
        order[np.asarray(by_word, dtype=np.int64)] = np.arange(len(words), dtype=np.int64)
        return order

    # --- Similarity ---------------------------------------------------------

    def _similarities(self, answer: str) -> np.ndarray:
        """Cosine similarity of ``answer`` against every vocabulary word.

        Cached per answer, so replaying a game or running many games on the same
        answer costs one vocabulary-wide pass in total, not one per guess.
        """
        with self._lock:
            cached = self._cache.get(answer)
            if cached is not None:
                self._cache.move_to_end(answer)
                return cached

        similarities = self._compute_similarities(answer)

        with self._lock:
            self._cache[answer] = similarities
            self._cache.move_to_end(answer)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return similarities

    def _compute_similarities(self, answer: str) -> np.ndarray:
        """Embed the answer and score it against the whole vocabulary.

        None of the failures below name the answer. They are raised while a game
        is in progress, and an unhandled exception's message reaches the server
        log through the traceback — which would put the hidden word there
        (AGENTS.md: the answer is never logged). The word adds nothing anyway:
        every one of these says the model is broken, not that some particular
        input was unusual. The cause is chained, so the underlying error and its
        stack are still in the traceback.
        """
        try:
            raw = self._embedder.encode(answer)
        except Exception as exc:
            raise RankingError(
                "Embedding model could not produce a vector for the ranking answer."
            ) from exc
        if len(raw) == 0:
            raise RankingError("Embedding model returned an empty vector for the ranking answer.")
        if len(raw) != self._dimension:
            # Sizes are structural, not content: they identify the mismatch
            # without identifying the word.
            raise RankingError(
                f"Embedding vector dimensions differ: vocabulary={self._dimension}, "
                f"ranking answer={len(raw)}."
            )

        vector = np.asarray(raw, dtype=self._dtype)
        if not np.isfinite(vector).all():
            raise NonFiniteEmbeddingError("Embedding vector contains NaN or infinity.")

        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm):
            raise NonFiniteEmbeddingError("Embedding norm is NaN or infinity.")
        if norm == 0.0:
            # The ML harness returns 0.0 for every pair once either norm is zero,
            # so a zero answer makes the whole vocabulary tie and rank falls back
            # to lexicographic order.
            return np.zeros(len(self._words), dtype=self._dtype)

        similarities = self._unit @ (vector / norm)
        if not np.isfinite(similarities).all():
            raise NonFiniteEmbeddingError("Cosine similarity is NaN or infinity.")
        # Guard against floating-point overshoot past [-1, 1], as the ML harness
        # does with max/min.
        return np.clip(similarities, -1.0, 1.0)

    # --- Ranking ------------------------------------------------------------

    def rank_of(self, answer: str, word: str) -> int | None:
        """Return the 1-based rank of ``word`` for ``answer``, or ``None``.

        ``None`` means ``word`` is outside the vocabulary. That is a normal
        result — the ML harness's ``RankTable.rank_of`` returns ``None`` for the
        same input — and surfaces as ``rank: null`` in the API.

        Raises:
            RankingError: the answer is blank, or a vector cannot be produced.
            NonFiniteEmbeddingError: a vector, norm, or similarity is not finite.
        """
        normalized_answer = normalize_vocabulary_word(answer)
        if not normalized_answer:
            raise RankingError("Answer must not be empty after normalization.")

        normalized_word = normalize_vocabulary_word(word)
        if not normalized_word:
            return None
        # Rank 1 by construction, before any vector work: true even when the
        # answer ties with other words or its own vector is zero.
        if normalized_word == normalized_answer:
            return 1

        position = self._index.get(normalized_word)
        if position is None:
            return None

        similarities = self._similarities(normalized_answer)
        similarity = similarities[position]
        lexical = self._lex[position]

        ahead = int(np.count_nonzero(similarities > similarity))
        tied_ahead = int(np.count_nonzero((similarities == similarity) & (self._lex < lexical)))

        # The answer occupies rank 1 and is not part of the sorted tail, so its
        # own entry must not be counted when it happens to be in the vocabulary.
        answer_position = self._index.get(normalized_answer)
        if answer_position is not None:
            answer_similarity = similarities[answer_position]
            if answer_similarity > similarity:
                ahead -= 1
            elif answer_similarity == similarity and self._lex[answer_position] < lexical:
                tied_ahead -= 1

        return 2 + ahead + tied_ahead

    # --- Introspection ------------------------------------------------------

    @property
    def vocabulary_size(self) -> int:
        return len(self._words)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def cached_answer_count(self) -> int:
        """Answers currently holding a similarity array. Never exceeds ``cache_size``."""
        with self._lock:
            return len(self._cache)
