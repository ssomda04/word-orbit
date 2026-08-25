"""FastText implementation of the `EmbeddingService` Protocol.

Why this is not a copy of the ML harness
----------------------------------------
`ml/` owns *evaluation*: datasets, group statistics, pairwise accuracy, top-k
ranking, CSV report bundles. None of that belongs in a request path, and the
backend never imports `ml/` (AGENTS.md). What carries over is only the behaviour
the harness established and `docs/MODEL_EVALUATION.md` recorded: load a local
`.bin` **once**, never download one, and score with **raw cosine similarity** in
[-1, 1] so service scores stay comparable to the published baseline numbers.

The class is deliberately split in two:

- ``__init__`` takes an already-loaded model object, so unit tests inject a fake
  and run with neither the ``fasttext`` library nor a multi-gigabyte model file;
- ``load()`` is the only place that touches the filesystem and the optional
  dependency.

Out of scope here (docs/ROADMAP.md): `rank`, vocabulary membership reporting,
and 3D projection — see ``project_3d``.

Only the standard library is imported at module level, so this module stays
importable (and unit-testable) in an environment without the ``fasttext`` extra.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class FastTextConfigurationError(RuntimeError):
    """The FastText provider cannot be built.

    Covers a missing ``FASTTEXT_MODEL_PATH``, a path that does not point at a
    readable file, the optional dependency not being installed, and a model the
    library refuses to load. Deliberately *not* an ``AppError``: this is a
    startup/configuration failure that should stop the process, not a per-request
    error mapped onto the documented HTTP error envelope.
    """


# Repeated in every configuration error so the fix is always one message away.
_SETUP_HELP = (
    "Set FASTTEXT_MODEL_PATH to the absolute path of a FastText .bin you have "
    "already downloaded yourself (e.g. cc.ko.300.bin), and install the loader "
    "with `uv sync --extra fasttext`. The server never downloads a model, and "
    "model files are never committed to this repository."
)


class FastTextEmbeddingService:
    """`EmbeddingService` backed by a local Facebook FastText binary model."""

    def __init__(self, model: Any) -> None:
        """Wrap an already-loaded model.

        Taking the model as a parameter (rather than a path) is what makes this
        class testable without the library or the real weights.
        """
        self._model = model

    @classmethod
    def load(cls, model_path: Path) -> FastTextEmbeddingService:
        """Load a local ``.bin`` model. Never downloads anything.

        Raises:
            FastTextConfigurationError: the path is missing, is not a file, the
                ``fasttext`` package is not installed, or the model fails to load.
        """
        if not model_path.exists():
            raise FastTextConfigurationError(
                f"FastText model file not found: {model_path}. {_SETUP_HELP}"
            )
        if not model_path.is_file():
            raise FastTextConfigurationError(
                f"FASTTEXT_MODEL_PATH points at a directory, not a model file: "
                f"{model_path}. {_SETUP_HELP}"
            )

        # Imported lazily so the module works in the default (extra-free) install.
        try:
            import fasttext
        except ImportError as exc:
            raise FastTextConfigurationError(
                "The FastText loader is not installed. From backend/, run "
                f"`uv sync --extra fasttext`, then retry. {_SETUP_HELP}"
            ) from exc

        try:
            model = fasttext.load_model(str(model_path))
        except Exception as exc:  # noqa: BLE001 - the C++ loader raises broadly.
            raise FastTextConfigurationError(
                f"Could not load FastText model {model_path.name!r}: {exc}. "
                f"{_SETUP_HELP}"
            ) from exc

        return cls(model)

    @staticmethod
    def _normalize(text: str) -> str:
        """Canonical form for a lookup.

        Must match ``app.domain.game.normalize_word`` (NFKC + strip): if the two
        diverged, a guess could equal the answer and still not score 1.0.
        Blank input raises, exactly like ``DeterministicEmbeddingService``, so the
        mock and the real provider stay interchangeable.
        """
        normalized = unicodedata.normalize("NFKC", text).strip()
        if not normalized:
            raise ValueError("cannot embed empty or whitespace-only text")
        return normalized

    def _vector(self, text: str) -> list[float]:
        """Return the raw (un-normalized) vector for ``text``.

        FastText composes character n-grams, so an out-of-vocabulary word still
        gets a vector — that is intended, and no error or log is produced for it.

        The failures below do not name the word. This method is called with the
        guess *and* with the hidden answer, and it cannot tell which it holds, so
        naming either would put the answer in a traceback and from there into the
        server log (AGENTS.md).

        Sanitizing our own message is not enough. This is the one place in the
        request path where a secret word is handed to code we do not control, and
        a native loader may quote its input in its own error. ``from exc`` would
        keep that text as ``__cause__``, and ``traceback`` renders a cause — so
        the word would reach the log anyway. The cause is therefore suppressed
        with ``from None``, which also stops it being rendered by anything that
        chains *this* exception further up, and the exception *type* is carried
        across in its place: enough to tell a corrupt model from a bad call,
        without quoting anything.
        """
        word = self._normalize(text)
        try:
            raw: Sequence[float] = self._model.get_word_vector(word)
        except Exception as exc:  # noqa: BLE001 - pybind11 raises broadly.
            raise ValueError(
                f"FastText could not produce a vector for the requested word "
                f"({type(exc).__name__})."
            ) from None

        vector = [float(value) for value in raw]
        if not vector:
            raise ValueError("FastText returned an empty vector for the requested word.")
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("FastText returned a non-finite vector for the requested word.")
        return vector

    def encode(self, text: str) -> list[float]:
        """Return the embedding vector, zero vectors included (see `similarity`)."""
        return self._vector(text)

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Vectors for many texts, order preserved.

        FastText exposes no batch equivalent of ``get_word_vector``
        (``get_sentence_vector`` has different semantics), so this is a loop.
        """
        return [self._vector(text) for text in texts]

    def similarity(self, first: str, second: str) -> float:
        """Raw cosine similarity in [-1.0, 1.0].

        A zero vector (possible when no character n-gram matches at all) makes
        cosine undefined. Rather than raising — which would turn an odd guess into
        a 500 — this returns ``0.0``, i.e. "unrelated", which is inside the
        documented range.
        """
        left = self._vector(first)
        right = self._vector(second)

        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        dot = sum(a * b for a, b in zip(left, right, strict=True))
        # Clamp to guard against tiny floating-point overshoot past [-1, 1].
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

    def project_3d(self, texts: Sequence[str]) -> list[list[float]]:
        """Not implemented for this provider.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            "3D projection is not implemented for the FastText provider. It is "
            "planned for Phase 2, using PCA or another dimensionality-reduction "
            "method chosen in docs/MODEL_EVALUATION.md (see docs/ROADMAP.md). "
            "This service returns no placeholder coordinates on purpose: "
            "meaningless positions would be read as real semantic distance. "
            "The game API is unaffected — `coordinate` stays null by contract "
            "(docs/API_SPEC.md)."
        )
