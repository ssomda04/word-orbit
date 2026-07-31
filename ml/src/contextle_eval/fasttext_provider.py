"""Adapter for Facebook FastText binary models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class FastTextLoadError(RuntimeError):
    """Raised when a FastText model cannot be loaded."""


class FastTextVectorProvider:
    """Expose word vectors and vocabulary status from a loaded FastText model."""

    library_name = "fasttext-wheel"

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def load(cls, model_path: Path) -> FastTextVectorProvider:
        """Load a local Facebook ``.bin`` model without downloading anything."""
        if not model_path.is_file():
            raise FastTextLoadError(
                f"FastText model file not found: {model_path}. "
                "Download cc.ko.300.bin separately and pass its local path with --model-path."
            )
        try:
            import fasttext
        except ImportError as exc:
            raise FastTextLoadError(
                "The optional FastText loader is not installed. From backend/, run "
                "`uv sync --extra embeddings`, then retry the command."
            ) from exc

        try:
            return cls(fasttext.load_model(str(model_path)))
        except Exception as exc:
            raise FastTextLoadError(
                f"Could not load FastText model {model_path.name!r}: {exc}"
            ) from exc

    def vector(self, word: str) -> list[float]:
        """Return a finite vector, including subword-composed OOV vectors."""
        try:
            raw: Sequence[float] = self._model.get_word_vector(word)
            vector = [float(value) for value in raw]
        except Exception as exc:
            raise ValueError(f"FastText could not produce a vector for {word!r}: {exc}") from exc
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError(f"FastText returned an empty or non-finite vector for {word!r}.")
        return vector

    def vocabulary_status(self, word: str) -> bool | None:
        """Return direct-vocabulary membership, or ``None`` when unavailable."""
        get_word_id = getattr(self._model, "get_word_id", None)
        if get_word_id is None:
            return None
        try:
            return int(get_word_id(word)) != -1
        except (TypeError, ValueError):
            return None
