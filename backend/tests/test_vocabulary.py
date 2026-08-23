"""Vocabulary normalization, mirrored from the ML harness.

Every case here has a counterpart in ``ml/tests/test_rank_table.py``. The two
implementations must agree on the word set before they can agree on ranks, so
these are parity tests, not incidental unit tests.
"""

from pathlib import Path

import pytest

from app.domain.vocabulary import (
    VocabularyError,
    load_vocabulary,
    normalize_vocabulary,
    normalize_vocabulary_word,
)


def test_normalization_deduplicates_and_removes_blanks() -> None:
    """Ported from ml/tests/test_rank_table.py::test_normalization_deduplicates..."""
    assert normalize_vocabulary(["  Ａ  ", "A", "", " \t ", "Ｂ", " B "]) == ("A", "B")


def test_normalization_keeps_first_occurrence_order() -> None:
    assert normalize_vocabulary(["나", "가", "나", "다"]) == ("나", "가", "다")


def test_nfkc_is_applied() -> None:
    # Fullwidth and compatibility forms collapse onto their canonical form.
    assert normalize_vocabulary_word("Ａ") == "A"
    assert normalize_vocabulary_word("  바다\n") == "바다"


def test_blank_normalizes_to_empty_string_rather_than_raising() -> None:
    """Callers drop blanks; only a non-string is an error (as in the ML harness)."""
    assert normalize_vocabulary_word("   ") == ""


def test_non_string_entry_is_rejected() -> None:
    with pytest.raises(VocabularyError, match="must be strings"):
        normalize_vocabulary_word(None)  # type: ignore[arg-type]


def test_internal_whitespace_is_preserved() -> None:
    """A dictionary headword may contain a space.

    ``app.domain.game.normalize_word`` rejects such a word as a *guess*, but the
    vocabulary must keep it: dropping it here would shift the rank of every word
    below it and break parity with the ML harness.
    """
    assert normalize_vocabulary(["고유 명사"]) == ("고유 명사",)


def test_long_word_is_preserved() -> None:
    """The 50-character guess cap is a guess rule, not a vocabulary rule."""
    long_word = "가" * 80
    assert normalize_vocabulary([long_word]) == (long_word,)


def test_vocabulary_file_is_utf8_bom_safe(tmp_path: Path) -> None:
    """Ported from ml/tests/test_rank_table.py::test_vocabulary_file_is_utf8_bom_safe..."""
    vocabulary_path = tmp_path / "game_words.txt"
    vocabulary_path.write_text("﻿  Ａ  \nA\n\nＢ\n", encoding="utf-8")

    assert load_vocabulary(vocabulary_path) == ("A", "B")


def test_missing_vocabulary_file_has_clear_error(tmp_path: Path) -> None:
    """Ported from ml/tests/test_rank_table.py::test_missing_vocabulary_file..."""
    with pytest.raises(VocabularyError, match="Vocabulary file not found"):
        load_vocabulary(tmp_path / "missing.txt")


def test_non_utf8_vocabulary_file_has_clear_error(tmp_path: Path) -> None:
    vocabulary_path = tmp_path / "broken.txt"
    vocabulary_path.write_bytes(b"\xff\xfe\x00broken")

    with pytest.raises(VocabularyError, match="not valid UTF-8"):
        load_vocabulary(vocabulary_path)


def test_directory_instead_of_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(VocabularyError, match="Could not read vocabulary file"):
        load_vocabulary(tmp_path)
