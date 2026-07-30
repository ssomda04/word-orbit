"""Answer-word source for single-player games.

PLACEHOLDER. This is a small hard-coded list, not a dataset — no word data files
are committed (see AGENTS.md). Selection sits behind ``AnswerSelector`` so a
real source (an ML-curated list, a DB table) can replace it without touching
``GameService``; the ML area owns evaluation sets (docs/ROADMAP.md Phase 5).
"""

import random
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# Common, unambiguous Korean nouns. Deliberately small and easy to replace.
ANSWER_WORDS: tuple[str, ...] = (
    "사과",
    "학생",
    "선생",
    "학교",
    "바다",
    "강아지",
    "고양이",
    "자동차",
    "컴퓨터",
    "음악",
    "영화",
    "커피",
    "여행",
    "친구",
    "가족",
    "시간",
    "행복",
    "도시",
    "하늘",
    "책",
)


@runtime_checkable
class AnswerSelector(Protocol):
    """Chooses the hidden answer for a new game."""

    def __call__(self) -> str: ...


class RandomAnswerSelector:
    """Picks a random word from ``words``.

    Accepts an explicit ``random.Random`` so tests can pin the choice instead of
    monkeypatching module-level state.
    """

    def __init__(
        self,
        words: Sequence[str] = ANSWER_WORDS,
        rng: random.Random | None = None,
    ) -> None:
        if not words:
            raise ValueError("words must not be empty")
        self._words = tuple(words)
        self._rng = rng or random.Random()

    def __call__(self) -> str:
        return self._rng.choice(self._words)
