"""Selects and caches the active rank provider.

Same shape as ``app.services.embedding.factory``, and for the same reason: the
vocabulary matrix is built exactly once per process and shared, so a
double-checked lock (not ``lru_cache``) guards construction — ``lru_cache``
stores one result but does not stop two threads from both running the body.

Ranking is off unless ``VOCABULARY_PATH`` is set. Ranks are only meaningful
relative to a word set, and there is no default one: the game vocabulary is
generated from a Wiktionary dump and is never committed (see ``ml/data``). With
no path configured every rank is ``None``, which is what the contract has always
permitted, so the default install and CI are unaffected.
"""

import threading
from pathlib import Path

from app.core.config import Settings, get_settings
from app.domain.vocabulary import load_vocabulary
from app.services.embedding import get_embedding_service
from app.services.ranking.base import RankProvider
from app.services.ranking.null import NullRankProvider
from app.services.ranking.vector import VectorRankProvider

_instance: RankProvider | None = None
_lock = threading.Lock()


def get_rank_provider() -> RankProvider:
    """Return the process-wide rank provider, chosen by settings."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = _build_rank_provider()
        return _instance


def reset_rank_provider() -> None:
    """Drop the cached provider so the next call rebuilds it.

    For tests only: production wiring builds it once at startup
    (``app.main.create_app``) and never resets it.
    """
    global _instance
    with _lock:
        _instance = None


def _build_rank_provider() -> RankProvider:
    settings: Settings = get_settings()
    raw_path = settings.vocabulary_path.strip()
    if not raw_path:
        return NullRankProvider()

    # `expanduser` for `~/vocab.txt`; relative paths resolve against the process
    # working directory, so the docs ask for an absolute path.
    words = load_vocabulary(Path(raw_path).expanduser())
    if not words:
        raise ValueError(
            f"VOCABULARY_PATH={raw_path!r} contains no usable words. Provide one "
            f"word per line, or leave the variable empty to disable ranking."
        )
    return VectorRankProvider(
        words,
        get_embedding_service(),
        cache_size=settings.rank_cache_size,
    )
