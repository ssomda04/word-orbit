"""Application configuration.

Settings are loaded from environment variables (and an optional local `.env`
file). Never hard-code secrets or hosts here — see `.env.example` for the full
list of supported variables.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment.

    Field names are lower-case; the matching env var is the upper-case form
    (e.g. ``FRONTEND_ORIGIN`` -> ``frontend_origin``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"

    # Comma-separated list of allowed browser origins for CORS.
    # Kept as a raw string so a single env var can hold multiple origins.
    frontend_origin: str = "http://localhost:3000"

    # Which embedding implementation to use:
    # "mock" | "deterministic" | "fasttext" | "sentence-transformers".
    # "mock" and "deterministic" both resolve to DeterministicEmbeddingService.
    embedding_provider: str = "mock"

    # Absolute path to a local FastText binary model (e.g. cc.ko.300.bin).
    # Required only when EMBEDDING_PROVIDER=fasttext. The server never downloads
    # a model. Validated in the embedding factory rather than here, so a mock run
    # (tests, CI) never fails over a variable it does not use.
    fasttext_model_path: str = ""

    # Populated only when a real model is selected (see docs/MODEL_EVALUATION.md).
    model_name: str = ""

    # Absolute path to a UTF-8 word-per-line vocabulary. Guess ranks are computed
    # relative to this word set; leave empty to disable ranking (`rank` stays
    # null, which the API contract permits). Validated in the ranking factory,
    # so a run without ranking never fails over a variable it does not use.
    vocabulary_path: str = ""

    # How many answers' vocabulary-wide similarity arrays to keep in memory.
    # Cost is roughly `RANK_CACHE_SIZE * vocabulary_size * 8` bytes.
    rank_cache_size: int = 32

    # Reserved for later phases (multiplayer / history). Empty = not configured.
    database_url: str = ""
    redis_url: str = ""

    @property
    def cors_origins(self) -> list[str]:
        """Parse ``frontend_origin`` into a clean list of allowed origins."""
        return [origin.strip() for origin in self.frontend_origin.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
