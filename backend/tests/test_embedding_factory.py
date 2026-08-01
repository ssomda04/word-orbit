"""Provider selection, model-path validation, and startup warm-up.

None of these need the `fasttext` library or a real model: `FastTextEmbeddingService.load`
is patched where an actual load would happen.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services.embedding import (
    DeterministicEmbeddingService,
    FastTextConfigurationError,
    FastTextEmbeddingService,
    get_embedding_service,
    reset_embedding_service,
)
from app.services.embedding import factory as factory_module
from tests.test_fasttext_embedding import FakeFastTextModel

Configure = Callable[..., None]


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch) -> Iterator[Configure]:
    """Point the cached Settings at a fresh environment.

    `get_settings` and the embedding service are both process-cached, so both
    caches have to be dropped after the environment changes.
    """

    def _configure(**env: str) -> None:
        monkeypatch.delenv("FASTTEXT_MODEL_PATH", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        reset_embedding_service()

    yield _configure
    get_settings.cache_clear()
    reset_embedding_service()


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """A file that exists — enough for path validation, not a real model."""
    path = tmp_path / "cc.ko.300.bin"
    path.write_bytes(b"not a real model")
    return path


@pytest.fixture
def patched_load(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the real loader; the returned list records every load call."""
    loaded: list[Path] = []

    def _fake_load(cls: type[FastTextEmbeddingService], path: Path) -> FastTextEmbeddingService:
        loaded.append(path)
        return cls(FakeFastTextModel())

    monkeypatch.setattr(FastTextEmbeddingService, "load", classmethod(_fake_load))
    return loaded


# --- Mock providers (regression) -------------------------------------------


@pytest.mark.parametrize("provider", ["mock", "deterministic", "MOCK", "  Deterministic  "])
def test_mock_providers_resolve_to_the_deterministic_service(
    configure: Configure, provider: str
) -> None:
    configure(EMBEDDING_PROVIDER=provider)

    assert isinstance(get_embedding_service(), DeterministicEmbeddingService)


def test_default_provider_is_the_mock(configure: Configure) -> None:
    """No EMBEDDING_PROVIDER set at all still yields the dependency-free mock."""
    configure()

    assert isinstance(get_embedding_service(), DeterministicEmbeddingService)


# --- FastText provider ------------------------------------------------------


@pytest.mark.parametrize("provider", ["fasttext", "FastText", "  fast-text  ", "fast_text"])
def test_fasttext_provider_aliases_build_the_fasttext_service(
    configure: Configure, model_file: Path, patched_load: list[Path], provider: str
) -> None:
    configure(EMBEDDING_PROVIDER=provider, FASTTEXT_MODEL_PATH=str(model_file))

    service = get_embedding_service()

    assert isinstance(service, FastTextEmbeddingService)
    assert patched_load == [model_file]


def test_missing_model_path_fails_with_actionable_guidance(configure: Configure) -> None:
    configure(EMBEDDING_PROVIDER="fasttext")

    with pytest.raises(FastTextConfigurationError) as excinfo:
        get_embedding_service()

    message = str(excinfo.value)
    assert "FASTTEXT_MODEL_PATH" in message
    assert "uv sync --extra fasttext" in message
    assert "never downloads" in message


def test_blank_model_path_is_treated_as_missing(configure: Configure) -> None:
    configure(EMBEDDING_PROVIDER="fasttext", FASTTEXT_MODEL_PATH="   ")

    with pytest.raises(FastTextConfigurationError, match="FASTTEXT_MODEL_PATH"):
        get_embedding_service()


def test_nonexistent_model_path_names_the_path(configure: Configure, tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "cc.ko.300.bin"
    configure(EMBEDDING_PROVIDER="fasttext", FASTTEXT_MODEL_PATH=str(missing))

    with pytest.raises(FastTextConfigurationError) as excinfo:
        get_embedding_service()

    assert str(missing) in str(excinfo.value)


def test_directory_model_path_is_rejected(configure: Configure, tmp_path: Path) -> None:
    configure(EMBEDDING_PROVIDER="fasttext", FASTTEXT_MODEL_PATH=str(tmp_path))

    with pytest.raises(FastTextConfigurationError, match="directory, not a model file"):
        get_embedding_service()


# --- Other providers --------------------------------------------------------


def test_sentence_transformers_is_still_not_implemented(configure: Configure) -> None:
    configure(EMBEDDING_PROVIDER="sentence-transformers")

    with pytest.raises(NotImplementedError):
        get_embedding_service()


def test_unknown_provider_lists_fasttext_as_an_option(configure: Configure) -> None:
    configure(EMBEDDING_PROVIDER="word2vec")

    with pytest.raises(ValueError) as excinfo:
        get_embedding_service()

    message = str(excinfo.value)
    assert "word2vec" in message
    assert "fasttext" in message


# --- One instance per process ----------------------------------------------


def test_service_is_built_once_per_process(configure: Configure) -> None:
    configure(EMBEDDING_PROVIDER="mock")

    assert get_embedding_service() is get_embedding_service()


def test_fasttext_model_is_loaded_only_once(
    configure: Configure, model_file: Path, patched_load: list[Path]
) -> None:
    """The expensive part must not run twice, whatever the caller does."""
    configure(EMBEDDING_PROVIDER="fasttext", FASTTEXT_MODEL_PATH=str(model_file))

    first = get_embedding_service()
    second = get_embedding_service()

    assert first is second
    assert len(patched_load) == 1


def test_reset_forces_a_rebuild(configure: Configure) -> None:
    configure(EMBEDDING_PROVIDER="mock")
    first = get_embedding_service()

    reset_embedding_service()

    assert get_embedding_service() is not first


def test_concurrent_first_calls_load_the_model_once(
    configure: Configure, model_file: Path, patched_load: list[Path]
) -> None:
    """A plain `lru_cache` would let two simultaneous misses both load the model."""
    import threading

    configure(EMBEDDING_PROVIDER="fasttext", FASTTEXT_MODEL_PATH=str(model_file))
    barrier = threading.Barrier(8)
    services: list[object] = []

    def _worker() -> None:
        barrier.wait()
        services.append(get_embedding_service())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(patched_load) == 1
    assert len({id(service) for service in services}) == 1


# --- Startup warm-up --------------------------------------------------------


def test_startup_builds_the_fasttext_service_before_any_request(
    configure: Configure, model_file: Path, patched_load: list[Path]
) -> None:
    configure(EMBEDDING_PROVIDER="fasttext", FASTTEXT_MODEL_PATH=str(model_file))

    # Entering the context manager runs the lifespan; no request has been sent.
    with TestClient(create_app()) as client:
        assert patched_load == [model_file]

        response = client.post("/api/dev/similarity", json={"first": "학생", "second": "선생"})

    assert response.status_code == 200
    assert response.json()["provider"] == "fasttext"
    # Serving requests must not trigger a second load.
    assert len(patched_load) == 1


def test_startup_fails_loudly_on_a_bad_model_path(configure: Configure, tmp_path: Path) -> None:
    """A misconfigured path must stop the server, not surface as a runtime 500."""
    configure(EMBEDDING_PROVIDER="fasttext", FASTTEXT_MODEL_PATH=str(tmp_path / "missing.bin"))

    with pytest.raises(FastTextConfigurationError), TestClient(create_app()):
        pass  # pragma: no cover - startup raises before the body runs


def test_startup_warm_up_keeps_the_mock_provider_working(configure: Configure) -> None:
    configure(EMBEDDING_PROVIDER="mock")

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert isinstance(factory_module.get_embedding_service(), DeterministicEmbeddingService)
