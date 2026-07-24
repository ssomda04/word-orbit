"""Tests for the deterministic embedding service and the dev similarity route."""

import pytest
from fastapi.testclient import TestClient

from app.services.embedding import DeterministicEmbeddingService


@pytest.fixture
def embedder() -> DeterministicEmbeddingService:
    return DeterministicEmbeddingService()


def test_same_input_same_output(embedder: DeterministicEmbeddingService) -> None:
    """Determinism: identical input always yields identical vectors."""
    assert embedder.encode("학생") == embedder.encode("학생")


def test_different_input_different_output(embedder: DeterministicEmbeddingService) -> None:
    assert embedder.encode("학생") != embedder.encode("선생")


def test_similarity_in_range(embedder: DeterministicEmbeddingService) -> None:
    score = embedder.similarity("학생", "선생")
    assert -1.0 <= score <= 1.0


def test_self_similarity_is_one(embedder: DeterministicEmbeddingService) -> None:
    assert embedder.similarity("학생", "학생") == pytest.approx(1.0)


def test_encode_produces_unit_vector(embedder: DeterministicEmbeddingService) -> None:
    vec = embedder.encode("사과")
    magnitude = sum(x * x for x in vec) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-9)


def test_project_3d_shape(embedder: DeterministicEmbeddingService) -> None:
    coords = embedder.project_3d(["학생", "선생", "사과"])
    assert len(coords) == 3
    assert all(len(point) == 3 for point in coords)


def test_empty_input_rejected(embedder: DeterministicEmbeddingService) -> None:
    with pytest.raises(ValueError):
        embedder.encode("   ")


def test_dev_similarity_endpoint(client: TestClient) -> None:
    response = client.post("/api/dev/similarity", json={"first": "학생", "second": "선생"})
    assert response.status_code == 200
    body = response.json()
    assert -1.0 <= body["similarity"] <= 1.0
    assert body["provider"] == "mock"


def test_dev_similarity_rejects_blank(client: TestClient) -> None:
    response = client.post("/api/dev/similarity", json={"first": "", "second": "선생"})
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_INPUT"
