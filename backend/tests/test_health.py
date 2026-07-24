"""Tests for the health endpoint and its response shape."""

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape(client: TestClient) -> None:
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
