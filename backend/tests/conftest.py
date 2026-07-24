"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    """A TestClient bound to a fresh app instance (default: mock embeddings)."""
    return TestClient(create_app())
