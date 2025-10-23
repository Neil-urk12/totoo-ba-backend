"""Tests for main FastAPI application endpoints.

Tests the root and health check endpoints to ensure basic
application functionality.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_root():
    """Test the root endpoint returns correct status and application info."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
