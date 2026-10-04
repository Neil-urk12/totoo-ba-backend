"""Tests for main FastAPI application endpoints.

Tests the root and health check endpoints to ensure basic
application functionality.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_product_verification_service
from app.core.config import get_settings
from app.main import app, settings
from app.services.product_verification_service import ImageVerificationOutcome


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture(autouse=True)
def mock_verification_dependency():
    """Mock verification service dependency for API route tests."""
    mock_service = Mock()
    mock_service.verify_product_by_image = AsyncMock(
        return_value=ImageVerificationOutcome(
            verification_status="invalid",
            confidence=0,
            matched_product=None,
            extracted_fields={},
            ai_reasoning="File content mismatch",
            alternative_matches=[],
            processing_metadata={},
            is_valid_image=False,
            error_message="File type mismatch. The uploaded file does not match the declared content type.",
        )
    )

    async def override_verification_service():
        return mock_service

    async def override_settings():
        return settings

    app.dependency_overrides[get_product_verification_service] = (
        override_verification_service
    )
    app.dependency_overrides[get_settings] = override_settings
    yield mock_service
    app.dependency_overrides.clear()


async def test_root(client):
    """Test the root endpoint returns correct status and application info."""
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


async def test_health(client):
    """Test the health check endpoint returns healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


async def test_verify_image_invalid_content_type(client):
    """Test that uploading a non-image file type returns 400 Bad Request."""
    response = await client.post(
        "/api/v1/products/verify-image",
        files={"image": ("test.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


async def test_verify_image_content_mismatch(client):
    """Test that declaring an image MIME type with non-image bytes returns 400."""
    response = await client.post(
        "/api/v1/products/verify-image",
        files={"image": ("test.png", b"not-a-png", "image/png")},
    )
    assert response.status_code == 400
    assert "File type mismatch" in response.json()["detail"]


@pytest.mark.asyncio
async def test_verify_image_success(client, mock_verification_dependency):
    image_bytes = b"\xff\xd8\xffimage_payload"
    mock_verification_dependency.verify_product_by_image.return_value = (
        ImageVerificationOutcome(
            verification_status="verified",
            confidence=90,
            matched_product={"brand_name": "Test Drug"},
            extracted_fields={"brand_name": "Test Drug"},
            ai_reasoning="Strong match found",
            alternative_matches=[],
            processing_metadata={"total_time_ms": 150.0},
        )
    )

    response = await client.post(
        "/api/v1/products/verify-image",
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "verification_status": "verified",
        "confidence": 90,
        "matched_product": {"brand_name": "Test Drug"},
        "extracted_fields": {"brand_name": "Test Drug"},
        "ai_reasoning": "Strong match found",
        "alternative_matches": [],
        "processing_metadata": {"total_time_ms": 150.0},
    }
    mock_verification_dependency.verify_product_by_image.assert_awaited_once_with(
        image_bytes=image_bytes, mime_type="image/jpeg"
    )


async def test_new_verify_image_alias(client):
    """Test that backward-compatible /new-verify-image alias route functions."""
    response = await client.post(
        "/api/v1/products/new-verify-image",
        files={"image": ("test.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
