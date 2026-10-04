"""Tests for ProductVerificationService.

Tests the business logic for product verification, scoring, and ranking
using mocked repository dependencies.
"""

from datetime import date
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.models.drug_industry import DrugIndustry
from app.models.drug_products import DrugProducts
from app.models.food_products import FoodProducts
from app.services.product_verification_service import ProductVerificationService
from app.services.vision_service import ExtractedData, ProcessingMetadata


@pytest.fixture
def mock_products_repo():
    """Create a mock ProductsRepository for testing.

    Returns:
        Mock: Mocked repository with AsyncMock for async methods.
    """
    repo = Mock()
    repo.fuzzy_search_by_product_info = AsyncMock()
    repo.search_by_any_id = AsyncMock()
    return repo


@pytest.fixture
def mock_vision_service():
    """Create a mock vision service for testing."""
    service = Mock()
    service.extract_product_info = AsyncMock()
    return service


@pytest.fixture
def verification_service(mock_products_repo):
    """Create a ProductVerificationService instance with mocked repository.

    Args:
        mock_products_repo: Mocked repository fixture.

    Returns:
        ProductVerificationService: Service instance for testing.
    """
    return ProductVerificationService(mock_products_repo)


@pytest.fixture
def verification_service_with_vision(mock_products_repo, mock_vision_service):
    """Create a ProductVerificationService instance with mocked repository and vision service."""
    return ProductVerificationService(
        products_repo=mock_products_repo, vision_service=mock_vision_service
    )


@pytest.mark.asyncio
async def test_search_and_rank_products_exact_match(
    verification_service, mock_products_repo
):
    """Test product search with exact match on brand name and registration number.

    Verifies that exact matches receive high relevance scores and correct matched fields.
    """
    # Given
    product_info = {"brand_name": "Test Drug", "registration_number": "DR-XY12345"}
    drug_product = DrugProducts(
        brand_name="Test Drug",
        registration_number="DR-XY12345",
        generic_name="Test Generic",
        manufacturer="Test Manufacturer",
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = [drug_product]

    # When
    results = await verification_service.search_and_rank_products(product_info)

    # Then
    assert len(results) == 1
    assert results[0].relevance_score > 0.8
    assert "brand_name" in results[0].matched_fields
    assert "registration_number" in results[0].matched_fields
    mock_products_repo.fuzzy_search_by_product_info.assert_called_once_with(
        product_info
    )


@pytest.mark.asyncio
async def test_search_and_rank_products_partial_match(
    verification_service, mock_products_repo
):
    """Test product search with partial brand name match.

    Verifies that partial matches receive moderate relevance scores.
    """
    # Given
    product_info = {"brand_name": "Test"}
    drug_product = DrugProducts(
        brand_name="Test Drug",
        registration_number="DR-XY12345",
        generic_name="Test Generic",
        manufacturer="Test Manufacturer",
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = [drug_product]

    # When
    results = await verification_service.search_and_rank_products(product_info)

    # Then
    assert len(results) == 1
    assert 0.3 < results[0].relevance_score < 0.6
    assert "brand_name" in results[0].matched_fields
    mock_products_repo.fuzzy_search_by_product_info.assert_called_once_with(
        product_info
    )


@pytest.mark.asyncio
async def test_search_and_rank_products_company_name_match(
    verification_service, mock_products_repo
):
    """Test product search with company name match.

    Verifies that company name matches are properly scored and identified.
    """
    # Given
    product_info = {"company_name": "Test Company"}
    food_product = FoodProducts(
        brand_name="Some Food Brand",
        registration_number="FR-XY12345",
        product_name="Test Product",
        company_name="Test Company Ltd",
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = [food_product]

    results = await verification_service.search_and_rank_products(product_info)

    assert len(results) == 1
    assert results[0].relevance_score > 0.1
    assert "company_name" in results[0].matched_fields
    mock_products_repo.fuzzy_search_by_product_info.assert_called_once_with(
        product_info
    )


@pytest.mark.asyncio
async def test_search_and_rank_products_no_match(
    verification_service, mock_products_repo
):
    """Test product search with no matching results.

    Verifies that empty results are handled correctly.
    """
    # Given
    product_info = {"brand_name": "Unknown"}
    mock_products_repo.fuzzy_search_by_product_info.return_value = []

    # When
    results = await verification_service.search_and_rank_products(product_info)

    # Then
    assert len(results) == 0
    mock_products_repo.fuzzy_search_by_product_info.assert_called_once_with(
        product_info
    )


@pytest.mark.asyncio
async def test_verify_product_by_id_exact_registration_match(
    verification_service, mock_products_repo
):
    """Exact registration number returns a verified outcome."""
    drug_product = DrugProducts(
        brand_name="Test Drug",
        registration_number="DR-XY12345",
        generic_name="Test Generic",
        manufacturer="Test Manufacturer",
    )
    mock_products_repo.search_by_any_id.return_value = [drug_product]

    outcome = await verification_service.verify_product_by_id("DR-XY12345")

    assert outcome.is_verified is True
    assert outcome.details["exact_match"] is True
    assert outcome.details["confidence_score"] == 100
    assert outcome.details["matched_field"] == "registration_number"
    assert "Verified Drug Product" in outcome.message
    mock_products_repo.search_by_any_id.assert_called_once_with("DR-XY12345")


@pytest.mark.asyncio
async def test_verify_product_by_id_normalizes_spaced_registration(
    verification_service, mock_products_repo
):
    """Spaced registration numbers match after normalization."""
    food_product = FoodProducts(
        brand_name="Test Food",
        registration_number="FR-4000003862579",
        product_name="Test Product",
        company_name="Test Company",
    )
    mock_products_repo.search_by_any_id.return_value = [food_product]

    outcome = await verification_service.verify_product_by_id("FR-4000 003862579")

    assert outcome.is_verified is True
    assert outcome.details["matched_field"] == "registration_number"


@pytest.mark.asyncio
async def test_verify_product_by_id_establishment_license_match(
    verification_service, mock_products_repo
):
    """Establishment license numbers verify with the correct message."""
    establishment = DrugIndustry(
        license_number="LI-12345",
        name_of_establishment="Test Pharma Inc",
        owner="Owner",
        address="Address",
        region="NCR",
        activity="Manufacturing",
        issuance_date=date(2024, 1, 1),
        expiry_date=date(2025, 1, 1),
    )
    mock_products_repo.search_by_any_id.return_value = [establishment]

    outcome = await verification_service.verify_product_by_id("LI-12345")

    assert outcome.is_verified is True
    assert outcome.details["matched_field"] == "license_number"
    assert "Verified Establishment" in outcome.message


@pytest.mark.asyncio
async def test_verify_product_by_id_invalid_id(
    verification_service, mock_products_repo
):
    """Short IDs return an invalid-ID outcome without querying the repository."""
    outcome = await verification_service.verify_product_by_id("AB")

    assert outcome.is_verified is False
    assert outcome.details["error_code"] == "INVALID_ID"
    mock_products_repo.search_by_any_id.assert_not_called()


@pytest.mark.asyncio
async def test_verify_product_by_id_not_found(verification_service, mock_products_repo):
    """No repository matches return a not-found outcome with suggestions."""
    mock_products_repo.search_by_any_id.return_value = []

    outcome = await verification_service.verify_product_by_id("DR-NOTFOUND")

    assert outcome.is_verified is False
    assert outcome.details["confidence_score"] == 0
    assert outcome.details["suggestions"]
    assert "not found" in outcome.message


@pytest.mark.asyncio
async def test_verify_product_by_id_database_error(
    verification_service, mock_products_repo
):
    """Repository SQLAlchemy failures return a database-error outcome."""
    mock_products_repo.search_by_any_id.side_effect = SQLAlchemyError(
        "connection failed"
    )

    outcome = await verification_service.verify_product_by_id("DR-XY12345")

    assert outcome.is_verified is False
    assert outcome.details["error_code"] == "DATABASE_ERROR"


@pytest.mark.asyncio
async def test_verify_product_by_id_internal_error(
    verification_service, mock_products_repo
):
    """Non-database exceptions return an internal-error outcome."""
    mock_products_repo.search_by_any_id.side_effect = RuntimeError("unexpected failure")

    outcome = await verification_service.verify_product_by_id("DR-XY12345")

    assert outcome.is_verified is False
    assert outcome.details["error_code"] == "INTERNAL_ERROR"


@pytest.mark.asyncio
async def test_verify_product_by_id_partial_match(
    verification_service, mock_products_repo
):
    """Substring ID matches (scored at 0.8) return a partial-match outcome, not verified."""
    drug_product = DrugProducts(
        brand_name="Test Drug",
        registration_number="DR-XY12345",
        generic_name="Test Generic",
        manufacturer="Test Manufacturer",
    )
    mock_products_repo.search_by_any_id.return_value = [drug_product]

    outcome = await verification_service.verify_product_by_id("DR-XY1234")

    assert outcome.is_verified is False
    assert outcome.details["exact_match"] is False
    assert outcome.details["confidence_score"] == 80
    assert "Possible match" in outcome.message
    assert len(outcome.details["possible_matches"]) >= 1


@pytest.mark.asyncio
async def test_verify_product_by_image_invalid_magic_bytes(
    verification_service_with_vision,
):
    """Corrupted or invalid image bytes return invalid outcome."""
    outcome = await verification_service_with_vision.verify_product_by_image(
        image_bytes=b"not_an_image_at_all", mime_type="image/jpeg"
    )

    assert outcome.is_valid_image is False
    assert outcome.verification_status == "invalid"
    assert outcome.confidence == 0
    assert outcome.matched_product is None
    assert "File type mismatch" in (outcome.error_message or "")


@pytest.mark.asyncio
async def test_verify_product_by_image_exact_match(
    verification_service_with_vision, mock_vision_service, mock_products_repo
):
    """Valid image with matching product yields verified outcome."""
    # JPEG header
    valid_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]) + b"image_payload"

    mock_vision_service.extract_product_info.return_value = (
        ExtractedData(
            brand_name="Test Drug",
            registration_number="DR-XY12345",
            product_description="Paracetamol 500mg",
            manufacturer="Test Pharma",
        ),
        ProcessingMetadata(
            groq_vision_time=0.1,
            groq_llama31_time=0.05,
            total_time=0.15,
            layers_used=["Groq Vision", "Groq Llama 3.1"],
            groq_vision_confidence=0.95,
        ),
    )

    drug_product = DrugProducts(
        brand_name="Test Drug",
        registration_number="DR-XY12345",
        generic_name="Paracetamol",
        manufacturer="Test Pharma",
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = [drug_product]

    outcome = await verification_service_with_vision.verify_product_by_image(
        image_bytes=valid_jpeg, mime_type="image/jpeg"
    )

    assert outcome.is_valid_image is True
    assert outcome.verification_status == "verified"
    assert outcome.confidence >= 80
    assert outcome.matched_product is not None
    assert outcome.matched_product["brand_name"] == "Test Drug"
    assert "Strong match found" in outcome.ai_reasoning


@pytest.mark.asyncio
async def test_verify_product_by_image_no_match(
    verification_service_with_vision, mock_vision_service, mock_products_repo
):
    """Valid image with no database matches yields not_found outcome."""
    valid_png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"png_data"

    mock_vision_service.extract_product_info.return_value = (
        ExtractedData(brand_name="Unregistered Product"),
        ProcessingMetadata(total_time=0.1),
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = []

    outcome = await verification_service_with_vision.verify_product_by_image(
        image_bytes=valid_png, mime_type="image/png"
    )

    assert outcome.is_valid_image is True
    assert outcome.verification_status == "not_found"
    assert outcome.confidence == 0
    assert outcome.matched_product is None
    assert "No matching products found" in outcome.ai_reasoning


@pytest.mark.asyncio
async def test_verify_product_by_image_null_timings(
    verification_service_with_vision, mock_vision_service, mock_products_repo
):
    mock_vision_service.extract_product_info.return_value = (
        ExtractedData(),
        ProcessingMetadata(
            groq_vision_time=None,
            groq_llama31_time=None,
            groq_fallback_time=None,
            total_time=None,
        ),
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = []

    outcome = await verification_service_with_vision.verify_product_by_image(
        image_bytes=b"\xff\xd8\xffimage_payload", mime_type="image/jpeg"
    )

    assert {
        key: outcome.processing_metadata[key]
        for key in (
            "groq_vision_time_ms",
            "groq_llama31_time_ms",
            "groq_fallback_time_ms",
            "total_time_ms",
        )
    } == {
        "groq_vision_time_ms": 0.0,
        "groq_llama31_time_ms": 0.0,
        "groq_fallback_time_ms": 0.0,
        "total_time_ms": 0.0,
    }
