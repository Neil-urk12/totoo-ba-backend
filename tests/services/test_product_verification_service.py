import pytest
from unittest.mock import AsyncMock, Mock
from app.services.product_verification_service import ProductVerificationService
from app.models.drug_products import DrugProducts
from app.models.food_products import FoodProducts

@pytest.fixture
def mock_products_repo():
    repo = Mock()
    repo.fuzzy_search_by_product_info = AsyncMock()
    return repo

@pytest.fixture
def verification_service(mock_products_repo):
    return ProductVerificationService(mock_products_repo)

@pytest.mark.asyncio
async def test_search_and_rank_products_exact_match(verification_service, mock_products_repo):
    # Given
    product_info = {
        "brand_name": "Test Drug",
        "registration_number": "DR-XY12345"
    }
    drug_product = DrugProducts(
        brand_name="Test Drug",
        registration_number="DR-XY12345",
        generic_name="Test Generic",
        manufacturer="Test Manufacturer"
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = [drug_product]

    # When
    results = await verification_service.search_and_rank_products(product_info)

    # Then
    assert len(results) == 1
    assert results[0].relevance_score > 0.8
    assert "brand_name" in results[0].matched_fields
    assert "registration_number" in results[0].matched_fields
    mock_products_repo.fuzzy_search_by_product_info.assert_called_once_with(product_info)

@pytest.mark.asyncio
async def test_search_and_rank_products_partial_match(verification_service, mock_products_repo):
    # Given
    product_info = {"brand_name": "Test"}
    drug_product = DrugProducts(
        brand_name="Test Drug",
        registration_number="DR-XY12345",
        generic_name="Test Generic",
        manufacturer="Test Manufacturer"
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = [drug_product]

    # When 
    results = await verification_service.search_and_rank_products(product_info)

    # Then
    assert len(results) == 1
    assert 0.3 < results[0].relevance_score < 0.6
    assert "brand_name" in results[0].matched_fields
    mock_products_repo.fuzzy_search_by_product_info.assert_called_once_with(product_info)

@pytest.mark.asyncio
async def test_search_and_rank_products_company_name_match(verification_service, mock_products_repo):
    # Given
    product_info = {"company_name": "Test Company"}
    food_product = FoodProducts(
        brand_name="Some Food Brand",
        registration_number="FR-XY12345",
        product_name="Test Product",
        company_name="Test Company Ltd"
    )
    mock_products_repo.fuzzy_search_by_product_info.return_value = [food_product]

    results = await verification_service.search_and_rank_products(product_info)

    assert len(results) == 1
    assert results[0].relevance_score > 0.1
    assert "company_name" in results[0].matched_fields
    mock_products_repo.fuzzy_search_by_product_info.assert_called_once_with(product_info)

@pytest.mark.asyncio
async def test_search_and_rank_products_no_match(verification_service, mock_products_repo):
    # Given
    product_info = {"brand_name": "Unknown"}
    mock_products_repo.fuzzy_search_by_product_info.return_value = []

    # When 
    results = await verification_service.search_and_rank_products(product_info)

    # Then
    assert len(results) == 0
    mock_products_repo.fuzzy_search_by_product_info.assert_called_once_with(product_info)
