"""
Product verification service layer.
Handles business logic for product verification, scoring, and ranking.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from app.api.repository.products_repository import ProductsRepository, FDAModel
from app.models import (
    DrugProducts, FoodProducts, DrugIndustry, FoodIndustry,
    MedicalDeviceIndustry, CosmeticIndustry, DrugsNewApplications
)


@dataclass
class ProductSearchResult:
    """
    Data Transfer Object for product search results with business logic applied.
    """
    model_instance: FDAModel
    relevance_score: float
    matched_fields: List[str]
    product_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        # Get the primary key value - models use registration_number, license_number, etc.
        primary_key = self._get_primary_key_value()
        
        result = {
            'id': primary_key,
            'relevance_score': self.relevance_score,
            'matched_fields': self.matched_fields,
            'type': self.product_type
        }
        
        # Add model-specific fields
        if isinstance(self.model_instance, DrugProducts):
            result.update({
                'registration_number': self.model_instance.registration_number,
                'brand_name': self.model_instance.brand_name,
                'generic_name': self.model_instance.generic_name,
                'manufacturer': self.model_instance.manufacturer,
            })
        elif isinstance(self.model_instance, FoodProducts):
            result.update({
                'registration_number': self.model_instance.registration_number,
                'product_name': self.model_instance.product_name,
                'company_name': self.model_instance.company_name,
            })
        elif isinstance(self.model_instance, DrugIndustry):
            result.update({
                'license_number': self.model_instance.license_number,
                'name_of_establishment': self.model_instance.name_of_establishment,
            })
        elif isinstance(self.model_instance, FoodIndustry):
            result.update({
                'license_number': self.model_instance.license_number,
                'name_of_establishment': self.model_instance.name_of_establishment,
            })
        elif isinstance(self.model_instance, MedicalDeviceIndustry):
            result.update({
                'license_number': self.model_instance.license_number,
                'name_of_establishment': self.model_instance.name_of_establishment,
            })
        elif isinstance(self.model_instance, CosmeticIndustry):
            result.update({
                'license_number': self.model_instance.license_number,
                'name_of_establishment': self.model_instance.name_of_establishment,
            })
        elif isinstance(self.model_instance, DrugsNewApplications):
            result.update({
                'document_tracking_number': self.model_instance.document_tracking_number,
                'brand_name': self.model_instance.brand_name,
                'applicant': self.model_instance.applicant,
                'application_type': self.model_instance.application_type,
            })
        
        return result
    
    def _get_primary_key_value(self) -> str:
        """Get the primary key value from the model instance."""
        if isinstance(self.model_instance, (DrugProducts, FoodProducts)):
            return self.model_instance.registration_number
        elif isinstance(self.model_instance, (DrugIndustry, FoodIndustry, MedicalDeviceIndustry, CosmeticIndustry)):
            return self.model_instance.license_number
        elif isinstance(self.model_instance, DrugsNewApplications):
            return self.model_instance.document_tracking_number
        else:
            # Fallback - try to get any available identifier
            for attr in ['registration_number', 'license_number', 'document_tracking_number']:
                if hasattr(self.model_instance, attr):
                    value = getattr(self.model_instance, attr)
                    if value:
                        return value
            return 'unknown'


class ProductVerificationService:
    """
    Service layer for product verification business logic.
    Handles scoring, ranking, and verification logic.
    """
    
    def __init__(self, products_repo: ProductsRepository):
        """
        Initialize service with repository dependency.
        
        Args:
            products_repo: Products repository for data access
        """
        self.products_repo = products_repo
    
    async def search_and_rank_products(self, product_info: Dict[str, Any]) -> List[ProductSearchResult]:
        """
        Search for products and apply business logic for ranking.
        
        Args:
            product_info: Search criteria
            
        Returns:
            List of search results sorted by relevance
        """
        # Repository handles data access only
        raw_results = await self.products_repo.fuzzy_search_by_product_info(product_info)
        
        # Service layer applies business logic
        scored_results = []
        for model_instance in raw_results:
            score, matched_fields = self._calculate_relevance_score(model_instance, product_info)
            product_type = self._get_product_type(model_instance)
            
            scored_results.append(ProductSearchResult(
                model_instance=model_instance,
                relevance_score=score,
                matched_fields=matched_fields,
                product_type=product_type
            ))
        
        # Sort by relevance score (highest first)
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return scored_results
    
    async def verify_product_by_id(self, product_id: str) -> List[ProductSearchResult]:
        """
        Verify a product by its ID with business logic applied.
        
        Optimized to use search_by_any_id which reduces queries from 21 to 7.
        
        Args:
            product_id: Product ID to verify (registration/license/tracking number)
            
        Returns:
            List of matching products with verification scores
        """
        # Get raw data from repository using optimized single method (7 queries instead of 21)
        all_matches = await self.products_repo.search_by_any_id(product_id)
        
        # Apply business logic for exact vs partial matches
        results = []
        for model_instance in all_matches:
            score, matched_fields = self._calculate_id_match_score(model_instance, product_id)
            product_type = self._get_product_type(model_instance)
            
            results.append(ProductSearchResult(
                model_instance=model_instance,
                relevance_score=score,
                matched_fields=matched_fields,
                product_type=product_type
            ))
        
        # Sort by relevance (exact matches first)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def _calculate_relevance_score(self, model_instance: FDAModel, search_info: Dict[str, Any]) -> tuple[float, List[str]]:
        """
        Calculate relevance score for a search result with business logic.
        
        Args:
            model_instance: Database model instance
            search_info: Original search criteria
            
        Returns:
            Tuple of (relevance_score, matched_fields)
        """
        score = 0.0
        matched_fields = []
        
        # Convert model to dict for easier field access
        model_dict = self._model_to_search_dict(model_instance)
        
        # Registration number match (highest weight)
        if search_info.get('registration_number') and model_dict.get('registration_number'):
            if search_info['registration_number'].lower() in model_dict['registration_number'].lower():
                score += 0.4  # 40% weight for registration number
                matched_fields.append('registration_number')
        
        # Brand name match with improved scoring
        if search_info.get('brand_name'):
            brand_fields = ['brand_name', 'product_name']
            for field in brand_fields:
                if model_dict.get(field):
                    search_brand = search_info['brand_name'].lower()
                    field_brand = model_dict[field].lower()
                    
                    # Exact match (highest score)
                    if search_brand == field_brand:
                        score += 0.5  # 50% weight for exact brand match
                        matched_fields.append(field)
                        break
                    # Core brand substring match (e.g., "C2" in "C2 COOL & CLEAN")
                    elif search_brand in field_brand or field_brand in search_brand:
                        score += 0.4  # 40% weight for brand substring match
                        matched_fields.append(field)
                        break
                    # Word-level brand match
                    else:
                        search_words = set(search_brand.split())
                        field_words = set(field_brand.split())
                        if search_words & field_words:
                            score += 0.3  # 30% weight for word-level brand match
                            matched_fields.append(field)
                            break
        
        # Company/establishment name match
        if search_info.get('company_name'):
            company_fields = ['company_name', 'applicant', 'name_of_establishment']
            for field in company_fields:
                if model_dict.get(field):
                    if search_info['company_name'].lower() in model_dict[field].lower():
                        score += 0.2  # 20% weight for company name
                        matched_fields.append(field)
                    break
        
        # Product description/name match - handle both product_description and legacy fields
        # Use word-level matching to handle word order variations
        product_description = search_info.get('product_description') or search_info.get('generic_name') or search_info.get('product_name')
        if product_description:
            product_fields = ['product_name', 'generic_name']
            for field in product_fields:
                if model_dict.get(field):
                    # Exact phrase match (highest score)
                    if product_description.lower() in model_dict[field].lower():
                        score += 0.25  # 25% weight for exact product description match
                        matched_fields.append(field)
                        break
                    # Word-level tokenized match (handles word order variations)
                    else:
                        search_words = set(word.lower() for word in product_description.strip().split() if len(word) >= 3)
                        field_words = set(word.lower() for word in str(model_dict[field]).strip().split() if len(word) >= 3)
                        
                        if search_words and field_words:
                            # Calculate word overlap ratio
                            common_words = search_words & field_words
                            overlap_ratio = len(common_words) / len(search_words)
                            
                            # Award partial score based on word overlap
                            if overlap_ratio >= 0.5:  # At least 50% of words match
                                score += 0.2 * overlap_ratio  # Up to 20% weight for partial match
                                matched_fields.append(field)
                                break
        
        return score, matched_fields
    
    def _calculate_id_match_score(self, model_instance: FDAModel, product_id: str) -> tuple[float, List[str]]:
        """
        Calculate match score for ID-based searches.
        
        Args:
            model_instance: Database model instance
            product_id: ID being searched for
            
        Returns:
            Tuple of (match_score, matched_fields)
        """
        model_dict = self._model_to_search_dict(model_instance)
        matched_fields = []
        
        # Exact matches get highest score
        if model_dict.get('registration_number') and model_dict['registration_number'].lower() == product_id.lower():
            return 1.0, ['registration_number']
        elif model_dict.get('license_number') and model_dict['license_number'].lower() == product_id.lower():
            return 1.0, ['license_number']
        elif model_dict.get('document_tracking_number') and model_dict['document_tracking_number'].lower() == product_id.lower():
            return 1.0, ['document_tracking_number']
        
        # Partial matches get lower scores
        score = 0.0
        if model_dict.get('registration_number') and product_id.lower() in model_dict['registration_number'].lower():
            score = 0.8
            matched_fields.append('registration_number')
        elif model_dict.get('license_number') and product_id.lower() in model_dict['license_number'].lower():
            score = 0.8
            matched_fields.append('license_number')
        elif model_dict.get('document_tracking_number') and product_id.lower() in model_dict['document_tracking_number'].lower():
            score = 0.8
            matched_fields.append('document_tracking_number')
        
        return score, matched_fields
    
    def _model_to_search_dict(self, model_instance: FDAModel) -> Dict[str, Any]:
        """
        Convert model instance to dictionary for search operations.
        
        Args:
            model_instance: SQLAlchemy model instance
            
        Returns:
            Dictionary with searchable fields
        """
        if isinstance(model_instance, DrugProducts):
            return {
                'registration_number': model_instance.registration_number,
                'brand_name': model_instance.brand_name,
                'generic_name': model_instance.generic_name,
                'manufacturer': model_instance.manufacturer,
            }
        elif isinstance(model_instance, FoodProducts):
            return {
                'registration_number': model_instance.registration_number,
                'product_name': model_instance.product_name,
                'company_name': model_instance.company_name,
            }
        elif isinstance(model_instance, DrugIndustry):
            return {
                'license_number': model_instance.license_number,
                'name_of_establishment': model_instance.name_of_establishment,
            }
        elif isinstance(model_instance, FoodIndustry):
            return {
                'license_number': model_instance.license_number,
                'name_of_establishment': model_instance.name_of_establishment,
            }
        elif isinstance(model_instance, MedicalDeviceIndustry):
            return {
                'license_number': model_instance.license_number,
                'name_of_establishment': model_instance.name_of_establishment,
            }
        elif isinstance(model_instance, CosmeticIndustry):
            return {
                'license_number': model_instance.license_number,
                'name_of_establishment': model_instance.name_of_establishment,
            }
        elif isinstance(model_instance, DrugsNewApplications):
            return {
                'document_tracking_number': model_instance.document_tracking_number,
                'brand_name': model_instance.brand_name,
                'applicant': model_instance.applicant,
                'application_type': model_instance.application_type,
            }
        
        return {}
    
    def _get_product_type(self, model_instance: FDAModel) -> str:
        """
        Get product type string for model instance.
        
        Args:
            model_instance: SQLAlchemy model instance
            
        Returns:
            Product type string
        """
        if isinstance(model_instance, DrugProducts):
            return 'drug_product'
        elif isinstance(model_instance, FoodProducts):
            return 'food_product'
        elif isinstance(model_instance, DrugIndustry):
            return 'drug_industry'
        elif isinstance(model_instance, FoodIndustry):
            return 'food_industry'
        elif isinstance(model_instance, MedicalDeviceIndustry):
            return 'medical_device_industry'
        elif isinstance(model_instance, CosmeticIndustry):
            return 'cosmetic_industry'
        elif isinstance(model_instance, DrugsNewApplications):
            return 'drug_application'
        
        return 'unknown'