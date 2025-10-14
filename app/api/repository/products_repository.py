"""
Products repository for FDA verification and product-related database operations.
Handles searches across multiple FDA database tables.
"""

from typing import Dict, List, Any, Optional, Union
from sqlalchemy import select, or_, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from .database_repository import MultiTableRepository
from app.models import (
    DrugProducts, FoodProducts, DrugIndustry, FoodIndustry,
    MedicalDeviceIndustry, CosmeticIndustry, DrugsNewApplications
)

# Type aliases for better type safety
ProductModel = Union[DrugProducts, FoodProducts]
EstablishmentModel = Union[DrugIndustry, FoodIndustry, MedicalDeviceIndustry, CosmeticIndustry]
ApplicationModel = DrugsNewApplications
FDAModel = Union[ProductModel, EstablishmentModel, ApplicationModel]


class ProductsRepository(MultiTableRepository):
    """
    Repository for product verification and FDA database operations.
    Handles searches across all FDA-related tables.
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize products repository with all relevant models and session."""
        super().__init__(session)
        self.models = {
            'drug_products': DrugProducts,
            'food_products': FoodProducts,
            'drug_industry': DrugIndustry,
            'food_industry': FoodIndustry,
            'medical_device_industry': MedicalDeviceIndustry,
            'cosmetic_industry': CosmeticIndustry,
            'drug_applications': DrugsNewApplications
        }
    
    async def search_across_tables(self, search_criteria: Dict[str, Any]) -> Dict[str, List[FDAModel]]:
        """
        Search across all FDA tables for product verification.
        
        Args:
            search_criteria: Dictionary containing search parameters like:
                - registration_number: FDA registration number
                - license_number: License number for establishments
                - document_tracking_number: Application tracking number
                - brand_name: Product brand name
                - company_name: Company/establishment name
                - product_name: Generic product name
        
        Returns:
            Dictionary with table names as keys and lists of matching records
        """
        results = {}
        
        # Search drug products
        results['drug_products'] = await self._search_drug_products(self.session, search_criteria)
        
        # Search food products
        results['food_products'] = await self._search_food_products(self.session, search_criteria)
        
        # Search drug industry (establishments)
        results['drug_industry'] = await self._search_drug_industry(self.session, search_criteria)
        
        # Search food industry (establishments)
        results['food_industry'] = await self._search_food_industry(self.session, search_criteria)
        
        # Search medical device industry
        results['medical_device_industry'] = await self._search_medical_device_industry(self.session, search_criteria)
        
        # Search cosmetic industry
        results['cosmetic_industry'] = await self._search_cosmetic_industry(self.session, search_criteria)
        
        # Search drug applications
        results['drug_applications'] = await self._search_drug_applications(self.session, search_criteria)
        
        return results
    
    async def _search_drug_products(self, session: AsyncSession, criteria: Dict[str, Any]) -> List[DrugProducts]:
        """Search in drug products table."""
        conditions = []
        
        if criteria.get('registration_number'):
            conditions.append(DrugProducts.registration_number.ilike(f"%{criteria['registration_number']}%"))
        
        if criteria.get('brand_name'):
            conditions.append(DrugProducts.brand_name.ilike(f"%{criteria['brand_name']}%"))
        
        if criteria.get('generic_name'):
            conditions.append(DrugProducts.generic_name.ilike(f"%{criteria['generic_name']}%"))
        
        if criteria.get('company_name'):
            conditions.append(DrugProducts.manufacturer.ilike(f"%{criteria['company_name']}%"))
        
        if not conditions:
            return []
        
        query = select(DrugProducts).where(or_(*conditions)).limit(50)
        result = await session.execute(query)
        
        return result.scalars().all()
    
    async def _search_food_products(self, session: AsyncSession, criteria: Dict[str, Any]) -> List[FoodProducts]:
        """Search in food products table."""
        conditions = []
        
        if criteria.get('registration_number'):
            conditions.append(FoodProducts.registration_number.ilike(f"%{criteria['registration_number']}%"))
        
        if criteria.get('product_name'):
            conditions.append(FoodProducts.product_name.ilike(f"%{criteria['product_name']}%"))
        
        if criteria.get('brand_name'):
            conditions.append(FoodProducts.product_name.ilike(f"%{criteria['brand_name']}%"))
        
        if criteria.get('company_name'):
            conditions.append(FoodProducts.company_name.ilike(f"%{criteria['company_name']}%"))
        
        if not conditions:
            return []
        
        query = select(FoodProducts).where(or_(*conditions)).limit(50)
        result = await session.execute(query)
        
        return result.scalars().all()
    
    async def _search_drug_industry(self, session: AsyncSession, criteria: Dict[str, Any]) -> List[DrugIndustry]:
        """Search in drug industry establishments table."""
        conditions = []
        
        if criteria.get('license_number'):
            conditions.append(DrugIndustry.license_number.ilike(f"%{criteria['license_number']}%"))
        
        if criteria.get('company_name'):
            conditions.append(DrugIndustry.name_of_establishment.ilike(f"%{criteria['company_name']}%"))
        
        if not conditions:
            return []
        
        query = select(DrugIndustry).where(or_(*conditions)).limit(50)
        result = await session.execute(query)
        
        return result.scalars().all()
    
    async def _search_food_industry(self, session: AsyncSession, criteria: Dict[str, Any]) -> List[FoodIndustry]:
        """Search in food industry establishments table."""
        conditions = []
        
        if criteria.get('license_number'):
            conditions.append(FoodIndustry.license_number.ilike(f"%{criteria['license_number']}%"))
        
        if criteria.get('company_name'):
            conditions.append(FoodIndustry.name_of_establishment.ilike(f"%{criteria['company_name']}%"))
        
        if not conditions:
            return []
        
        query = select(FoodIndustry).where(or_(*conditions)).limit(50)
        result = await session.execute(query)
        
        return result.scalars().all()
    
    async def _search_medical_device_industry(self, session: AsyncSession, criteria: Dict[str, Any]) -> List[MedicalDeviceIndustry]:
        """Search in medical device industry establishments table."""
        conditions = []
        
        if criteria.get('license_number'):
            conditions.append(MedicalDeviceIndustry.license_number.ilike(f"%{criteria['license_number']}%"))
        
        if criteria.get('company_name'):
            conditions.append(MedicalDeviceIndustry.name_of_establishment.ilike(f"%{criteria['company_name']}%"))
        
        if not conditions:
            return []
        
        query = select(MedicalDeviceIndustry).where(or_(*conditions)).limit(50)
        result = await session.execute(query)
        
        return result.scalars().all()
    
    async def _search_cosmetic_industry(self, session: AsyncSession, criteria: Dict[str, Any]) -> List[CosmeticIndustry]:
        """Search in cosmetic industry establishments table."""
        conditions = []
        
        if criteria.get('license_number'):
            conditions.append(CosmeticIndustry.license_number.ilike(f"%{criteria['license_number']}%"))
        
        if criteria.get('company_name'):
            conditions.append(CosmeticIndustry.name_of_establishment.ilike(f"%{criteria['company_name']}%"))
        
        if not conditions:
            return []
        
        query = select(CosmeticIndustry).where(or_(*conditions)).limit(50)
        result = await session.execute(query)
        
        return result.scalars().all()
    
    async def _search_drug_applications(self, session: AsyncSession, criteria: Dict[str, Any]) -> List[DrugsNewApplications]:
        """Search in drug applications table."""
        conditions = []
        
        if criteria.get('document_tracking_number'):
            conditions.append(DrugsNewApplications.document_tracking_number.ilike(f"%{criteria['document_tracking_number']}%"))
        
        if criteria.get('brand_name'):
            conditions.append(DrugsNewApplications.brand_name.ilike(f"%{criteria['brand_name']}%"))
        
        if criteria.get('company_name'):
            conditions.append(DrugsNewApplications.applicant.ilike(f"%{criteria['company_name']}%"))
        
        if not conditions:
            return []
        
        query = select(DrugsNewApplications).where(or_(*conditions)).limit(50)
        result = await session.execute(query)
        
        return result.scalars().all()
    
    async def search_by_registration_number(self, registration_number: str) -> List[FDAModel]:
        """
        Search specifically by registration number across relevant tables.
        
        Args:
            registration_number: The registration number to search for
            
        Returns:
            List of matching products with their details
        """
        search_criteria = {'registration_number': registration_number}
        all_results = await self.search_across_tables(search_criteria)
        
        # Flatten results
        matches = []
        for table_name, results in all_results.items():
            matches.extend(results)
        
        return matches
    
    async def search_by_license_number(self, license_number: str) -> List[EstablishmentModel]:
        """
        Search specifically by license number across establishment tables.
        
        Args:
            license_number: The license number to search for
            
        Returns:
            List of matching establishments with their details
        """
        search_criteria = {'license_number': license_number}
        all_results = await self.search_across_tables(search_criteria)
        
        # Filter only establishment results (not products)
        establishment_tables = ['drug_industry', 'food_industry', 'medical_device_industry', 'cosmetic_industry']
        matches = []
        
        for table_name in establishment_tables:
            if table_name in all_results:
                matches.extend(all_results[table_name])
        
        return matches
    
    async def search_by_document_tracking_number(self, tracking_number: str) -> List[DrugsNewApplications]:
        """
        Search specifically by document tracking number in applications.
        
        Args:
            tracking_number: The document tracking number to search for
            
        Returns:
            List of matching applications with their details
        """
        search_criteria = {'document_tracking_number': tracking_number}
        all_results = await self.search_across_tables(search_criteria)
        
        matches = []
        if 'drug_applications' in all_results:
            matches.extend(all_results['drug_applications'])
        
        return matches
    
    async def fuzzy_search_by_product_info(self, product_info: Dict[str, Any]) -> List[FDAModel]:
        """
        Perform fuzzy search using multiple product information fields.
        
        Args:
            product_info: Dictionary containing product information like:
                - brand_name: Product brand name
                - product_name: Generic product name
                - company_name: Company/manufacturer name
                - registration_number: Registration number (if available)
        
        Returns:
            List of matching model instances (unsorted - service layer handles ranking)
        """
        all_results = await self.search_across_tables(product_info)
        
        # Flatten results - no scoring or sorting (that's service layer responsibility)
        matches = []
        for table_name, results in all_results.items():
            matches.extend(results)
        
        return matches
    
    
    def _model_to_dict(self, model_instance: FDAModel, table_name: str) -> Dict[str, Any]:
        """
        Convert a model instance to dictionary format for backward compatibility.
        
        Args:
            model_instance: SQLAlchemy model instance
            table_name: Name of the source table
            
        Returns:
            Dictionary representation of the model
        """
        # Base fields that all models have
        result = {
            'id': model_instance.id,
            'table_name': table_name,
            'source_table': table_name
        }
        
        # Add model-specific fields based on type
        if isinstance(model_instance, DrugProducts):
            result.update({
                'registration_number': model_instance.registration_number,
                'brand_name': model_instance.brand_name,
                'generic_name': model_instance.generic_name,
                'manufacturer': model_instance.manufacturer,
                'type': 'drug_product'
            })
        elif isinstance(model_instance, FoodProducts):
            result.update({
                'registration_number': model_instance.registration_number,
                'product_name': model_instance.product_name,
                'company_name': model_instance.company_name,
                'type': 'food_product'
            })
        elif isinstance(model_instance, DrugIndustry):
            result.update({
                'license_number': model_instance.license_number,
                'name_of_establishment': model_instance.name_of_establishment,
                'type': 'drug_industry'
            })
        elif isinstance(model_instance, FoodIndustry):
            result.update({
                'license_number': model_instance.license_number,
                'name_of_establishment': model_instance.name_of_establishment,
                'type': 'food_industry'
            })
        elif isinstance(model_instance, MedicalDeviceIndustry):
            result.update({
                'license_number': model_instance.license_number,
                'name_of_establishment': model_instance.name_of_establishment,
                'type': 'medical_device_industry'
            })
        elif isinstance(model_instance, CosmeticIndustry):
            result.update({
                'license_number': model_instance.license_number,
                'name_of_establishment': model_instance.name_of_establishment,
                'type': 'cosmetic_industry'
            })
        elif isinstance(model_instance, DrugsNewApplications):
            result.update({
                'document_tracking_number': model_instance.document_tracking_number,
                'brand_name': model_instance.brand_name,
                'applicant': model_instance.applicant,
                'application_type': model_instance.application_type,
                'type': 'drug_application'
            })
        
        return result