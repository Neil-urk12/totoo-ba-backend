"""
Products repository for FDA verification and product-related database operations.
Handles searches across multiple FDA database tables.
"""

from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    CosmeticIndustry,
    DrugIndustry,
    DrugProducts,
    DrugsNewApplications,
    FoodIndustry,
    FoodProducts,
    MedicalDeviceIndustry,
)

from .database_repository import MultiTableRepository

# Type aliases for better type safety
ProductModel = DrugProducts | FoodProducts
EstablishmentModel = (
    DrugIndustry | FoodIndustry | MedicalDeviceIndustry | CosmeticIndustry
)
ApplicationModel = DrugsNewApplications
FDAModel = ProductModel | EstablishmentModel | ApplicationModel


class ProductsRepository(MultiTableRepository):
    """
    Repository for product verification and FDA database operations.
    Handles searches across all FDA-related tables.
    """

    def __init__(self, session: AsyncSession):
        """Initialize products repository with all relevant models and session."""
        super().__init__(session)
        self.models = {
            "drug_products": DrugProducts,
            "food_products": FoodProducts,
            "drug_industry": DrugIndustry,
            "food_industry": FoodIndustry,
            "medical_device_industry": MedicalDeviceIndustry,
            "cosmetic_industry": CosmeticIndustry,
            "drug_applications": DrugsNewApplications,
        }

    async def search_across_tables(
        self, search_criteria: dict[str, Any]
    ) -> dict[str, list[FDAModel]]:
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

        # Search food products using full-text search (FTS)
        # NOTE: This uses the search_vector tsvector column with GIN index
        # To use the old ILIKE-based search, change to: _search_food_products
        results["food_products"] = await self._search_food_products_fts(
            self.session, search_criteria
        )

        # Search drug products using full-text search (FTS)
        # NOTE: This uses the search_vector tsvector column with GIN index
        # To use the old ILIKE-based search, change to: _search_drug_products
        results["drug_products"] = await self._search_drug_products_fts(
            self.session, search_criteria
        )

        # Search drug industry (establishments)
        results["drug_industry"] = await self._search_drug_industry(
            self.session, search_criteria
        )

        # Search food industry (establishments)
        results["food_industry"] = await self._search_food_industry(
            self.session, search_criteria
        )

        # Search medical device industry
        results["medical_device_industry"] = await self._search_medical_device_industry(
            self.session, search_criteria
        )

        # Search cosmetic industry
        results["cosmetic_industry"] = await self._search_cosmetic_industry(
            self.session, search_criteria
        )

        # Search drug applications
        results["drug_applications"] = await self._search_drug_applications(
            self.session, search_criteria
        )

        return results

    async def _search_drug_products(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[DrugProducts]:
        """Search in drug products table with word-level tokenized matching."""

        # Build separate condition groups for AND logic
        brand_conditions = []
        product_conditions = []
        other_conditions = []

        if criteria.get("registration_number"):
            other_conditions.append(
                DrugProducts.registration_number.ilike(
                    f"%{criteria['registration_number']}%"
                )
            )

        if criteria.get("brand_name"):
            brand = criteria["brand_name"]
            if len(brand) <= 3:
                brand_conditions.append(
                    or_(
                        DrugProducts.brand_name.ilike(f"{brand} %"),
                        DrugProducts.brand_name.ilike(f"% {brand} %"),
                        DrugProducts.brand_name.ilike(f"% {brand}"),
                        DrugProducts.brand_name.ilike(f"{brand}"),
                    )
                )
            else:
                brand_conditions.append(DrugProducts.brand_name.ilike(f"%{brand}%"))

        if criteria.get("generic_name"):
            product_conditions.append(
                DrugProducts.generic_name.ilike(f"%{criteria['generic_name']}%")
            )

        if criteria.get("product_description"):
            words = criteria["product_description"].strip().split()
            if words:
                word_conditions = []
                for word in words:
                    if len(word) >= 3:
                        word_conditions.append(
                            DrugProducts.generic_name.ilike(f"%{word}%")
                        )
                if word_conditions:
                    product_conditions.append(or_(*word_conditions))

        if criteria.get("company_name"):
            other_conditions.append(
                DrugProducts.manufacturer.ilike(f"%{criteria['company_name']}%")
            )

        if criteria.get("manufacturer"):
            other_conditions.append(
                DrugProducts.manufacturer.ilike(f"%{criteria['manufacturer']}%")
            )

        # Build final query
        all_conditions = []

        if brand_conditions:
            all_conditions.append(
                or_(*brand_conditions)
                if len(brand_conditions) > 1
                else brand_conditions[0]
            )

        if product_conditions:
            all_conditions.append(
                or_(*product_conditions)
                if len(product_conditions) > 1
                else product_conditions[0]
            )

        if other_conditions:
            all_conditions.extend(other_conditions)

        if not all_conditions:
            return []

        # Use AND if we have brand + product, otherwise OR
        if brand_conditions and product_conditions:
            query = select(DrugProducts).where(and_(*all_conditions)).limit(50)
        else:
            query = select(DrugProducts).where(or_(*all_conditions)).limit(50)

        result = await session.execute(query)
        results = result.scalars().all()

        if results:
            for _i, _prod in enumerate(results[:5]):
                pass

        return results

    async def _search_drug_products_fts(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[DrugProducts]:
        """Search in drug products table using full-text search with tsvector.

        This method uses PostgreSQL's full-text search capabilities with the
        pre-generated search_vector column and GIN index for optimal performance.

        Args:
            session: AsyncSession for database operations
            criteria: Search criteria dictionary

        Returns:
            List of matching DrugProducts ordered by relevance
        """

        # Build search terms from criteria
        search_terms = []

        # Collect all searchable terms
        if criteria.get("registration_number"):
            search_terms.append(criteria["registration_number"])

        if criteria.get("brand_name"):
            search_terms.append(criteria["brand_name"])

        if criteria.get("generic_name"):
            search_terms.append(criteria["generic_name"])

        if criteria.get("product_description"):
            # Split product description into words for better matching
            words = criteria["product_description"].strip().split()
            search_terms.extend([word for word in words if len(word) >= 3])

        if criteria.get("company_name"):
            search_terms.append(criteria["company_name"])

        if criteria.get("manufacturer"):
            search_terms.append(criteria["manufacturer"])

        if not search_terms:
            return []

        # Create search query using plainto_tsquery (handles plain text better)
        # Join terms with spaces - PostgreSQL will handle them intelligently
        search_string = " ".join(search_terms)

        # Use the pre-generated search_vector column with GIN index
        # This is MUCH faster than generating tsvector on-the-fly
        # The @@ operator checks if tsvector matches tsquery

        query = (
            select(DrugProducts)
            .where(
                # Use the indexed search_vector column directly
                DrugProducts.search_vector.op("@@")(
                    func.plainto_tsquery("english", search_string)
                )
            )
            # Order by relevance using ts_rank with the indexed column
            .order_by(
                func.ts_rank(
                    DrugProducts.search_vector,
                    func.plainto_tsquery("english", search_string),
                ).desc()
            )
            .limit(50)
        )

        result = await session.execute(query)
        results = result.scalars().all()

        if results:
            for _i, _prod in enumerate(results[:5]):
                pass

        return results

    async def _search_food_products_fts(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[FoodProducts]:
        """Search in food products table using full-text search with tsvector.

        This method uses PostgreSQL's full-text search capabilities with the
        pre-generated search_vector column and GIN index for optimal performance.

        Args:
            session: AsyncSession for database operations
            criteria: Search criteria dictionary

        Returns:
            List of matching FoodProducts ordered by relevance
        """

        # Build search terms from criteria
        search_terms = []

        # Collect all searchable terms
        if criteria.get("registration_number"):
            search_terms.append(criteria["registration_number"])

        if criteria.get("brand_name"):
            search_terms.append(criteria["brand_name"])

        if criteria.get("product_name"):
            search_terms.append(criteria["product_name"])

        if criteria.get("product_description"):
            # Split product description into words for better matching
            words = criteria["product_description"].strip().split()
            search_terms.extend([word for word in words if len(word) >= 3])

        if criteria.get("company_name"):
            search_terms.append(criteria["company_name"])

        if criteria.get("manufacturer"):
            search_terms.append(criteria["manufacturer"])

        if not search_terms:
            return []

        # Create search query using plainto_tsquery (handles plain text better)
        # Join terms with spaces - PostgreSQL will handle them intelligently
        search_string = " ".join(search_terms)

        # Use the pre-generated search_vector column with GIN index
        # This is MUCH faster than generating tsvector on-the-fly
        # The @@ operator checks if tsvector matches tsquery

        query = (
            select(FoodProducts)
            .where(
                # Use the indexed search_vector column directly
                FoodProducts.search_vector.op("@@")(
                    func.plainto_tsquery("english", search_string)
                )
            )
            # Order by relevance using ts_rank with the indexed column
            .order_by(
                func.ts_rank(
                    FoodProducts.search_vector,
                    func.plainto_tsquery("english", search_string),
                ).desc()
            )
            .limit(50)
        )

        result = await session.execute(query)
        results = result.scalars().all()

        if results:
            for _i, _prod in enumerate(results[:5]):
                pass

        return results

    async def _search_food_products(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[FoodProducts]:
        """Search in food products table with word-level tokenized matching."""

        # Build separate condition groups for AND logic
        brand_conditions = []
        product_conditions = []
        other_conditions = []

        if criteria.get("registration_number"):
            other_conditions.append(
                FoodProducts.registration_number.ilike(
                    f"%{criteria['registration_number']}%"
                )
            )

        # Brand name matching (required if brand is provided)
        if criteria.get("brand_name"):
            brand = criteria["brand_name"]
            if len(brand) <= 3:
                # Word boundary matching for short brands
                brand_conditions.append(
                    or_(
                        FoodProducts.brand_name.ilike(f"{brand} %"),
                        FoodProducts.brand_name.ilike(f"% {brand} %"),
                        FoodProducts.brand_name.ilike(f"% {brand}"),
                        FoodProducts.brand_name.ilike(f"{brand}"),
                    )
                )
            else:
                brand_conditions.append(FoodProducts.brand_name.ilike(f"%{brand}%"))

        # Product name/description matching
        if criteria.get("product_name"):
            product_conditions.append(
                FoodProducts.product_name.ilike(f"%{criteria['product_name']}%")
            )

        if criteria.get("product_description"):
            # Word-level tokenized matching for product description
            words = criteria["product_description"].strip().split()
            if words:
                # Require at least 2 out of 3 words to match (for flexibility)
                word_conditions = []
                for word in words:
                    if len(word) >= 3:
                        word_conditions.append(
                            FoodProducts.product_name.ilike(f"%{word}%")
                        )

                if word_conditions:
                    # Use OR for individual words, but they'll be ANDed with brand
                    product_conditions.append(or_(*word_conditions))

        if criteria.get("company_name"):
            other_conditions.append(
                FoodProducts.company_name.ilike(f"%{criteria['company_name']}%")
            )

        if criteria.get("manufacturer"):
            other_conditions.append(
                FoodProducts.company_name.ilike(f"%{criteria['manufacturer']}%")
            )

        # Build final query with AND logic between groups
        all_conditions = []

        # If brand is specified, it MUST match
        if brand_conditions:
            all_conditions.append(
                or_(*brand_conditions)
                if len(brand_conditions) > 1
                else brand_conditions[0]
            )

        # If product description is specified, at least one word must match
        if product_conditions:
            all_conditions.append(
                or_(*product_conditions)
                if len(product_conditions) > 1
                else product_conditions[0]
            )

        # Other conditions are optional (OR)
        if other_conditions:
            all_conditions.extend(other_conditions)

        if not all_conditions:
            return []


        # Use AND if we have brand + product, otherwise OR
        if brand_conditions and product_conditions:
            # Brand AND Product (both must match)
            query = select(FoodProducts).where(and_(*all_conditions)).limit(50)
        else:
            # Fallback to OR if only one type
            query = select(FoodProducts).where(or_(*all_conditions)).limit(50)

        result = await session.execute(query)
        results = result.scalars().all()
        if results:
            for _i, _prod in enumerate(results[:5]):
                pass

        return results

    async def _search_drug_industry(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[DrugIndustry]:
        """Search in drug industry establishments table."""
        conditions = []

        if criteria.get("license_number"):
            conditions.append(
                DrugIndustry.license_number.ilike(f"%{criteria['license_number']}%")
            )

        if criteria.get("company_name"):
            conditions.append(
                DrugIndustry.name_of_establishment.ilike(
                    f"%{criteria['company_name']}%"
                )
            )

        if not conditions:
            return []

        query = select(DrugIndustry).where(or_(*conditions)).limit(50)
        result = await session.execute(query)

        return result.scalars().all()

    async def _search_food_industry(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[FoodIndustry]:
        """Search in food industry establishments table."""
        conditions = []

        if criteria.get("license_number"):
            conditions.append(
                FoodIndustry.license_number.ilike(f"%{criteria['license_number']}%")
            )

        if criteria.get("company_name"):
            conditions.append(
                FoodIndustry.name_of_establishment.ilike(
                    f"%{criteria['company_name']}%"
                )
            )

        if not conditions:
            return []

        query = select(FoodIndustry).where(or_(*conditions)).limit(50)
        result = await session.execute(query)

        return result.scalars().all()

    async def _search_medical_device_industry(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[MedicalDeviceIndustry]:
        """Search in medical device industry establishments table."""
        conditions = []

        if criteria.get("license_number"):
            conditions.append(
                MedicalDeviceIndustry.license_number.ilike(
                    f"%{criteria['license_number']}%"
                )
            )

        if criteria.get("company_name"):
            conditions.append(
                MedicalDeviceIndustry.name_of_establishment.ilike(
                    f"%{criteria['company_name']}%"
                )
            )

        if not conditions:
            return []

        query = select(MedicalDeviceIndustry).where(or_(*conditions)).limit(50)
        result = await session.execute(query)

        return result.scalars().all()

    async def _search_cosmetic_industry(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[CosmeticIndustry]:
        """Search in cosmetic industry establishments table."""
        conditions = []

        if criteria.get("license_number"):
            conditions.append(
                CosmeticIndustry.license_number.ilike(f"%{criteria['license_number']}%")
            )

        if criteria.get("company_name"):
            conditions.append(
                CosmeticIndustry.name_of_establishment.ilike(
                    f"%{criteria['company_name']}%"
                )
            )

        if not conditions:
            return []

        query = select(CosmeticIndustry).where(or_(*conditions)).limit(50)
        result = await session.execute(query)

        return result.scalars().all()

    async def _search_drug_applications(
        self, session: AsyncSession, criteria: dict[str, Any]
    ) -> list[DrugsNewApplications]:
        """Search in drug applications table."""
        conditions = []

        if criteria.get("document_tracking_number"):
            conditions.append(
                DrugsNewApplications.document_tracking_number.ilike(
                    f"%{criteria['document_tracking_number']}%"
                )
            )

        if criteria.get("brand_name"):
            conditions.append(
                DrugsNewApplications.brand_name.ilike(f"%{criteria['brand_name']}%")
            )

        if criteria.get("company_name"):
            conditions.append(
                DrugsNewApplications.applicant_company.ilike(f"%{criteria['company_name']}%")
            )

        if not conditions:
            return []

        query = select(DrugsNewApplications).where(or_(*conditions)).limit(50)
        result = await session.execute(query)

        return result.scalars().all()

    async def search_by_any_id(self, id_value: str) -> list[FDAModel]:
        """
        Optimized search that queries only relevant tables for any ID type.
        Reduces queries from 21 to 7 by smartly targeting tables based on ID fields.

        This method searches:
        - Products tables (drug, food) for registration_number
        - Establishment tables (drug, food, medical, cosmetic) for license_number
        - Applications table for document_tracking_number

        Args:
            id_value: The ID to search for (registration/license/tracking number)

        Returns:
            List of all matching records across relevant tables
        """
        matches = []

        # Search products for registration_number (2 queries)
        matches.extend(
            await self._search_drug_products(
                self.session, {"registration_number": id_value}
            )
        )
        matches.extend(
            await self._search_food_products(
                self.session, {"registration_number": id_value}
            )
        )

        # Search establishments for license_number (4 queries)
        matches.extend(
            await self._search_drug_industry(self.session, {"license_number": id_value})
        )
        matches.extend(
            await self._search_food_industry(self.session, {"license_number": id_value})
        )
        matches.extend(
            await self._search_medical_device_industry(
                self.session, {"license_number": id_value}
            )
        )
        matches.extend(
            await self._search_cosmetic_industry(
                self.session, {"license_number": id_value}
            )
        )

        # Search applications for document_tracking_number (1 query)
        matches.extend(
            await self._search_drug_applications(
                self.session, {"document_tracking_number": id_value}
            )
        )

        return matches

    async def search_by_registration_number(
        self, registration_number: str
    ) -> list[FDAModel]:
        """
        Search specifically by registration number across relevant tables.

        Args:
            registration_number: The registration number to search for

        Returns:
            List of matching products with their details
        """
        search_criteria = {"registration_number": registration_number}
        all_results = await self.search_across_tables(search_criteria)

        # Flatten results
        matches = []
        for _table_name, results in all_results.items():
            matches.extend(results)

        return matches

    async def search_by_license_number(
        self, license_number: str
    ) -> list[EstablishmentModel]:
        """
        Search specifically by license number across establishment tables.

        Args:
            license_number: The license number to search for

        Returns:
            List of matching establishments with their details
        """
        search_criteria = {"license_number": license_number}
        all_results = await self.search_across_tables(search_criteria)

        # Filter only establishment results (not products)
        establishment_tables = [
            "drug_industry",
            "food_industry",
            "medical_device_industry",
            "cosmetic_industry",
        ]
        matches = []

        for table_name in establishment_tables:
            if table_name in all_results:
                matches.extend(all_results[table_name])

        return matches

    async def search_by_document_tracking_number(
        self, tracking_number: str
    ) -> list[DrugsNewApplications]:
        """
        Search specifically by document tracking number in applications.

        Args:
            tracking_number: The document tracking number to search for

        Returns:
            List of matching applications with their details
        """
        search_criteria = {"document_tracking_number": tracking_number}
        all_results = await self.search_across_tables(search_criteria)

        matches = []
        if "drug_applications" in all_results:
            matches.extend(all_results["drug_applications"])

        return matches

    async def fuzzy_search_by_product_info(
        self, product_info: dict[str, Any]
    ) -> list[FDAModel]:
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
        for _table_name, results in all_results.items():
            matches.extend(results)

        return matches

    def _model_to_dict(
        self, model_instance: FDAModel, table_name: str
    ) -> dict[str, Any]:
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
            "id": model_instance.id,
            "table_name": table_name,
            "source_table": table_name,
        }

        # Add model-specific fields based on type
        if isinstance(model_instance, DrugProducts):
            result.update(
                {
                    "registration_number": model_instance.registration_number,
                    "brand_name": model_instance.brand_name,
                    "generic_name": model_instance.generic_name,
                    "manufacturer": model_instance.manufacturer,
                    "type": "drug_product",
                }
            )
        elif isinstance(model_instance, FoodProducts):
            result.update(
                {
                    "registration_number": model_instance.registration_number,
                    "product_name": model_instance.product_name,
                    "company_name": model_instance.company_name,
                    "type": "food_product",
                }
            )
        elif isinstance(model_instance, DrugIndustry):
            result.update(
                {
                    "license_number": model_instance.license_number,
                    "name_of_establishment": model_instance.name_of_establishment,
                    "type": "drug_industry",
                }
            )
        elif isinstance(model_instance, FoodIndustry):
            result.update(
                {
                    "license_number": model_instance.license_number,
                    "name_of_establishment": model_instance.name_of_establishment,
                    "type": "food_industry",
                }
            )
        elif isinstance(model_instance, MedicalDeviceIndustry):
            result.update(
                {
                    "license_number": model_instance.license_number,
                    "name_of_establishment": model_instance.name_of_establishment,
                    "type": "medical_device_industry",
                }
            )
        elif isinstance(model_instance, CosmeticIndustry):
            result.update(
                {
                    "license_number": model_instance.license_number,
                    "name_of_establishment": model_instance.name_of_establishment,
                    "type": "cosmetic_industry",
                }
            )
        elif isinstance(model_instance, DrugsNewApplications):
            result.update(
                {
                    "document_tracking_number": model_instance.document_tracking_number,
                    "brand_name": model_instance.brand_name,
                    "applicant_company": model_instance.applicant_company,
                    "application_type": model_instance.application_type,
                    "type": "drug_application",
                }
            )

        return result
