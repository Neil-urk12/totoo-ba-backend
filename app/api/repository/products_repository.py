"""
Products repository for FDA verification and product-related database operations.
Handles searches across multiple FDA database tables.
"""

from typing import Any

from loguru import logger
from sqlalchemy import func, or_, select
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

# Type aliases for better type safety
ProductModel = DrugProducts | FoodProducts
EstablishmentModel = (
    DrugIndustry | FoodIndustry | MedicalDeviceIndustry | CosmeticIndustry
)
ApplicationModel = DrugsNewApplications
FDAModel = ProductModel | EstablishmentModel | ApplicationModel


class ProductsRepository:
    """
    Repository for product verification and FDA database operations.
    Handles searches across all FDA-related tables.
    """

    def __init__(self, session: AsyncSession):
        """Initialize products repository with a database session."""
        self.session = session

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
        results["food_products"] = await self._search_products_fts(
            FoodProducts, "product_name", search_criteria
        )

        # Search drug products using full-text search (FTS)
        results["drug_products"] = await self._search_products_fts(
            DrugProducts, "generic_name", search_criteria
        )

        # Search drug industry (establishments)
        results["drug_industry"] = await self._search_establishments(
            DrugIndustry, search_criteria
        )

        # Search food industry (establishments)
        results["food_industry"] = await self._search_establishments(
            FoodIndustry, search_criteria
        )

        # Search medical device industry
        results["medical_device_industry"] = await self._search_establishments(
            MedicalDeviceIndustry, search_criteria
        )

        # Search cosmetic industry
        results["cosmetic_industry"] = await self._search_establishments(
            CosmeticIndustry, search_criteria
        )

        # Search drug applications
        results["drug_applications"] = await self._search_drug_applications(
            self.session, search_criteria
        )

        return results

    async def _search_products_by_registration(
        self, model: type[ProductModel], criteria: dict[str, Any]
    ) -> list[ProductModel]:
        """Find products containing a registration number."""
        if not criteria.get("registration_number"):
            return []
        query = (
            select(model)
            .where(
                model.registration_number.ilike(f"%{criteria['registration_number']}%")
            )
            .limit(50)
        )
        result = await self.session.execute(query)
        return result.scalars().all()

    async def _search_products_fts(
        self, model: type[ProductModel], description_key: str, criteria: dict[str, Any]
    ) -> list[ProductModel]:
        """Search indexed product text, then merge fuzzy brand matches."""
        logger.debug(f"Repository: Searching {model.__tablename__} using FTS")

        # Build search terms from criteria
        search_terms = []

        # Helper function to clean search terms (remove special characters that confuse tsquery)
        def clean_term(term: str) -> str:
            """Remove special characters that have meaning in tsquery"""
            # Replace & and other tsquery operators with spaces
            return (
                term.replace("&", " ")
                .replace("|", " ")
                .replace("!", " ")
                .replace("(", " ")
                .replace(")", " ")
            )

        # Collect all searchable terms
        if criteria.get("registration_number"):
            search_terms.append(clean_term(criteria["registration_number"]))

        if criteria.get("brand_name"):
            search_terms.append(clean_term(criteria["brand_name"]))

        # For the model description and product_description, only add individual words (not the full string)
        # to avoid duplication and improve matching
        if criteria.get(description_key):
            words = clean_term(criteria[description_key]).strip().split()
            search_terms.extend([word for word in words if len(word) >= 3])

        if criteria.get("product_description"):
            # Split product description into words for better matching
            words = clean_term(criteria["product_description"]).strip().split()
            search_terms.extend([word for word in words if len(word) >= 3])

        if criteria.get("company_name"):
            search_terms.append(clean_term(criteria["company_name"]))

        if criteria.get("manufacturer"):
            search_terms.append(clean_term(criteria["manufacturer"]))

        if not search_terms:
            return []

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            term_lower = term.lower()
            if term_lower not in seen and term.strip():
                seen.add(term_lower)
                unique_terms.append(term)

        # Create search query using OR logic (at least one term must match)
        # Use websearch_to_tsquery which allows 'or' syntax for better partial matching
        # Join terms with ' OR ' so PostgreSQL matches records with ANY of these terms
        search_string = " OR ".join(unique_terms)

        # Use the pre-generated search_vector column with GIN index
        # This is MUCH faster than generating tsvector on-the-fly
        # The @@ operator checks if tsvector matches tsquery
        # Using websearch_to_tsquery allows OR logic for flexible matching

        query = (
            select(model)
            .where(
                # Use the indexed search_vector column directly
                model.search_vector.op("@@")(
                    func.websearch_to_tsquery("english", search_string)
                )
            )
            # Order by relevance using ts_rank with the indexed column
            # Records matching more terms will rank higher
            .order_by(
                func.ts_rank(
                    model.search_vector,
                    func.websearch_to_tsquery("english", search_string),
                ).desc()
            )
            .limit(50)
        )

        logger.debug(
            f"Executing FTS query on {model.__tablename__}: {search_string[:100]}..."
        )
        result = await self.session.execute(query)
        results = result.scalars().all()

        logger.info(
            f"Repository: {model.__tablename__} FTS returned {len(results)} results"
        )

        # Fuzzy matching fallback for brand names (handles text recognition errors like "Neozept" vs "Neozep")
        # Always run when brand_name is provided to catch typos even if FTS found other matches
        if criteria.get("brand_name"):
            brand_name = criteria["brand_name"]

            # Check if any FTS result actually matched the brand name closely
            brand_matched = any(
                result.brand_name
                and (
                    brand_name.lower() in result.brand_name.lower()
                    or result.brand_name.lower() in brand_name.lower()
                )
                for result in results[:10]  # Check top 10 FTS results
            )

            # Run fuzzy search if brand wasn't matched or we have few results
            if not brand_matched or len(results) < 10:
                logger.debug(
                    f"Applying fuzzy brand search for: {brand_name} (brand_matched={brand_matched})"
                )

                # Use PostgreSQL trigram similarity (pg_trgm extension)
                # similarity() returns 0-1 score (1 = exact match, 0 = no similarity)
                # Threshold of 0.3 catches typos while filtering noise
                fuzzy_query = (
                    select(model)
                    .where(func.similarity(model.brand_name, brand_name) > 0.3)
                    .order_by(func.similarity(model.brand_name, brand_name).desc())
                    .limit(20)
                )

                fuzzy_result = await self.session.execute(fuzzy_query)
                fuzzy_matches = fuzzy_result.scalars().all()

                # Merge results, avoiding duplicates
                existing_ids = {r.registration_number for r in results}
                for match in fuzzy_matches:
                    if match.registration_number not in existing_ids:
                        results.append(match)
                        existing_ids.add(match.registration_number)

                logger.info(
                    f"Repository: Added {len(fuzzy_matches)} fuzzy matches, total: {len(results)}"
                )
            else:
                logger.debug(
                    "Skipping fuzzy search - brand already matched in FTS results"
                )

        return results

    async def _search_establishments(
        self, model: type[EstablishmentModel], criteria: dict[str, Any]
    ) -> list[EstablishmentModel]:
        """Search establishments by license number or company name."""
        conditions = []
        if criteria.get("license_number"):
            conditions.append(
                model.license_number.ilike(f"%{criteria['license_number']}%")
            )
        if criteria.get("company_name"):
            conditions.append(
                model.name_of_establishment.ilike(f"%{criteria['company_name']}%")
            )
        if not conditions:
            return []
        query = select(model).where(or_(*conditions)).limit(50)
        result = await self.session.execute(query)
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
                DrugsNewApplications.applicant_company.ilike(
                    f"%{criteria['company_name']}%"
                )
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
        logger.debug(f"Repository: Optimized ID search for: {id_value[:20]}...")
        matches = []

        # Search products for registration_number (2 queries)
        matches.extend(
            await self._search_products_by_registration(
                DrugProducts, {"registration_number": id_value}
            )
        )
        matches.extend(
            await self._search_products_by_registration(
                FoodProducts, {"registration_number": id_value}
            )
        )

        # Search establishments for license_number (4 queries)
        matches.extend(
            await self._search_establishments(
                DrugIndustry, {"license_number": id_value}
            )
        )
        matches.extend(
            await self._search_establishments(
                FoodIndustry, {"license_number": id_value}
            )
        )
        matches.extend(
            await self._search_establishments(
                MedicalDeviceIndustry, {"license_number": id_value}
            )
        )
        matches.extend(
            await self._search_establishments(
                CosmeticIndustry, {"license_number": id_value}
            )
        )

        # Search applications for document_tracking_number (1 query)
        matches.extend(
            await self._search_drug_applications(
                self.session, {"document_tracking_number": id_value}
            )
        )

        logger.info(
            f"Repository: Optimized ID search complete - "
            f"found {len(matches)} matches across 7 queries"
        )

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
        logger.debug(
            f"Repository: Fuzzy search with {len(product_info)} criteria: "
            f"{list(product_info.keys())}"
        )
        all_results = await self.search_across_tables(product_info)

        # Flatten results - no scoring or sorting (that's service layer responsibility)
        matches = []
        for table_name, results in all_results.items():
            if results:
                logger.debug(
                    f"Repository: {table_name} returned {len(results)} results"
                )
            matches.extend(results)

        logger.info(f"Repository: Fuzzy search complete - {len(matches)} total matches")

        return matches
