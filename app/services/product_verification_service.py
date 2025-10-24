"""
Product verification service layer.
Handles business logic for product verification, scoring, and ranking.
"""

import difflib
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.api.repository.products_repository import FDAModel, ProductsRepository
from app.models import (
    CosmeticIndustry,
    DrugIndustry,
    DrugProducts,
    DrugsNewApplications,
    FoodIndustry,
    FoodProducts,
    MedicalDeviceIndustry,
)


@dataclass
class ProductSearchResult:
    """
    Data Transfer Object for product search results with business logic applied.
    """

    model_instance: FDAModel
    relevance_score: float
    matched_fields: list[str]
    product_type: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        # Get the primary key value - models use registration_number, license_number, etc.
        primary_key = self._get_primary_key_value()

        result = {
            "id": primary_key,
            "relevance_score": self.relevance_score,
            "matched_fields": self.matched_fields,
            "type": self.product_type,
        }

        # Add model-specific fields
        if isinstance(self.model_instance, DrugProducts):
            result.update(
                {
                    "registration_number": self.model_instance.registration_number,
                    "brand_name": self.model_instance.brand_name,
                    "generic_name": self.model_instance.generic_name,
                    "manufacturer": self.model_instance.manufacturer,
                }
            )
        elif isinstance(self.model_instance, FoodProducts):
            result.update(
                {
                    "registration_number": self.model_instance.registration_number,
                    "brand_name": self.model_instance.brand_name,
                    "product_name": self.model_instance.product_name,
                    "company_name": self.model_instance.company_name,
                }
            )
        elif isinstance(
            self.model_instance,
            (DrugIndustry, FoodIndustry, MedicalDeviceIndustry, CosmeticIndustry),
        ):
            result.update(
                {
                    "license_number": self.model_instance.license_number,
                    "name_of_establishment": self.model_instance.name_of_establishment,
                }
            )
        elif isinstance(self.model_instance, DrugsNewApplications):
            result.update(
                {
                    "document_tracking_number": self.model_instance.document_tracking_number,
                    "brand_name": self.model_instance.brand_name,
                    "applicant_company": self.model_instance.applicant_company,
                    "application_type": self.model_instance.application_type,
                }
            )

        return result

    def _get_primary_key_value(self) -> str:
        """Get the primary key value from the model instance."""
        if isinstance(self.model_instance, (DrugProducts, FoodProducts)):
            return self.model_instance.registration_number
        if isinstance(
            self.model_instance,
            (DrugIndustry, FoodIndustry, MedicalDeviceIndustry, CosmeticIndustry),
        ):
            return self.model_instance.license_number
        if isinstance(self.model_instance, DrugsNewApplications):
            return self.model_instance.document_tracking_number
        # Fallback - try to get any available identifier
        for attr in [
            "registration_number",
            "license_number",
            "document_tracking_number",
        ]:
            if hasattr(self.model_instance, attr):
                value = getattr(self.model_instance, attr)
                if value:
                    return value
        return "unknown"


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

    async def search_and_rank_products(
        self, product_info: dict[str, Any]
    ) -> list[ProductSearchResult]:
        """
        Search for products and apply business logic for ranking.

        Args:
            product_info: Search criteria

        Returns:
            List of search results sorted by relevance
        """
        logger.debug(f"Searching products with criteria: {list(product_info.keys())}")

        # Repository handles data access only
        raw_results = await self.products_repo.fuzzy_search_by_product_info(
            product_info
        )
        logger.info(f"Repository returned {len(raw_results)} raw results")

        # Service layer applies business logic
        scored_results = []
        for model_instance in raw_results:
            score, matched_fields = self._calculate_relevance_score(
                model_instance, product_info
            )
            product_type = self._get_product_type(model_instance)

            scored_results.append(
                ProductSearchResult(
                    model_instance=model_instance,
                    relevance_score=score,
                    matched_fields=matched_fields,
                    product_type=product_type,
                )
            )

        # Sort by relevance score (highest first)
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)

        if scored_results:
            top_score = scored_results[0].relevance_score
            logger.info(
                f"Ranked {len(scored_results)} products, "
                f"top_score={top_score:.2f}, "
                f"top_type={scored_results[0].product_type}"
            )
        else:
            logger.info("No scored results found")

        return scored_results

    async def verify_product_by_id(self, product_id: str) -> list[ProductSearchResult]:
        """
        Verify a product by its ID with business logic applied.

        Optimized to use search_by_any_id which reduces queries from 21 to 7.

        Args:
            product_id: Product ID to verify (registration/license/tracking number)

        Returns:
            List of matching products with verification scores
        """
        logger.debug("Service: Verifying product by ID (optimized search)")

        # Get raw data from repository using optimized single method (7 queries instead of 21)
        all_matches = await self.products_repo.search_by_any_id(product_id)
        logger.info(f"Service: Found {len(all_matches)} matches for ID verification")

        # Apply business logic for exact vs partial matches
        results = []
        for model_instance in all_matches:
            score, matched_fields = self._calculate_id_match_score(
                model_instance, product_id
            )
            product_type = self._get_product_type(model_instance)

            results.append(
                ProductSearchResult(
                    model_instance=model_instance,
                    relevance_score=score,
                    matched_fields=matched_fields,
                    product_type=product_type,
                )
            )

        # Sort by relevance (exact matches first)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        if results:
            exact_matches = [r for r in results if r.relevance_score >= 1.0]
            logger.info(
                f"Service: ID verification complete - "
                f"exact_matches={len(exact_matches)}, "
                f"total_results={len(results)}"
            )
        else:
            logger.info("Service: No matches found for product ID")

        return results

    def _calculate_relevance_score(
        self, model_instance: FDAModel, search_info: dict[str, Any]
    ) -> tuple[float, list[str]]:
        """
        Calculate relevance score for a search result with business logic.

        Args:
            model_instance: Database model instance
            search_info: Original search criteria

        Returns:
            Tuple of (relevance_score, matched_fields)
        """
        logger.debug(f"Calculating relevance score for {type(model_instance).__name__}")
        score = 0.0
        matched_fields = []

        # Convert model to dict for easier field access
        model_dict = self._model_to_search_dict(model_instance)

        # Registration number match (highest weight)
        if (
            search_info.get("registration_number")
            and model_dict.get("registration_number")
            and (
                search_info["registration_number"].lower()
                in model_dict["registration_number"].lower()
            )
        ):
            score += 0.4  # 40% weight for registration number
            matched_fields.append("registration_number")

        # Brand name match with improved scoring
        if search_info.get("brand_name"):
            brand_fields = ["brand_name", "product_name"]
            for field in brand_fields:
                if model_dict.get(field):
                    search_brand = search_info["brand_name"].lower()
                    field_brand = model_dict[field].lower()

                    # Exact match (highest score)
                    if search_brand == field_brand:
                        score += 0.5  # 50% weight for exact brand match
                        matched_fields.append(field)
                        break
                    # Core brand substring match (e.g., "C2" in "C2 COOL & CLEAN")
                    if search_brand in field_brand or field_brand in search_brand:
                        # When short search term is in longer brand, prioritize brands with extra matching words
                        if search_brand in field_brand and len(field_brand) > len(search_brand):
                            # Base score for the substring match
                            base_score = 0.40

                            # Major bonus for additional words in brand that match product description
                            if search_info.get("product_description"):
                                prod_desc = search_info["product_description"].lower()
                                brand_words = set(field_brand.split())
                                desc_words = {word for word in prod_desc.split() if len(word) >= 3}
                                common_brand_desc = brand_words & desc_words

                                # Remove the search brand itself from the count
                                common_brand_desc.discard(search_brand)

                                if len(common_brand_desc) >= 2:
                                    base_score += 0.10
                                elif len(common_brand_desc) == 1:
                                    base_score += 0.05
                        else:
                            similarity = difflib.SequenceMatcher(None, search_brand, field_brand).ratio()

                            # Base score weighted by similarity: 0.3 to 0.45
                            base_score = 0.30 + (similarity * 0.15)

                        score += base_score
                        matched_fields.append(field)
                        break
                    # Word-level brand match
                    search_words = set(search_brand.split())
                    field_words = set(field_brand.split())
                    if search_words & field_words:
                        score += 0.3  # 30% weight for word-level brand match
                        matched_fields.append(field)
                        break

        # Company/establishment name match
        if search_info.get("company_name"):
            company_fields = [
                "company_name",
                "applicant_company",
                "name_of_establishment",
            ]
            for field in company_fields:
                if model_dict.get(field):
                    if search_info["company_name"].lower() in model_dict[field].lower():
                        score += 0.2  # 20% weight for company name
                        matched_fields.append(field)
                    break

        # Product description/name match - handle both product_description and legacy fields
        # Use word-level matching to handle word order variations
        product_description = (
            search_info.get("product_description")
            or search_info.get("generic_name")
            or search_info.get("product_name")
        )
        if product_description:
            product_fields = ["product_name", "generic_name"]
            for field in product_fields:
                if model_dict.get(field):
                    # Exact phrase match (highest score)
                    if product_description.lower() in model_dict[field].lower():
                        score += 0.30  # 30% weight for exact product description match
                        matched_fields.append(field)
                        break
                    # Reverse match (database product in search description)
                    if model_dict[field].lower() in product_description.lower():
                        score += 0.28  # 28% weight for reverse exact match
                        matched_fields.append(field)
                        break
                    # Word-level tokenized match (handles word order variations)
                    search_words = {
                        word.lower()
                        for word in product_description.strip().split()
                        if len(word) >= 3
                    }
                    field_words = {
                        word.lower()
                        for word in str(model_dict[field]).strip().split()
                        if len(word) >= 3
                    }

                    if search_words and field_words:
                        # Calculate word overlap ratio
                        common_words = search_words & field_words
                        overlap_ratio = len(common_words) / len(search_words)
                        
                        # Also calculate reverse overlap (important for longer database product names)
                        reverse_overlap_ratio = len(common_words) / len(field_words) if field_words else 0
                        
                        # Use the better of the two ratios
                        best_overlap = max(overlap_ratio, reverse_overlap_ratio)

                        # Award partial score based on word overlap
                        # Lowered threshold to 25% to handle OCR text with extra packaging info
                        if best_overlap >= 0.25:  # At least 25% of words match
                            # Enhanced scoring: more generous for high overlap
                            # 25% = 7.5%, 50% = 15%, 75% = 22.5%, 100% = 30%
                            score += (
                                0.30 * best_overlap
                            )  # Up to 30% weight for partial match
                            matched_fields.append(field)
                            break

        # Flavor/Key Term Bonus: Boost products where key terms match (e.g., "APPLE", "LEMON", "CHOCOLATE")
        # This helps differentiate between "APPLE GREEN TEA" and "GREEN APPLE" variants
        if product_description:
            # Extract key flavor/descriptor terms (usually important nouns/adjectives)
            flavor_keywords = {
                word.lower() for word in product_description.strip().split() 
                if len(word) >= 4 and word.lower() not in {
                    "flavored", "flavor", "drink", "juice", "plus", "with", "from"
                }
            }
            
            for field in ["product_name", "generic_name"]:
                if model_dict.get(field) and flavor_keywords:
                    field_lower = str(model_dict[field]).lower()
                    matching_keywords = [kw for kw in flavor_keywords if kw in field_lower]
                    
                    if matching_keywords:
                        # Boost score based on number of matching keywords
                        keyword_bonus = min(0.15, len(matching_keywords) * 0.05)
                        score += keyword_bonus
                        logger.debug(
                            f"Flavor keyword bonus: +{keyword_bonus:.2f} "
                            f"(matched: {', '.join(matching_keywords)})"
                        )
                        break

        # Context bonus: If this is a drug product and generic_name has good matches with product_description
        # This helps rank "Neozep" (nasal decongestant) above "NEO" (jalapeno) when both match brand
        if isinstance(model_instance, DrugProducts) and product_description and model_dict.get("generic_name"):
                search_words = {
                    word.lower()
                    for word in product_description.strip().split()
                    if len(word) >= 3
                }
                generic_words = {
                    word.lower()
                    for word in str(model_dict["generic_name"]).strip().split()
                    if len(word) >= 3
                }

                # If generic name contains terms from product description, boost score
                common_words = search_words & generic_words
                if len(common_words) >= 2:
                    # Significant overlap = strong context match
                    score += 0.1
                    logger.debug(
                        f"Drug context bonus: +0.1 ({len(common_words)} matching terms)"
                    )
                elif len(common_words) == 1:
                    # Some overlap = moderate context match
                    score += 0.05
                    logger.debug("Drug context bonus: +0.05 (1 matching term)")

        # Penalty for extra descriptors in database product when extracted data is simpler
        # E.g., if we extract "C2" but database has "C2 COOL & CLEAN", apply small penalty
        # This prevents wrong sub-brand matches when vision misses the sub-brand text
        if search_info.get("brand_name") and model_dict.get("brand_name"):
            search_brand = search_info["brand_name"].lower().strip()
            db_brand = model_dict["brand_name"].lower().strip()
            
            # If database brand is significantly longer and contains search brand as substring
            if search_brand in db_brand and len(db_brand) > len(search_brand) * 1.5:
                # Check if product description provides context that matches the extra brand terms
                has_context_match = False
                if product_description:
                    extra_brand_words = set(db_brand.split()) - set(search_brand.split())
                    desc_words = set(product_description.lower().split())
                    # If product description contains words from the extra brand terms, it's okay
                    if extra_brand_words & desc_words:
                        has_context_match = True
                
                if not has_context_match:
                    # Apply minor penalty for potential sub-brand mismatch
                    penalty = 0.05
                    score = max(0, score - penalty)
                    logger.debug(
                        f"Sub-brand length penalty: -{penalty:.2f} "
                        f"(search='{search_brand}', db='{db_brand}')"
                    )

        logger.debug(f"Score calculated: {score:.2f}, matched_fields={matched_fields}")
        return score, matched_fields

    def _calculate_id_match_score(
        self, model_instance: FDAModel, product_id: str
    ) -> tuple[float, list[str]]:
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
        if (
            model_dict.get("registration_number")
            and model_dict["registration_number"].lower() == product_id.lower()
        ):
            return 1.0, ["registration_number"]
        if (
            model_dict.get("license_number")
            and model_dict["license_number"].lower() == product_id.lower()
        ):
            return 1.0, ["license_number"]
        if (
            model_dict.get("document_tracking_number")
            and model_dict["document_tracking_number"].lower() == product_id.lower()
        ):
            return 1.0, ["document_tracking_number"]

        # Partial matches get lower scores
        score = 0.0
        if (
            model_dict.get("registration_number")
            and product_id.lower() in model_dict["registration_number"].lower()
        ):
            score = 0.8
            matched_fields.append("registration_number")
        elif (
            model_dict.get("license_number")
            and product_id.lower() in model_dict["license_number"].lower()
        ):
            score = 0.8
            matched_fields.append("license_number")
        elif (
            model_dict.get("document_tracking_number")
            and product_id.lower() in model_dict["document_tracking_number"].lower()
        ):
            score = 0.8
            matched_fields.append("document_tracking_number")

        return score, matched_fields

    def _model_to_search_dict(self, model_instance: FDAModel) -> dict[str, Any]:
        """
        Convert model instance to dictionary for search operations.

        Args:
            model_instance: SQLAlchemy model instance

        Returns:
            Dictionary with searchable fields
        """
        if isinstance(model_instance, DrugProducts):
            return {
                "registration_number": model_instance.registration_number,
                "brand_name": model_instance.brand_name,
                "generic_name": model_instance.generic_name,
                "manufacturer": model_instance.manufacturer,
            }
        if isinstance(model_instance, FoodProducts):
            return {
                "registration_number": model_instance.registration_number,
                "brand_name": model_instance.brand_name,
                "product_name": model_instance.product_name,
                "company_name": model_instance.company_name,
            }
        if isinstance(
            model_instance,
            (DrugIndustry, FoodIndustry, MedicalDeviceIndustry, CosmeticIndustry),
        ):
            return {
                "license_number": model_instance.license_number,
                "name_of_establishment": model_instance.name_of_establishment,
            }
        if isinstance(model_instance, DrugsNewApplications):
            return {
                "document_tracking_number": model_instance.document_tracking_number,
                "brand_name": model_instance.brand_name,
                "applicant_company": model_instance.applicant_company,
                "application_type": model_instance.application_type,
            }
        return None

        # return {}

    def _get_product_type(self, model_instance: FDAModel) -> str:
        """
        Get product type string for model instance.

        Args:
            model_instance: SQLAlchemy model instance

        Returns:
            Product type string
        """
        if isinstance(model_instance, DrugProducts):
            return "drug_product"
        if isinstance(model_instance, FoodProducts):
            return "food_product"
        if isinstance(model_instance, DrugIndustry):
            return "drug_industry"
        if isinstance(model_instance, FoodIndustry):
            return "food_industry"
        if isinstance(model_instance, MedicalDeviceIndustry):
            return "medical_device_industry"
        if isinstance(model_instance, CosmeticIndustry):
            return "cosmetic_industry"
        if isinstance(model_instance, DrugsNewApplications):
            return "drug_application"
        return None

        # return 'unknown'
