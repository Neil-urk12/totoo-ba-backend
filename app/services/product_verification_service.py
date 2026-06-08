"""
Product verification service layer.
Handles business logic for product verification, scoring, and ranking.
"""

import difflib
from dataclasses import dataclass
from typing import Any

from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from app.api.repository.products_repository import FDAModel, ProductsRepository
from app.utils.helpers import normalize_string
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
class VerificationOutcome:
    """Final verdict for text-by-ID product verification."""

    product_id: str
    is_verified: bool
    message: str
    details: dict[str, Any]


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

    async def verify_product_by_id(self, product_id: str) -> VerificationOutcome:
        """
        Verify a product by its ID and return a final verification outcome.

        Optimized to use search_by_any_id which reduces queries from 21 to 7.

        Args:
            product_id: Product ID to verify (registration/license/tracking number)

        Returns:
            VerificationOutcome with verdict, message, and details
        """
        product_id = product_id.strip()

        if not product_id or len(product_id) < 3:
            logger.warning(f"Invalid product ID provided: length={len(product_id)}")
            return VerificationOutcome(
                product_id=product_id,
                is_verified=False,
                message=(
                    f"Invalid product ID: {product_id}. "
                    "Product ID must be at least 3 characters long."
                ),
                details={"error_code": "INVALID_ID"},
            )

        try:
            logger.debug(f"Searching for product ID: {product_id}")
            ranked_results = await self._rank_id_matches(product_id)
            all_matches = [result.to_dict() for result in ranked_results]
            logger.debug(f"Found {len(all_matches)} potential matches for product ID")
            return self._outcome_from_id_matches(product_id, all_matches)
        except SQLAlchemyError as e:
            logger.error(f"Database error during verification for ID={product_id}: {e!s}")
            logger.exception("Full traceback:")
            return VerificationOutcome(
                product_id=product_id,
                is_verified=False,
                message="Error during verification: Database query failed",
                details={
                    "error_code": "DATABASE_ERROR",
                    "error_message": "Internal server error occurred during verification",
                    "verification_method": "repository_database_lookup",
                },
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during verification for ID={product_id}: "
                f"{type(e).__name__}: {e!s}"
            )
            logger.exception("Full traceback:")
            return VerificationOutcome(
                product_id=product_id,
                is_verified=False,
                message="Error during verification: Unexpected internal error",
                details={
                    "error_code": "INTERNAL_ERROR",
                    "error_message": "An unexpected error occurred during verification",
                    "verification_method": "repository_database_lookup",
                },
            )

    async def _rank_id_matches(self, product_id: str) -> list[ProductSearchResult]:
        """Search and rank ID matches from the repository."""
        logger.debug("Service: Verifying product by ID (optimized search)")

        all_matches = await self.products_repo.search_by_any_id(product_id)
        logger.info(f"Service: Found {len(all_matches)} matches for ID verification")

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

    def _outcome_from_id_matches(
        self, product_id: str, all_matches: list[dict[str, Any]]
    ) -> VerificationOutcome:
        """Build a verification outcome from ranked search results."""
        normalized_product_id = normalize_string(product_id)
        details: dict[str, Any] = {
            "verification_method": "repository_database_lookup",
            "search_results_count": len(all_matches),
        }

        exact_matches: list[dict[str, Any]] = []
        partial_matches: list[dict[str, Any]] = []

        for match in all_matches:
            matched_field = self._exact_id_matched_field(match, normalized_product_id)
            if matched_field:
                exact_matches.append(
                    {
                        "product": match,
                        "matched_field": matched_field,
                        "relevance_score": match.get("relevance_score", 1.0),
                    }
                )
                continue

            relevance = match.get("relevance_score", 0.0)
            if relevance >= 0.8:
                partial_matches.append(
                    {"product": match, "relevance_score": relevance}
                )

        if exact_matches:
            best_match = exact_matches[0]
            product_info = best_match["product"]
            logger.info(
                f"Product verified: ID={product_id}, type={product_info.get('type')}, "
                f"matched_field={best_match['matched_field']}"
            )
            details.update(
                {
                    "verified_product": product_info,
                    "matched_field": best_match["matched_field"],
                    "exact_match": True,
                    "confidence_score": 100,
                }
            )
            return VerificationOutcome(
                product_id=product_id,
                is_verified=True,
                message=self._verified_message_for_product(product_info),
                details=details,
            )

        if partial_matches:
            best_partial = partial_matches[0]
            logger.warning(
                f"Partial match found for ID={product_id}, "
                f"relevance={best_partial['relevance_score']:.0%}, "
                f"count={len(partial_matches)}"
            )
            details.update(
                {
                    "possible_matches": partial_matches[:3],
                    "exact_match": False,
                    "confidence_score": int(best_partial["relevance_score"] * 100),
                }
            )
            return VerificationOutcome(
                product_id=product_id,
                is_verified=False,
                message=(
                    f"⚠️ Possible match found (relevance: "
                    f"{best_partial['relevance_score']:.0%}). "
                    "Please verify details manually."
                ),
                details=details,
            )

        logger.info(f"Product ID not found: {product_id}")
        details.update(
            {
                "exact_match": False,
                "confidence_score": 0,
                "suggestions": [
                    "Verify the product ID is correct",
                    "Check if the product is registered with FDA Philippines",
                    "Try using the brand name or establishment name instead",
                ],
            }
        )
        return VerificationOutcome(
            product_id=product_id,
            is_verified=False,
            message=f"❌ Product ID '{product_id}' not found in FDA database",
            details=details,
        )

    def _exact_id_matched_field(
        self, match: dict[str, Any], normalized_product_id: str
    ) -> str | None:
        """Return the matched ID field name when the product ID matches exactly."""
        for field in (
            "registration_number",
            "license_number",
            "document_tracking_number",
        ):
            value = match.get(field)
            if value and normalize_string(value) == normalized_product_id:
                return field
        return None

    def _verified_message_for_product(self, product_info: dict[str, Any]) -> str:
        """Build a human-readable message for a verified product."""
        product_type = product_info.get("type", "unknown")
        if product_type == "drug_product":
            return (
                f"✅ Verified Drug Product: {product_info.get('brand_name', 'N/A')} "
                f"({product_info.get('generic_name', 'N/A')})"
            )
        if product_type == "food_product":
            return (
                f"✅ Verified Food Product: {product_info.get('product_name', 'N/A')} "
                f"by {product_info.get('company_name', 'N/A')}"
            )
        if isinstance(product_type, str) and product_type.endswith("_industry"):
            return (
                f"✅ Verified Establishment: "
                f"{product_info.get('name_of_establishment', 'N/A')}"
            )
        if product_type == "drug_application":
            return (
                f"✅ Verified Drug Application: {product_info.get('brand_name', 'N/A')} "
                f"({product_info.get('application_type', 'N/A')})"
            )
        return "✅ Product verified in FDA database"

    def _parse_drug_ingredients(self, ingredient_text: str) -> set[str]:
        """
        Parse drug ingredient text into a set of normalized ingredient names.

        Handles formats like:
        - "Phenylephrine HCI + Chlorphenamine Maleate + Paracetamol"
        - "Phenylephrine Hydrochloride and Chlorphenamine Maleate"

        Args:
            ingredient_text: Raw ingredient text

        Returns:
            Set of normalized ingredient names (lowercase, without salt forms)
        """
        import re

        if not ingredient_text:
            return set()

        # Normalize the text
        text = ingredient_text.lower().strip()

        # Split by common delimiters
        # Replace 'and', '+', '/', ',' with a pipe for splitting
        text = re.sub(r'\s+(?:and|\+|/|,)\s+', '|', text)

        # Split into individual ingredients
        raw_ingredients = [ing.strip() for ing in text.split('|')]

        # Normalize each ingredient
        normalized = set()
        for ingredient in raw_ingredients:
            if not ingredient:
                continue

            # Remove salt/acid forms (e.g., "hydrochloride", "maleate", "sulfate", "hcl", "hci")
            # These are common variations that should be treated as equivalent
            ingredient = re.sub(r'\s+(hydrochloride|hcl|hci|maleate|sulfate|citrate|phosphate|sodium|potassium)\b', '', ingredient)

            # Remove extra whitespace
            ingredient = ' '.join(ingredient.split())

            # Only keep meaningful ingredient names (at least 3 chars)
            if len(ingredient) >= 3:
                normalized.add(ingredient)

        return normalized

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

        # Drug Product Priority Boost: Strongly prioritize drug products when generic name matches
        # This is critical for cases like "Solmux" (Carbocisteine drug) vs food products
        if isinstance(model_instance, DrugProducts):
            generic_name = model_dict.get("generic_name")

            # Check if generic name appears in extracted product description or brand name
            if generic_name:
                generic_lower = generic_name.lower()

                # Check in product description
                if product_description and generic_lower in product_description.lower():
                    score += 0.25  # Strong boost for generic name in description
                    logger.debug(
                        f"Drug generic match boost: +0.25 ('{generic_name}' in description)"
                    )

                # Check in brand name (sometimes generic is mentioned)
                elif search_info.get("brand_name") and generic_lower in search_info["brand_name"].lower():
                    score += 0.20  # Boost for generic in brand
                    logger.debug(
                        f"Drug generic match boost: +0.20 ('{generic_name}' in brand)"
                    )

                # Multi-ingredient matching for drug products
                # Critical for drugs like "Neozep Forte" (Phenylephrine + Chlorphenamine + Paracetamol)
                # vs "Neozep" (Phenylephrine + Chlorphenamine)
                elif product_description:
                    # Parse ingredients from both search and database
                    # Ingredients are typically separated by '+' or 'and'
                    search_ingredients = self._parse_drug_ingredients(product_description)
                    db_ingredients = self._parse_drug_ingredients(generic_name)

                    if search_ingredients and db_ingredients:
                        # Count matching ingredients
                        matched_ingredients = search_ingredients & db_ingredients
                        total_search_ingredients = len(search_ingredients)
                        total_db_ingredients = len(db_ingredients)

                        if matched_ingredients:
                            # Calculate ingredient match ratio
                            match_ratio = len(matched_ingredients) / max(total_search_ingredients, total_db_ingredients)

                            # Perfect match: all ingredients match
                            if match_ratio == 1.0 and total_search_ingredients == total_db_ingredients:
                                score += 0.35  # Very strong boost for perfect ingredient match
                                logger.debug(
                                    f"Perfect drug ingredient match: +0.35 "
                                    f"({len(matched_ingredients)}/{total_search_ingredients} ingredients)"
                                )
                            # Complete match: DB has all searched ingredients (DB may have more)
                            elif matched_ingredients == search_ingredients:
                                score += 0.30  # Strong boost when all searched ingredients found
                                logger.debug(
                                    f"Complete drug ingredient match: +0.30 "
                                    f"({len(matched_ingredients)}/{total_search_ingredients} ingredients found)"
                                )
                            # Partial match: some ingredients match but not all
                            elif match_ratio >= 0.6:
                                boost = 0.15 + (match_ratio * 0.10)  # 0.15-0.25 based on ratio
                                score += boost
                                logger.debug(
                                    f"Partial drug ingredient match: +{boost:.2f} "
                                    f"({len(matched_ingredients)}/{total_search_ingredients} ingredients, ratio={match_ratio:.2f})"
                                )
                            # Weak match: less than 60% ingredients match
                            else:
                                # Apply small penalty for incomplete match (missing key ingredients)
                                penalty = -0.10
                                score = max(0, score + penalty)
                                logger.debug(
                                    f"Incomplete drug ingredient match: {penalty:.2f} penalty "
                                    f"({len(matched_ingredients)}/{total_search_ingredients} ingredients, ratio={match_ratio:.2f})"
                                )

                    # Fallback: word-level matching for generic name
                    else:
                        search_words = {
                            word.lower()
                            for word in product_description.strip().split()
                            if len(word) >= 3
                        }
                        generic_words = {
                            word.lower()
                            for word in str(generic_name).strip().split()
                            if len(word) >= 3
                        }

                        # If generic name contains terms from product description, boost score
                        common_words = search_words & generic_words
                        if len(common_words) >= 2:
                            # Significant overlap = strong context match
                            score += 0.15
                            logger.debug(
                                f"Drug context bonus: +0.15 ({len(common_words)} matching terms)"
                            )
                        elif len(common_words) == 1:
                            # Some overlap = moderate context match
                            score += 0.10
                            logger.debug("Drug context bonus: +0.10 (1 matching term)")

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
        normalized_product_id = normalize_string(product_id)

        id_fields = (
            ("registration_number", "registration_number"),
            ("license_number", "license_number"),
            ("document_tracking_number", "document_tracking_number"),
        )

        for field_name, matched_field in id_fields:
            value = model_dict.get(field_name)
            if value and normalize_string(value) == normalized_product_id:
                return 1.0, [matched_field]

        score = 0.0
        for field_name, matched_field in id_fields:
            value = model_dict.get(field_name)
            if value and normalized_product_id in normalize_string(value):
                score = 0.8
                matched_fields.append(matched_field)
                break

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
