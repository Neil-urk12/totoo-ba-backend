import asyncio
import re
from sqlalchemy import or_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import async_session
from app.models import (
    DrugProducts,
    FoodProducts,
    DrugIndustry,
    FoodIndustry,
    MedicalDeviceIndustry,
    CosmeticIndustry,
    DrugsNewApplications,
)
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional


def safe_date_format(date_obj) -> Optional[str]:
    """Safely format date object to ISO string, handling both datetime and string types."""
    if date_obj is None:
        return None
    if hasattr(date_obj, 'isoformat'):
        return date_obj.isoformat()
    return str(date_obj)


def normalize_string(text: str) -> str:
    """Normalize string for comparison: uppercase and remove spaces."""
    if not text:
        return ""
    return text.upper().replace(" ", "")


def normalize_text(text: str) -> str:
    """Normalize text for better matching - remove special chars, extra spaces."""
    if not text:
        return ""
    # Convert to lowercase, remove special characters except alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text.lower())
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings using SequenceMatcher."""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def token_based_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity based on token overlap.
    Better for partial matches like "C2" matching "C2 COOL & CLEAN".
    """
    if not str1 or not str2:
        return 0.0

    # Normalize and tokenize
    tokens1 = set(normalize_text(str1).split())
    tokens2 = set(normalize_text(str2).split())

    if not tokens1 or not tokens2:
        return 0.0

    # Jaccard similarity
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    jaccard = len(intersection) / len(union) if union else 0.0

    # Also check if shorter string is fully contained in longer string
    shorter = str1 if len(str1) <= len(str2) else str2
    longer = str2 if len(str1) <= len(str2) else str1
    containment = 1.0 if normalize_text(shorter) in normalize_text(longer) else 0.0

    # Return maximum of jaccard and containment for better matching
    return max(jaccard, containment)


def hybrid_similarity(str1: str, str2: str) -> float:
    """
    Hybrid approach combining sequence matching and token-based matching.
    Returns the higher score between the two methods.
    """
    if not str1 or not str2:
        return 0.0

    sequence_score = calculate_similarity(str1, str2)
    token_score = token_based_similarity(str1, str2)

    # Return weighted average favoring the higher score
    return max(
        sequence_score, token_score * 0.9
    )  # Slight preference for sequence matching


async def search_drug_products(
    session: AsyncSession, extracted_fields: dict
) -> List[Dict]:
    """Search drug products with improved matching."""
    results = []

    conditions = []

    registration_number = (extracted_fields.get("registration_number") or "").strip()
    brand_name = extracted_fields.get("brand_name", "").strip()
    generic_name = extracted_fields.get("generic_name", "").strip()
    manufacturer = extracted_fields.get("manufacturer", "").strip()

    if registration_number:
        conditions.append(
            DrugProducts.registration_number.ilike(f"%{registration_number}%")
        )

    if brand_name:
        brand_tokens = normalize_text(brand_name).split()
        brand_conditions = []
        for token in brand_tokens:
            if len(token) >= 2:
                brand_conditions.append(DrugProducts.brand_name.ilike(f"%{token}%"))
        if brand_conditions:
            conditions.append(or_(*brand_conditions))

    if generic_name:
        generic_tokens = normalize_text(generic_name).split()
        generic_conditions = []
        for token in generic_tokens:
            if len(token) >= 3:
                generic_conditions.append(DrugProducts.generic_name.ilike(f"%{token}%"))
        if generic_conditions:
            conditions.append(or_(*generic_conditions))

    if manufacturer:
        conditions.append(
            or_(
                DrugProducts.manufacturer.ilike(f"%{manufacturer}%"),
                DrugProducts.trader.ilike(f"%{manufacturer}%"),
                DrugProducts.importer.ilike(f"%{manufacturer}%"),
            )
        )

    if not conditions:
        query = select(DrugProducts).limit(50)
        result = await session.execute(query)
        products = result.scalars().all()
    else:
        query = select(DrugProducts).where(or_(*conditions)).limit(25)
        result = await session.execute(query)
        products = result.scalars().all()

        # If the filtered query returns no results, perform a full table scan as fallback
        if not products:
            fallback_query = select(DrugProducts).limit(50)
            fallback_result = await session.execute(fallback_query)
            products = fallback_result.scalars().all()

    for product in products:
        scores = []

        if registration_number and product.registration_number:
            scores.append(
                calculate_similarity(registration_number, product.registration_number)
                * 3
            )

        if brand_name and product.brand_name:
            scores.append(hybrid_similarity(brand_name, product.brand_name) * 2)

        if generic_name and product.generic_name:
            scores.append(hybrid_similarity(generic_name, product.generic_name) * 1.5)

        if manufacturer:
            manufacturer_fields = [
                product.manufacturer,
                product.trader,
                product.importer,
            ]
            manufacturer_scores = [
                hybrid_similarity(manufacturer, field)
                for field in manufacturer_fields
                if field
            ]
            if manufacturer_scores:
                scores.append(max(manufacturer_scores))

        if scores:
            total_weight = sum([3, 2, 1.5, 1][: len(scores)])
            avg_score = sum(scores) / total_weight
        else:
            avg_score = 0.0

        if avg_score > 0.1:  # Lower threshold when using fallback results
            results.append(
                {
                    "product": {
                        "type": "drug_product",
                        "registration_number": product.registration_number,
                        "brand_name": product.brand_name,
                        "generic_name": product.generic_name,
                        "manufacturer": product.manufacturer,
                        "dosage_form": product.dosage_form,
                        "dosage_strength": product.dosage_strength,
                        "issuance_date": safe_date_format(product.issuance_date),
                        "expiry_date": safe_date_format(product.expiry_date),
                    },
                    "similarity_score": round(avg_score, 3),
                }
            )

    return results


async def search_food_products(
    session: AsyncSession, extracted_fields: dict
) -> List[Dict]:
    """Search food products table with improved fuzzy matching."""
    results = []

    conditions = []

    registration_number = (extracted_fields.get("registration_number") or "").strip()
    brand_name = extracted_fields.get("brand_name", "").strip()
    generic_name = extracted_fields.get("generic_name", "").strip()
    product_name = (
        extracted_fields.get("product_name", "").strip() or generic_name
    )  # Fallback to generic_name
    company_name = extracted_fields.get("company_name", "").strip()
    manufacturer = extracted_fields.get("manufacturer", "").strip()

    # If we have registration number, prioritize exact search
    if registration_number:
        conditions.append(
            FoodProducts.registration_number.ilike(f"%{registration_number}%")
        )

    # Build more flexible brand name search
    if brand_name:
        # Extract tokens for partial matching
        brand_tokens = normalize_text(brand_name).split()
        brand_conditions = []
        for token in brand_tokens:
            if len(token) >= 2:  # Only use tokens with 2+ chars
                brand_conditions.append(FoodProducts.brand_name.ilike(f"%{token}%"))
        if brand_conditions:
            conditions.append(or_(*brand_conditions))

    # Product name / generic name search with token matching
    if product_name:
        product_tokens = normalize_text(product_name).split()
        product_conditions = []
        for token in product_tokens:
            if len(token) >= 3:  # Use meaningful tokens
                product_conditions.extend(
                    [
                        FoodProducts.product_name.ilike(f"%{token}%"),
                        FoodProducts.brand_name.ilike(f"%{token}%"),
                    ]
                )
        if product_conditions:
            conditions.append(or_(*product_conditions))

    # Company/manufacturer search
    if company_name or manufacturer:
        company = company_name or manufacturer
        conditions.append(FoodProducts.company_name.ilike(f"%{company}%"))

    # If no conditions, do a broader search
    if not conditions:
        # Fallback: search all food products (limit results)
        query = select(FoodProducts).limit(50)
        result = await session.execute(query)
        products = result.scalars().all()
    else:
        query = select(FoodProducts).where(or_(*conditions)).limit(25)
        result = await session.execute(query)
        products = result.scalars().all()

        # If the filtered query returns no results, perform a full table scan as fallback
        if not products:
            fallback_query = select(FoodProducts).limit(50)
            fallback_result = await session.execute(fallback_query)
            products = fallback_result.scalars().all()

    for product in products:
        scores = []

        # Registration number (highest priority - exact match)
        if registration_number and product.registration_number:
            reg_score = calculate_similarity(
                registration_number, product.registration_number
            )
            scores.append(reg_score * 3)  # 3x weight for registration number

        # Brand name (use hybrid matching)
        if brand_name and product.brand_name:
            brand_score = hybrid_similarity(brand_name, product.brand_name)
            scores.append(brand_score * 2)  # 2x weight for brand

        # Product name vs generic_name
        if product_name and product.product_name:
            product_score = hybrid_similarity(product_name, product.product_name)
            scores.append(product_score * 1.5)  # 1.5x weight

        # Company name
        if (company_name or manufacturer) and product.company_name:
            company = company_name or manufacturer
            company_score = hybrid_similarity(company, product.company_name)
            scores.append(company_score)

        # Calculate weighted average
        if scores:
            # Normalize by total weights
            total_weight = sum([3, 2, 1.5, 1][: len(scores)])
            avg_score = sum(scores) / total_weight
        else:
            avg_score = 0.0

        # Lower threshold to 0.1 (10%) for more results
        if avg_score > 0.1:  # Lower threshold when using fallback results
            results.append(
                {
                    "product": {
                        "type": "food_product",
                        "registration_number": product.registration_number,
                        "product_name": product.product_name,
                        "brand_name": product.brand_name,
                        "company_name": product.company_name,
                        "type_of_product": product.type_of_product,
                        "issuance_date": safe_date_format(product.issuance_date),
                        "expiry_date": safe_date_format(product.expiry_date),
                    },
                    "similarity_score": round(avg_score, 3),
                    "match_details": {
                        "brand_match": round(
                            hybrid_similarity(brand_name, product.brand_name), 2
                        )
                        if brand_name and product.brand_name
                        else 0,
                        "product_match": round(
                            hybrid_similarity(product_name, product.product_name), 2
                        )
                        if product_name and product.product_name
                        else 0,
                    },
                }
            )

    return results


async def search_establishments(
    session: AsyncSession, extracted_fields: dict
) -> List[Dict]:
    """Search establishment tables (drug, food, medical device, cosmetic industries) for matches."""
    results = []

    establishment_name = (extracted_fields.get("establishment_name") or "").strip()
    license_number = extracted_fields.get("license_number", "").strip()
    owner = extracted_fields.get("owner", "").strip()
    address = extracted_fields.get("address", "").strip()

    # Define establishment models to search
    establishment_models = [
        (DrugIndustry, "drug_industry"),
        (FoodIndustry, "food_industry"),
        (MedicalDeviceIndustry, "medical_device_industry"),
        (CosmeticIndustry, "cosmetic_industry"),
    ]

    for model_class, industry_type in establishment_models:
        conditions = []

        if license_number:
            conditions.append(model_class.license_number.ilike(f"%{license_number}%"))

        if establishment_name:
            conditions.append(
                model_class.name_of_establishment.ilike(f"%{establishment_name}%")
            )

        if owner:
            conditions.append(model_class.owner.ilike(f"%{owner}%"))

        if address:
            conditions.append(model_class.address.ilike(f"%{address}%"))

        if conditions:
            query = select(model_class).where(or_(*conditions)).limit(25)
            result = await session.execute(query)
            establishments = result.scalars().all()

            for establishment in establishments:
                scores = []

                if license_number and establishment.license_number:
                    scores.append(
                        calculate_similarity(
                            license_number, establishment.license_number
                        )
                    )
                if establishment_name and establishment.name_of_establishment:
                    scores.append(
                        calculate_similarity(
                            establishment_name, establishment.name_of_establishment
                        )
                    )
                if owner and establishment.owner:
                    scores.append(calculate_similarity(owner, establishment.owner))
                if address and establishment.address:
                    scores.append(calculate_similarity(address, establishment.address))

                avg_score = sum(scores) / len(scores) if scores else 0.0

                if avg_score > 0.1:  # Lower threshold for more results
                    results.append(
                        {
                            "product": {
                                "type": industry_type,
                                "license_number": establishment.license_number,
                                "name_of_establishment": establishment.name_of_establishment,
                                "owner": establishment.owner,
                                "address": establishment.address,
                                "region": establishment.region,
                                "activity": establishment.activity,
                                "expiry_date": safe_date_format(establishment.expiry_date),
                            },
                            "similarity_score": round(avg_score, 2),
                        }
                    )

    return results


async def search_drug_applications(
    session: AsyncSession, extracted_fields: dict
) -> List[Dict]:
    """Search drug new applications table for matches."""
    results = []

    conditions = []

    tracking_number = (extracted_fields.get("document_tracking_number") or "").strip()
    brand_name = (extracted_fields.get("brand_name") or "").strip()
    generic_name = (extracted_fields.get("generic_name") or "").strip()
    applicant_company = (extracted_fields.get("applicant_company") or "").strip()

    if tracking_number:
        conditions.append(
            DrugsNewApplications.document_tracking_number.ilike(f"%{tracking_number}%")
        )

    if brand_name:
        conditions.append(DrugsNewApplications.brand_name.ilike(f"%{brand_name}%"))

    if generic_name:
        conditions.append(DrugsNewApplications.generic_name.ilike(f"%{generic_name}%"))

    if applicant_company:
        conditions.append(
            DrugsNewApplications.applicant_company.ilike(f"%{applicant_company}%")
        )

    if conditions:
        query = select(DrugsNewApplications).where(or_(*conditions)).limit(25)
        result = await session.execute(query)
        applications = result.scalars().all()

        for application in applications:
            scores = []

            if tracking_number and application.document_tracking_number:
                scores.append(
                    calculate_similarity(
                        tracking_number, application.document_tracking_number
                    )
                )
            if brand_name and application.brand_name:
                scores.append(calculate_similarity(brand_name, application.brand_name))
            if generic_name and application.generic_name:
                scores.append(
                    calculate_similarity(generic_name, application.generic_name)
                )
            if applicant_company and application.applicant_company:
                scores.append(
                    calculate_similarity(
                        applicant_company, application.applicant_company
                    )
                )

            avg_score = sum(scores) / len(scores) if scores else 0.0

            if avg_score > 0.1:  # Lower threshold for more results
                results.append(
                    {
                        "product": {
                            "type": "drug_application",
                            "document_tracking_number": application.document_tracking_number,
                            "brand_name": application.brand_name,
                            "generic_name": application.generic_name,
                            "applicant_company": application.applicant_company,
                            "dosage_form": application.dosage_form,
                            "dosage_strength": application.dosage_strength,
                            "application_type": application.application_type,
                        },
                        "similarity_score": round(avg_score, 2),
                    }
                )

    return results


async def search_fda_database(extracted_fields: dict) -> dict:
    """
    Search your Supabase FDA database using extracted fields.
    Returns fuzzy-matched alternatives.

    Args:
        extracted_fields: Dictionary containing extracted fields like:
            - registration_number
            - brand_name
            - generic_name
            - product_name
            - manufacturer
            - company_name
            - establishment_name
            - license_number
            - owner
            - address
            - document_tracking_number
            - applicant_company

    Returns:
        Dictionary with 'alternatives' list containing matched products with similarity scores
    """
    all_results = []

    try:
        # Use a single session for all operations to avoid concurrent issues
        async with async_session() as session:
            # Search each table sequentially to avoid concurrent session operations
            try:
                drug_results = await search_drug_products(session, extracted_fields)
                all_results.extend(drug_results)
            except Exception as e:
                print(f"Search error: {e}")

            try:
                food_results = await search_food_products(session, extracted_fields)
                all_results.extend(food_results)
            except Exception as e:
                print(f"Search error: {e}")

            try:
                establishment_results = await search_establishments(session, extracted_fields)
                all_results.extend(establishment_results)
            except Exception as e:
                print(f"Search error: {e}")

            try:
                application_results = await search_drug_applications(session, extracted_fields)
                all_results.extend(application_results)
            except Exception as e:
                print(f"Search error: {e}")

            # Sort by similarity score (highest first)
            all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Limit to top 20 results
            all_results = all_results[:20]

    except Exception as e:
        print(f"Database search error: {e}")
        return {"alternatives": [], "error": str(e)}

    return {"alternatives": all_results, "total_found": len(all_results)}