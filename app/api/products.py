import json
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Import dependencies and repository
from app.api.deps import get_product_verification_service
from app.services.product_verification_service import ProductVerificationService

# Initialize router
router = APIRouter(prefix="/products")


# Define request and response models
class ProductVerificationResponse(BaseModel):
    """Response model for product verification"""

    product_id: str
    is_verified: bool
    message: str
    details: dict | None = None


# Initialize Gemini client
try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        GEMINI_AVAILABLE = False
        client = None
except ImportError:
    GEMINI_AVAILABLE = False
    client = None


# Pydantic models for structured Gemini output
class ExtractedFields(BaseModel):
    """Structured fields extracted from product image"""

    registration_number: str | None = Field(
        None,
        description="FDA Philippines registration number (e.g., BR-XXXX, DR-XXXXX, FR-XXXXX)",
        alias="registration_number",
    )
    brand_name: str | None = Field(
        None, description="Brand name of the product", alias="brand_name"
    )
    product_description: str | None = Field(
        None,
        description="Product description, type, flavor, or generic name. For food: product type/flavor. For drugs: generic name",
        alias="product_description",
    )
    manufacturer: str | None = Field(
        None, description="Manufacturer or distributor name", alias="manufacturer"
    )
    expiry_date: str | None = Field(
        None, description="Expiration or best before date", alias="expiry_date"
    )
    batch_number: str | None = Field(
        None, description="Batch or lot number", alias="batch_number"
    )
    net_weight: str | None = Field(
        None, description="Net weight or volume", alias="net_weight"
    )

    class Config:
        populate_by_name = True  # Allow population by both field name and alias


class VerificationResult(BaseModel):
    """AI verification result with structured output"""

    matched_product_index: int | None = Field(
        None, description="Index of matched product from database (null if no match)"
    )
    confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")
    extracted_fields: ExtractedFields
    reasoning: str = Field(
        ..., description="Explanation of matching decision and confidence level"
    )


class ImageVerificationResponse(BaseModel):
    """Response for image-based verification"""

    verification_status: str  # 'verified', 'uncertain', 'not_found'
    confidence: int
    matched_product: dict | None = None
    extracted_fields: dict
    ai_reasoning: str
    alternative_matches: list = []


# Product verification endpoint (final)
@router.get(
    "/verify/{product_id}",
    response_model=ProductVerificationResponse,
    summary="Verify Product by ID",
    description="Verifies if a product is legitimate using its ID",
)
async def verify_product(
    product_id: str,
    verification_service: ProductVerificationService = Depends(
        get_product_verification_service
    ),
):
    """
    Verify a product using its ID.
    This endpoint checks if a product is legitimate and verified in our system.

    The product_id can be:
    - FDA registration number (BR-XXXX, DR-XXXXX, FR-XXXXX, etc.)
    - License number for establishments
    - Document tracking number for applications
    """
    from app.utils.helpers import normalize_string

    product_id = product_id.strip()

    # Basic validation
    if not product_id or len(product_id) < 3:
        return ProductVerificationResponse(
            product_id=product_id,
            is_verified=False,
            message=f"Invalid product ID: {product_id}. Product ID must be at least 3 characters long.",
            details={"error_code": "INVALID_ID"},
        )

    try:
        # Use service layer for business logic
        search_results = await verification_service.verify_product_by_id(product_id)

        # Convert to legacy format for API compatibility
        all_matches = [result.to_dict() for result in search_results]

        is_verified = False
        message = f"Product ID '{product_id}' not found in FDA database"
        details = {
            "verification_method": "repository_database_lookup",
            "search_results_count": len(all_matches),
        }

        # Check for exact matches
        exact_matches = []
        partial_matches = []

        for match in all_matches:
            is_exact_match = False
            matched_field = None

            # Check for exact registration/license number match
            if match.get("registration_number") and normalize_string(
                match["registration_number"]
            ) == normalize_string(product_id):
                is_exact_match = True
                matched_field = "registration_number"
            elif match.get("license_number") and normalize_string(
                match["license_number"]
            ) == normalize_string(product_id):
                is_exact_match = True
                matched_field = "license_number"
            elif match.get("document_tracking_number") and normalize_string(
                match["document_tracking_number"]
            ) == normalize_string(product_id):
                is_exact_match = True
                matched_field = "document_tracking_number"

            if is_exact_match:
                exact_matches.append(
                    {
                        "product": match,
                        "matched_field": matched_field,
                        "relevance_score": match.get("relevance_score", 1.0),
                    }
                )
            else:
                # Check for partial matches with high relevance
                relevance = match.get("relevance_score", 0.0)
                if relevance > 0.8:
                    partial_matches.append(
                        {"product": match, "relevance_score": relevance}
                    )

        # Determine verification status
        if exact_matches:
            is_verified = True
            best_match = exact_matches[0]
            product_info = best_match["product"]

            # Create detailed message based on product type
            product_type = product_info.get("type", "unknown")
            if product_type == "drug_product":
                message = f"✅ Verified Drug Product: {product_info.get('brand_name', 'N/A')} ({product_info.get('generic_name', 'N/A')})"
            elif product_type == "food_product":
                message = f"✅ Verified Food Product: {product_info.get('product_name', 'N/A')} by {product_info.get('company_name', 'N/A')}"
            elif product_type.endswith("_industry"):
                message = f"✅ Verified Establishment: {product_info.get('name_of_establishment', 'N/A')}"
            elif product_type == "drug_application":
                message = f"✅ Verified Drug Application: {product_info.get('brand_name', 'N/A')} ({product_info.get('application_type', 'N/A')})"
            else:
                message = "✅ Product verified in FDA database"

            details.update(
                {
                    "verified_product": product_info,
                    "matched_field": best_match["matched_field"],
                    "exact_match": True,
                    "confidence_score": 100,
                }
            )

        elif partial_matches:
            # High relevance but not exact match
            best_partial = partial_matches[0]
            message = f"⚠️ Possible match found (relevance: {best_partial['relevance_score']:.0%}). Please verify details manually."
            details.update(
                {
                    "possible_matches": partial_matches[:3],
                    "exact_match": False,
                    "confidence_score": int(best_partial["relevance_score"] * 100),
                }
            )

        else:
            # No good matches found
            message = f"❌ Product ID '{product_id}' not found in FDA database"
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

        return ProductVerificationResponse(
            product_id=product_id,
            is_verified=is_verified,
            message=message,
            details=details,
        )

    except Exception:
        # Handle database errors gracefully
        # Log the detailed error server-side for debugging
        return ProductVerificationResponse(
            product_id=product_id,
            is_verified=False,
            message="Error during verification: Database query failed",
            details={
                "error_code": "DATABASE_ERROR",
                "error_message": "Internal server error occurred during verification",
                "verification_method": "repository_database_lookup",
            },
        )


# Image verification endpoint using Gemini 2.5 Flash
@router.post(
    "/verify-image",
    response_model=ImageVerificationResponse,
    summary="Verify Product from Image",
    description="Verifies a product by analyzing an uploaded image using Gemini 2.5 Flash AI",
)
async def verify_product_image(
    image: UploadFile = File(...),
    verification_service: ProductVerificationService = Depends(
        get_product_verification_service
    ),
):
    """
    Verify a product by analyzing an uploaded image.
    Uses Gemini 2.5 Flash to extract text and match against FDA database.

    No OCR preprocessing required - Gemini handles vision understanding directly.
    """
    if not GEMINI_AVAILABLE or not client:
        raise HTTPException(
            status_code=500,
            detail="AI verification service temporarily unavailable. Please try again later.",
        )

    # Validate image file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Please upload an image file.",
        )

    # Check file size (max 5MB)
    max_file_size = 5 * 1024 * 1024  # 5MB
    image.file.seek(0, 2)  # Move to end of file to get size
    file_size = image.file.tell()
    image.file.seek(0)  # Move back to beginning of file

    if file_size > max_file_size:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum size is 5MB."
        )

    try:
        # Read uploaded image as bytes
        image_bytes = await image.read()

        # Validate actual file content matches claimed content type
        if not validate_image_content(image_bytes, image.content_type):
            raise HTTPException(
                status_code=400,
                detail="File type mismatch. The uploaded file does not match the declared content type.",
            )

        # Step 1: Extract structured data directly from image using Gemini
        extracted_data = await extract_fields_from_image(
            image_bytes, image.content_type
        )

        # Step 2: Convert extracted fields to search dict and query FDA database
        search_dict = convert_extracted_fields_to_search_dict(
            extracted_data["extracted_fields"]
        )

        search_results = await verification_service.search_and_rank_products(
            search_dict
        )

        fuzzy_results = [result.to_dict() for result in search_results]

        # Step 3: AI-assisted intelligent matching
        ai_verification = await ai_assisted_verification(
            extracted_fields=search_dict,
            raw_text=extracted_data.get("raw_text", ""),
            fuzzy_results=fuzzy_results,
        )

        # Determine final match
        matched_product = None
        if ai_verification["matched_product_index"] is not None:
            idx = ai_verification["matched_product_index"]
            if idx < len(fuzzy_results):
                matched_product = fuzzy_results[idx]

        # Determine verification status
        verification_status = "not_found"
        if ai_verification["confidence"] > 80:
            verification_status = "verified"
        elif ai_verification["confidence"] > 50:
            verification_status = "uncertain"

        return ImageVerificationResponse(
            verification_status=verification_status,
            confidence=ai_verification["confidence"],
            matched_product=matched_product,
            extracted_fields=ai_verification["extracted_fields"],
            ai_reasoning=ai_verification["reasoning"],
            alternative_matches=fuzzy_results[:3],
        )

    except Exception as e:
        # Log the detailed error server-side for debugging
        raise HTTPException(
            status_code=500, detail="Image verification failed. Please try again."
        ) from e
    finally:
        await image.close()


def convert_extracted_fields_to_search_dict(extracted_fields: dict) -> dict:
    """
    Convert extracted fields dict to a search dictionary with proper field mapping.
    Maps product_description to the appropriate field names for different product types.

    Args:
        extracted_fields: Dictionary with extracted field data (from ExtractedFields.model_dump())

    Returns:
        Search dictionary with proper field mappings for database queries
    """
    search_dict = {}

    # Direct field mappings - handle dict access
    if extracted_fields.get("registration_number"):
        search_dict["registration_number"] = extracted_fields["registration_number"]
    if extracted_fields.get("brand_name"):
        search_dict["brand_name"] = extracted_fields["brand_name"]
    if extracted_fields.get("manufacturer"):
        search_dict["manufacturer"] = extracted_fields["manufacturer"]
        search_dict["company_name"] = extracted_fields[
            "manufacturer"
        ]  # Also map to company_name for food products
    if extracted_fields.get("expiry_date"):
        search_dict["expiry_date"] = extracted_fields["expiry_date"]
    if extracted_fields.get("batch_number"):
        search_dict["batch_number"] = extracted_fields["batch_number"]
    if extracted_fields.get("net_weight"):
        search_dict["net_weight"] = extracted_fields["net_weight"]

    # Map product_description to both generic_name and product_name for compatibility
    if extracted_fields.get("product_description"):
        search_dict["product_description"] = extracted_fields["product_description"]
        search_dict["generic_name"] = extracted_fields[
            "product_description"
        ]  # For drug products
        search_dict["product_name"] = extracted_fields[
            "product_description"
        ]  # For food products

    return search_dict


async def extract_fields_from_image(image_bytes: bytes, mime_type: str) -> dict:
    """
    Extract structured fields directly from product image using Gemini 2.5 Flash.
    No OCR preprocessing - Gemini handles vision understanding natively.
    """

    prompt = """You are an expert FDA Philippines product verification assistant.

Analyze this product image and extract ALL visible text and product information.

Extract these specific fields:
- registration_number: FDA Philippines registration number (patterns: BR-XXXX, DR-XXXXX, FR-XXXXX, etc.)
- brand_name: Product brand name (extract ONLY the core brand, e.g., "C2" not "C2 COOL & CLEAN")
- product_description: Product description, type, flavor, or generic name (e.g., "APPLE GREEN TEA", "CHOCOLATE MILK", "PAIN RELIEVER")
- manufacturer: Manufacturer or distributor company name
- expiry_date: Expiration date or "Best Before" date
- batch_number: Batch, lot, or production number
- net_weight: Net weight, volume, or quantity

IMPORTANT EXTRACTION RULES:
1. For brand_name: Extract the PRIMARY brand identifier only (shortest recognizable brand)
   - Example: "C2" not "C2 COOL & CLEAN"
   - Example: "Nestle" not "Nestle Pure Life"

2. For product_description: Extract the product type/flavor/description that describes what the product is
   - For beverages: "APPLE GREEN TEA", "ORANGE JUICE", "COLA"
   - For medicines: "PAIN RELIEVER", "COUGH SYRUP", "VITAMINS"
   - For food: "CHOCOLATE MILK", "INSTANT NOODLES", "COOKIES"

3. Extract exact text as it appears, but prioritize the SHORTEST meaningful brand name
4. If a field is not visible or unclear, set it to null
5. Pay special attention to FDA registration numbers (usually starts with letters like BR, DR, FR, etc.)

Also provide the complete raw text visible in the image for fallback matching."""

    try:
        # Create image part from bytes
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        # Generate structured content
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedFields,
                temperature=0.1,
            ),
        )

        extracted_fields: ExtractedFields = response.parsed

        # Also get raw text for additional context
        raw_text_prompt = "Extract all visible text from this image as plain text, preserving layout where possible."
        raw_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, raw_text_prompt],
            config=types.GenerateContentConfig(temperature=0.1),
        )

        return {
            "extracted_fields": extracted_fields.model_dump(),
            "raw_text": raw_response.text,
        }

    except Exception as e:
        # Log the detailed error server-side for debugging
        raise HTTPException(
            status_code=500, detail="Image processing failed. Please try again."
        ) from e


async def ai_assisted_verification(
    extracted_fields: dict, raw_text: str, fuzzy_results: dict
) -> dict:
    """
    Use Gemini 2.5 Flash for intelligent matching with improved fuzzy tolerance.
    """

    alternatives_text = "No potential matches found in database."
    if fuzzy_results:
        alternatives_text = json.dumps(fuzzy_results[:10], indent=2, ensure_ascii=False)

    prompt = f"""You are an FDA Philippines product verification expert with expertise in fuzzy matching.

Your task: Determine if the extracted product information matches any database records.

EXTRACTED PRODUCT DATA:
{json.dumps(extracted_fields, indent=2, ensure_ascii=False)}

RAW TEXT FROM IMAGE:
{raw_text[:1500]}

POTENTIAL DATABASE MATCHES (sorted by similarity):
{alternatives_text}

VERIFICATION TASKS:
1. Compare extracted fields against each database record using FUZZY MATCHING
2. Identify the correct match (if any) by index (0-based) or return null
3. Calculate confidence score (0-100):
   - 90-100: Perfect match on registration number + brand
   - 70-89: Strong match on brand + product type with minor variations
   - 50-69: Partial match with acceptable variations
   - 30-49: Weak match, significant uncertainty
   - 0-29: Poor match or no match

IMPORTANT FUZZY MATCHING RULES:
- Brand names: "C2" matches "C2 COOL & CLEAN" (core brand is the same)
- Product names: "APPLE GREEN TEA" matches "APPLE FLAVORED GREEN TEA-SOLO" (similar product)
- Allow for:
  * Additional descriptive words (COOL, CLEAN, FLAVORED, SOLO, etc.)
  * Different word order
  * Minor spelling variations
  * Special characters and formatting differences

- FDA registration numbers are the STRONGEST identifier (if present)
- If no registration number: rely on brand + product name combination
- Consider OCR errors (e.g., "0" vs "O", "1" vs "l")
- If multiple matches: choose the one with best overall field alignment

DECISION PRIORITY:
1. Registration number match = high confidence (80-100)
2. Brand + product type match = medium-high confidence (60-85)
3. Brand only match = medium confidence (40-65)
4. No clear match = low confidence (0-35)

Provide structured output with your decision and clear reasoning."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VerificationResult,
                temperature=0.3,  # Slightly higher for flexible matching
            ),
        )

        result: VerificationResult = response.parsed

        return {
            "matched_product_index": result.matched_product_index,
            "confidence": result.confidence,
            "extracted_fields": result.extracted_fields.model_dump(),
            "reasoning": result.reasoning,
        }

    except Exception:
        # Log the detailed error server-side for debugging
        return {
            "matched_product_index": None,
            "confidence": 0,
            "extracted_fields": extracted_fields,
            "reasoning": "AI verification temporarily unavailable. Please try again later.",
        }


def validate_image_content(file_bytes: bytes, mime_type: str) -> bool:
    """
    Validate that the file content matches the expected image type using magic numbers.
    """
    # Define magic numbers for common image formats
    magic_numbers = {
        "image/jpeg": bytes([0xFF, 0xD8, 0xFF]),
        "image/png": bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
        "image/gif": bytes([0x47, 0x49, 0x46, 0x38]),  # GIF87a or GIF89a
        "image/webp": bytes([0x52, 0x49, 0x46, 0x46]),  # First 4 bytes of WebP files
    }

    if mime_type not in magic_numbers:
        return False

    expected_header = magic_numbers[mime_type]
    return file_bytes.startswith(expected_header)


__all__ = ["router"]
