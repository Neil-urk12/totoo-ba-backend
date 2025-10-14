from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from pydantic import BaseModel, Field
from typing import Optional
import os
import json
import io

# Import dependencies and repository
from app.api.deps import get_product_verification_service
from app.services.product_verification_service import ProductVerificationService


# Initialize router
router = APIRouter(prefix="/products")


# Define request and response models
class ProductVerificationRequest(BaseModel):
    """Request model for product verification"""
    product_id: str
    verification_code: Optional[str] = None


class ProductVerificationResponse(BaseModel):
    """Response model for product verification"""
    product_id: str
    is_verified: bool
    message: str
    details: Optional[dict] = None


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
        print("Warning: GEMINI_API_KEY not found in environment variables")
except ImportError:
    GEMINI_AVAILABLE = False
    client = None
    print("Warning: google-genai package not installed. Run: pip install google-genai")


# Pydantic models for structured Gemini output
class ExtractedFields(BaseModel):
    """Structured fields extracted from product image"""
    registration_number: Optional[str] = Field(
        None, 
        description="FDA Philippines registration number (e.g., BR-XXXX, DR-XXXXX, FR-XXXXX)"
    )
    brand_name: Optional[str] = Field(None, description="Brand name of the product")
    generic_name: Optional[str] = Field(None, description="Generic or common name")
    manufacturer: Optional[str] = Field(None, description="Manufacturer or distributor name")
    expiry_date: Optional[str] = Field(None, description="Expiration or best before date")
    batch_number: Optional[str] = Field(None, description="Batch or lot number")
    net_weight: Optional[str] = Field(None, description="Net weight or volume")


class VerificationResult(BaseModel):
    """AI verification result with structured output"""
    matched_product_index: Optional[int] = Field(
        None,
        description="Index of matched product from database (null if no match)"
    )
    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score 0-100"
    )
    extracted_fields: ExtractedFields
    reasoning: str = Field(
        ...,
        description="Explanation of matching decision and confidence level"
    )


class ImageVerificationResponse(BaseModel):
    """Response for image-based verification"""
    verification_status: str  # 'verified', 'uncertain', 'not_found'
    confidence: int
    matched_product: Optional[dict] = None
    extracted_fields: dict
    ai_reasoning: str
    alternative_matches: list = []


# Product verification endpoint (basic)
@router.post(
    "/verify",
    response_model=ProductVerificationResponse,
    summary="Verify Product by ID",
    description="Verifies if a product is legitimate using its ID and optional verification code"
)
async def verify_product(
    request: ProductVerificationRequest,
    verification_service: ProductVerificationService = Depends(get_product_verification_service)
):
    """
    Verify a product using its ID and optional verification code.
    This endpoint checks if a product is legitimate and verified in our system.
    
    The product_id can be:
    - FDA registration number (BR-XXXX, DR-XXXXX, FR-XXXXX, etc.)
    - License number for establishments
    - Document tracking number for applications
    """
    from app.services.fda_verification import normalize_string
    
    product_id = request.product_id.strip()
    verification_code = request.verification_code

    # Basic validation
    if not product_id or len(product_id) < 3:
        return ProductVerificationResponse(
            product_id=product_id,
            is_verified=False,
            message=f"Invalid product ID: {product_id}. Product ID must be at least 3 characters long.",
            details={"error_code": "INVALID_ID"}
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
            "verification_code_provided": verification_code is not None,
            "search_results_count": len(all_matches)
        }
        
        # Check for exact matches
        exact_matches = []
        partial_matches = []
        
        for match in all_matches:
            is_exact_match = False
            matched_field = None
            
            # Check for exact registration/license number match
            if match.get('registration_number') and normalize_string(match['registration_number']) == normalize_string(product_id):
                is_exact_match = True
                matched_field = 'registration_number'
            elif match.get('license_number') and normalize_string(match['license_number']) == normalize_string(product_id):
                is_exact_match = True
                matched_field = 'license_number'
            elif match.get('document_tracking_number') and normalize_string(match['document_tracking_number']) == normalize_string(product_id):
                is_exact_match = True
                matched_field = 'document_tracking_number'
            
            if is_exact_match:
                exact_matches.append({
                    'product': match,
                    'matched_field': matched_field,
                    'relevance_score': match.get('relevance_score', 1.0)
                })
            else:
                # Check for partial matches with high relevance
                relevance = match.get('relevance_score', 0.0)
                if relevance > 0.8:
                    partial_matches.append({
                        'product': match,
                        'relevance_score': relevance
                    })
        
        # Determine verification status
        if exact_matches:
            is_verified = True
            best_match = exact_matches[0]
            product_info = best_match['product']
            
            # Create detailed message based on product type
            product_type = product_info.get('type', 'unknown')
            if product_type == 'drug_product':
                message = f"✅ Verified Drug Product: {product_info.get('brand_name', 'N/A')} ({product_info.get('generic_name', 'N/A')})"
            elif product_type == 'food_product':
                message = f"✅ Verified Food Product: {product_info.get('product_name', 'N/A')} by {product_info.get('company_name', 'N/A')}"
            elif product_type.endswith('_industry'):
                message = f"✅ Verified Establishment: {product_info.get('name_of_establishment', 'N/A')}"
            elif product_type == 'drug_application':
                message = f"✅ Verified Drug Application: {product_info.get('brand_name', 'N/A')} ({product_info.get('application_type', 'N/A')})"
            else:
                message = f"✅ Product verified in FDA database"
            
            details.update({
                "verified_product": product_info,
                "matched_field": best_match['matched_field'],
                "exact_match": True,
                "confidence_score": 100
            })
            
        elif partial_matches:
            # High relevance but not exact match
            best_partial = partial_matches[0]
            message = f"⚠️ Possible match found (relevance: {best_partial['relevance_score']:.0%}). Please verify details manually."
            details.update({
                "possible_matches": partial_matches[:3],
                "exact_match": False,
                "confidence_score": int(best_partial['relevance_score'] * 100)
            })
            
        else:
            # No good matches found
            message = f"❌ Product ID '{product_id}' not found in FDA database"
            details.update({
                "exact_match": False,
                "confidence_score": 0,
                "suggestions": [
                    "Verify the product ID is correct",
                    "Check if the product is registered with FDA Philippines",
                    "Try using the brand name or establishment name instead"
                ]
            })
        
        # Additional verification with verification code if provided
        if verification_code and is_verified:
            # You can implement additional verification logic here
            # For now, we'll just note that a verification code was provided
            details["verification_code_validated"] = True
            message += f" (Verification code: {verification_code})"
        
        return ProductVerificationResponse(
            product_id=product_id,
            is_verified=is_verified,
            message=message,
            details=details
        )
        
    except Exception as e:
        # Handle database errors gracefully
        # Log the detailed error server-side for debugging
        print(f"Repository query failed: {str(e)}")  # In production, use proper logging
        return ProductVerificationResponse(
            product_id=product_id,
            is_verified=False,
            message="Error during verification: Database query failed",
            details={
                "error_code": "DATABASE_ERROR",
                "error_message": "Internal server error occurred during verification",
                "verification_method": "repository_database_lookup"
            }
        )


# Get verification info by ID
@router.get(
    "/verify/{product_id}",
    response_model=ProductVerificationResponse,
    summary="Get Product Verification Info",
    description="Retrieves verification status for a specific product ID"
)
async def get_product_verification(
    product_id: str,
    verification_service: ProductVerificationService = Depends(get_product_verification_service)
):
    """
    Get verification status for a specific product by its ID.
    This is an alternative way to check product verification status.
    """
    if not product_id or len(product_id) < 3:
        return ProductVerificationResponse(
            product_id=product_id,
            is_verified=False,
            message=f"Invalid product ID: {product_id}. Product ID must be at least 3 characters long.",
            details={"error_code": "INVALID_ID"}
        )

    try:
        # Use service layer for business logic
        search_results = await verification_service.verify_product_by_id(product_id)
        
        # Convert to legacy format for API compatibility
        all_matches = [result.to_dict() for result in search_results]
        
        if all_matches:
            # First match is already a dict from service layer
            best_match = all_matches[0]
            return ProductVerificationResponse(
                product_id=product_id,
                is_verified=True,
                message=f"Product {product_id} found in FDA database",
                details={
                    "verification_method": "repository_id_lookup",
                    "product_info": best_match,
                    "total_matches": len(all_matches)
                }
            )
        else:
            return ProductVerificationResponse(
                product_id=product_id,
                is_verified=False,
                message=f"Product {product_id} not found in FDA database",
                details={
                    "verification_method": "repository_id_lookup",
                    "total_matches": 0
                }
            )
    
    except Exception as e:
        print(f"Repository query failed: {str(e)}")
        return ProductVerificationResponse(
            product_id=product_id,
            is_verified=False,
            message="Error during verification lookup",
            details={
                "error_code": "DATABASE_ERROR",
                "verification_method": "repository_id_lookup"
            }
        )


# Image verification endpoint using Gemini 2.5 Flash
@router.post(
    "/verify-image",
    response_model=ImageVerificationResponse,
    summary="Verify Product from Image",
    description="Verifies a product by analyzing an uploaded image using Gemini 2.5 Flash AI"
)
async def verify_product_image(
    image: UploadFile = File(...),
    verification_service: ProductVerificationService = Depends(get_product_verification_service)
):
    """
    Verify a product by analyzing an uploaded image.
    Uses Gemini 2.5 Flash to extract text and match against FDA database.
    
    No OCR preprocessing required - Gemini handles vision understanding directly.
    """
    if not GEMINI_AVAILABLE or not client:
        raise HTTPException(
            status_code=500,
            detail="AI verification service temporarily unavailable. Please try again later."
        )

    # Validate image file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Please upload an image file."
        )

    # Check file size (max 5MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    image.file.seek(0, 2)  # Move to end of file to get size
    file_size = image.file.tell()
    image.file.seek(0)  # Move back to beginning of file
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 5MB."
        )

    try:
        # Read uploaded image as bytes
        image_bytes = await image.read()
        
        # Validate actual file content matches claimed content type
        if not validate_image_content(image_bytes, image.content_type):
            raise HTTPException(
                status_code=400,
                detail="File type mismatch. The uploaded file does not match the declared content type."
            )
        
        # Step 1: Extract structured data directly from image using Gemini
        extracted_data = await extract_fields_from_image(image_bytes, image.content_type)
        
        # Step 2: Query FDA database with extracted fields using service
        search_results = await verification_service.search_and_rank_products(extracted_data['extracted_fields'])
        fuzzy_results = [result.to_dict() for result in search_results]
        
        # Step 3: AI-assisted intelligent matching
        ai_verification = await ai_assisted_verification(
            extracted_fields=extracted_data['extracted_fields'],
            raw_text=extracted_data.get('raw_text', ''),
            fuzzy_results=fuzzy_results
        )
        
        # Determine final match
        matched_product = None
        if ai_verification['matched_product_index'] is not None:
            idx = ai_verification['matched_product_index']
            if idx < len(fuzzy_results):
                matched_product = fuzzy_results[idx]
        
        # Determine verification status
        verification_status = 'not_found'
        if ai_verification['confidence'] > 80:
            verification_status = 'verified'
        elif ai_verification['confidence'] > 50:
            verification_status = 'uncertain'
        
        return ImageVerificationResponse(
            verification_status=verification_status,
            confidence=ai_verification['confidence'],
            matched_product=matched_product,
            extracted_fields=ai_verification['extracted_fields'],
            ai_reasoning=ai_verification['reasoning'],
            alternative_matches=fuzzy_results[:3]
        )
        
    except Exception as e:
        # Log the detailed error server-side for debugging
        print(f"Error processing image: {str(e)}")  # In production, use proper logging
        raise HTTPException(
            status_code=500,
            detail="Image verification failed. Please try again."
        )
    finally:
        await image.close()


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
- generic_name: Generic/common product name or product description (e.g., "APPLE GREEN TEA")
- manufacturer: Manufacturer or distributor company name
- expiry_date: Expiration date or "Best Before" date
- batch_number: Batch, lot, or production number
- net_weight: Net weight, volume, or quantity

IMPORTANT EXTRACTION RULES:
1. For brand_name: Extract the PRIMARY brand identifier only (shortest recognizable brand)
   - Example: "C2" not "C2 COOL & CLEAN"
   - Example: "Nestle" not "Nestle Pure Life"
   
2. For generic_name: Extract the product type/flavor/description
   - Example: "APPLE GREEN TEA", "CHOCOLATE MILK", "PAIN RELIEVER"
   
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
            )
        )
        
        extracted_fields: ExtractedFields = response.parsed

        print(f"Extracted fields {extracted_fields}")
        
        # Also get raw text for additional context
        raw_text_prompt = "Extract all visible text from this image as plain text, preserving layout where possible."
        raw_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, raw_text_prompt],
            config=types.GenerateContentConfig(temperature=0.1)
        )
        
        return {
            'extracted_fields': extracted_fields.model_dump(),
            'raw_text': raw_response.text
        }
        
    except Exception as e:
        # Log the detailed error server-side for debugging
        print(f"Gemini extraction failed: {str(e)}")  # In production, use proper logging
        raise HTTPException(
            status_code=500,
            detail="Image processing failed. Please try again."
        )


async def ai_assisted_verification(extracted_fields: dict, raw_text: str, fuzzy_results: dict) -> dict:
    """
    Use Gemini 2.5 Flash for intelligent matching with improved fuzzy tolerance.
    """
    
    alternatives_text = "No potential matches found in database."
    if fuzzy_results:
        alternatives_text = json.dumps(fuzzy_results[:10], indent=2, ensure_ascii=False)
    
    # Debug logging for candidates being sent to Gemini
    print("FDA candidates given to Gemini:", fuzzy_results)
    
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
            )
        )
        
        result: VerificationResult = response.parsed
        
        return {
            "matched_product_index": result.matched_product_index,
            "confidence": result.confidence,
            "extracted_fields": result.extracted_fields.model_dump(),
            "reasoning": result.reasoning
        }
        
    except Exception as e:
        # Log the detailed error server-side for debugging
        print(f"AI verification failed: {str(e)}")  # In production, use proper logging
        return {
            "matched_product_index": None,
            "confidence": 0,
            "extracted_fields": extracted_fields,
            "reasoning": "AI verification temporarily unavailable. Please try again later."
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
        "image/webp": bytes([0x52, 0x49, 0x46, 0x46])  # First 4 bytes of WebP files
    }
    
    if mime_type not in magic_numbers:
        return False
    
    expected_header = magic_numbers[mime_type]
    return file_bytes.startswith(expected_header)




__all__ = ["router"]

