"""Product verification API endpoints.

Provides REST API endpoints for verifying products using:
- Product ID lookup (registration numbers, license numbers)
- Image-based verification using Groq AI vision models
- Hybrid Vision verification using Groq
"""
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

# Import dependencies and repository
from app.api.deps import get_product_verification_service
from app.services.product_verification_service import ProductVerificationService

# Initialize router
router = APIRouter(prefix="/products")


# Define request and response models
class ProductVerificationResponse(BaseModel):
    """Response model for product verification.

    Attributes:
        product_id: The product identifier that was verified.
        is_verified: Whether the product was successfully verified.
        message: Human-readable verification message.
        details: Additional verification details and metadata.
    """

    product_id: str
    is_verified: bool
    message: str
    details: dict | None = None




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
    """Verify a product using its ID.

    This endpoint checks if a product is legitimate and verified in the FDA database.

    Args:
        product_id: Product identifier (registration number, license number, or tracking number).
        verification_service: Injected product verification service.

    Returns:
        ProductVerificationResponse: Verification result with product details.

    The product_id can be:
    - FDA registration number (BR-XXXX, DR-XXXXX, FR-XXXXX, etc.)
    - License number for establishments
    - Document tracking number for applications
    """
    product_id = product_id.strip()
    logger.info(f"Product verification request for ID: {product_id[:20]}...")
    outcome = await verification_service.verify_product_by_id(product_id)
    return ProductVerificationResponse(
        product_id=outcome.product_id,
        is_verified=outcome.is_verified,
        message=outcome.message,
        details=outcome.details,
    )












# Hybrid vision verification response model
class HybridVerificationResponse(BaseModel):
    """Response for hybrid vision-based verification.

    Attributes:
        verification_status: Status of verification ('verified', 'uncertain', 'not_found').
        confidence: Confidence score from 0-100.
        matched_product: Matched product details if found.
        extracted_fields: Fields extracted from the image.
        ai_reasoning: AI explanation of the verification decision.
        alternative_matches: List of alternative potential matches.
        processing_metadata: Performance metrics and processing details.
    """

    verification_status: str  # 'verified', 'uncertain', 'not_found'
    confidence: int
    matched_product: dict | None = None
    extracted_fields: dict
    ai_reasoning: str
    alternative_matches: list = []
    processing_metadata: dict  # Performance metrics


@router.post(
    "/verify-image",
    response_model=HybridVerificationResponse,
    summary="Verify Product from Image (Hybrid Vision)",
    description="Verifies a product using hybrid approach: Groq Vision + Fast Matching",
)
@router.post(
    "/new-verify-image",
    response_model=HybridVerificationResponse,
    summary="Verify Product from Image (Hybrid Vision) [Alias]",
    description="Backward-compatible alias for /verify-image",
    include_in_schema=False,
)
async def verify_product_image(
    image: UploadFile = File(...),
    verification_service: ProductVerificationService = Depends(
        get_product_verification_service
    ),
):
    """Verify a product by analyzing an uploaded image using hybrid Vision approach.

    Args:
        image: Uploaded image file (max 5MB, JPEG/PNG/GIF/WebP).
        verification_service: Injected product verification service.

    Returns:
        HybridVerificationResponse: Verification result with processing metadata.

    Raises:
        HTTPException: If image is invalid, too large, or processing fails.
    """
    logger.info("Hybrid Vision verification request received")

    # Validate image MIME type header
    if not image.content_type or not image.content_type.startswith("image/"):
        logger.warning(f"Invalid file type for hybrid Vision: {image.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Please upload an image file.",
        )

    # Check file size (max 5MB)
    max_file_size = 5 * 1024 * 1024  # 5MB
    image.file.seek(0, 2)
    file_size = image.file.tell()
    image.file.seek(0)

    if file_size > max_file_size:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum size is 5MB."
        )

    try:
        image_bytes = await image.read()
        outcome = await verification_service.verify_product_by_image(
            image_bytes=image_bytes, mime_type=image.content_type
        )

        if not outcome.is_valid_image:
            raise HTTPException(
                status_code=400,
                detail=outcome.error_message
                or "File type mismatch. The uploaded file does not match the declared content type.",
            )

        return HybridVerificationResponse(
            verification_status=outcome.verification_status,
            confidence=outcome.confidence,
            matched_product=outcome.matched_product,
            extracted_fields=outcome.extracted_fields,
            ai_reasoning=outcome.ai_reasoning,
            alternative_matches=outcome.alternative_matches,
            processing_metadata=outcome.processing_metadata,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid vision verification failed: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid vision verification failed: {str(e)}. Please try again.",
        ) from e
    finally:
        await image.close()



__all__ = ["router"]
