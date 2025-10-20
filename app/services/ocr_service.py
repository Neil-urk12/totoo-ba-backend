"""
Hybrid OCR Service with PaddleOCR + Groq + Gemini Fallback
Implements a three-layer approach for efficient and accurate text extraction from product images.
"""

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from groq import Groq
from PIL import Image

# Initialize services
# Using Tesseract OCR for CPU compatibility

try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if GROQ_AVAILABLE else None
except Exception:
    GROQ_AVAILABLE = False
    groq_client = None

# Initialize Gemini if available
if GEMINI_AVAILABLE:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
    else:
        GEMINI_AVAILABLE = False
        gemini_client = None
else:
    gemini_client = None


# Simple in-memory cache for Gemini results
_gemini_cache: dict[str, dict] = {}


@dataclass
class OCRResult:
    """Result from PaddleOCR extraction"""

    text: str
    bbox: list[list[float]]  # Bounding box coordinates
    confidence: float
    is_low_confidence: bool


@dataclass
class ConfidenceReport:
    """Confidence analysis report"""

    average_confidence: float
    critical_fields_confidence: dict[str, float]
    low_confidence_regions: list[OCRResult]
    needs_gemini_fallback: bool
    reason: str


@dataclass
class ExtractedData:
    """Final extracted and structured data"""

    registration_number: str | None = None
    brand_name: str | None = None
    product_description: str | None = None
    manufacturer: str | None = None
    expiry_date: str | None = None
    batch_number: str | None = None
    net_weight: str | None = None


@dataclass
class ProcessingMetadata:
    """Metadata about the processing pipeline"""

    paddle_time: float = 0.0
    groq_time: float = 0.0
    gemini_time: float = 0.0
    total_time: float = 0.0
    layers_used: list[str] = None
    paddle_confidence: float = 0.0
    gemini_used: bool = False

    def __post_init__(self):
        if self.layers_used is None:
            self.layers_used = []


class HybridOCRService:
    """
    Hybrid OCR service implementing PaddleOCR + Groq + Gemini fallback strategy.
    """

    def __init__(self):
        """Initialize the hybrid OCR service"""
        self.ocr_reader = None
        self._ocr_initialized = False

    def _ensure_ocr_initialized(self):
        """Lazy initialization of Tesseract OCR"""
        if not self._ocr_initialized:
            try:
                import pytesseract

                # Test if tesseract is available
                pytesseract.get_tesseract_version()

                self.ocr_reader = pytesseract
                self.ocr_type = 'tesseract'
                self._ocr_initialized = True
            except Exception:
                self.ocr_reader = None
                self.ocr_type = None
                self._ocr_initialized = True

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for optimal OCR performance.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Preprocessed image as numpy array
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to handle varying lighting
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        return cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)


    def _extract_with_tesseract(self, image_bytes: bytes) -> list[OCRResult]:
        """
        Extract text using Tesseract OCR.

        Args:
            image_bytes: Image bytes to process

        Returns:
            List of OCR results with bounding boxes and confidence
        """
        # Ensure OCR is initialized (lazy loading)
        self._ensure_ocr_initialized()

        if not self.ocr_reader:
            raise RuntimeError("Tesseract OCR is not available")

        # Convert bytes to image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to PIL Image for Tesseract
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Get detailed data from Tesseract
        data = self.ocr_reader.image_to_data(img_pil, output_type='dict')

        # Parse Tesseract results
        ocr_results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text:  # Skip empty text
                # Tesseract confidence is 0-100, convert to 0-1
                confidence = float(data['conf'][i]) / 100.0

                # Get bounding box coordinates
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

                is_low_confidence = confidence < 0.85
                ocr_results.append(
                    OCRResult(
                        text=text,
                        bbox=bbox,
                        confidence=confidence,
                        is_low_confidence=is_low_confidence,
                    )
                )

        return ocr_results

    def _analyze_confidence(self, ocr_results: list[OCRResult]) -> ConfidenceReport:
        """
        Analyze confidence scores and determine if Gemini fallback is needed.

        Args:
            ocr_results: Results from PaddleOCR

        Returns:
            Confidence report with analysis
        """
        if not ocr_results:
            return ConfidenceReport(
                average_confidence=0.0,
                critical_fields_confidence={},
                low_confidence_regions=[],
                needs_gemini_fallback=True,
                reason="No text detected by OCR",
            )

        # Calculate average confidence
        avg_conf = sum(r.confidence for r in ocr_results) / len(ocr_results)

        # Identify low confidence regions
        low_conf_regions = [r for r in ocr_results if r.is_low_confidence]

        # Check for critical field patterns
        critical_fields = {}
        for result in ocr_results:
            text = result.text.upper()
            # Check for registration number patterns
            if any(pattern in text for pattern in ["BR-", "DR-", "FR-", "LTO-"]):
                critical_fields["registration_number"] = result.confidence

        # Determine if Gemini is needed
        needs_gemini = False
        reason = "OCR confidence is sufficient"

        if avg_conf < 0.75:
            needs_gemini = True
            reason = f"Average confidence too low: {avg_conf:.2%}"
        elif critical_fields and any(c < 0.85 for c in critical_fields.values()):
            needs_gemini = True
            reason = "Critical field confidence below threshold"
        elif len(low_conf_regions) > len(ocr_results) * 0.4:
            needs_gemini = True
            reason = f"Too many low confidence regions: {len(low_conf_regions)}/{len(ocr_results)}"

        return ConfidenceReport(
            average_confidence=avg_conf,
            critical_fields_confidence=critical_fields,
            low_confidence_regions=low_conf_regions,
            needs_gemini_fallback=needs_gemini,
            reason=reason,
        )

    async def _extract_with_groq(
        self, raw_text: str, confidence_score: float
    ) -> ExtractedData:
        """
        Use Groq Llama 3.1 8B-instant for structured field extraction.

        Args:
            raw_text: Raw text from PaddleOCR
            confidence_score: Overall confidence from PaddleOCR

        Returns:
            Structured extracted data
        """
        if not GROQ_AVAILABLE or not groq_client:
            raise RuntimeError("Groq API is not available")

        prompt = f"""You are an FDA Philippines product information extractor.

Extract the following fields from the product text below. Be precise and extract ONLY what you see.

TEXT (OCR Confidence: {confidence_score:.0%}):
{raw_text}

EXTRACTION RULES:
1. registration_number: FDA registration number (BR-XXXX, DR-XXXXX, FR-XXXXX, etc.)
2. brand_name: PRIMARY brand name only (e.g., "C2" not "C2 COOL & CLEAN")
3. product_description: Product type, flavor, or description
4. manufacturer: Company or manufacturer name
5. expiry_date: Expiration or best before date
6. batch_number: Batch, lot, or production code
7. net_weight: Net weight or volume

Return ONLY a valid JSON object with these exact field names. Use null for fields not found.
Format: {{"registration_number": "...", "brand_name": "...", ...}}"""

        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            import json

            result_text = completion.choices[0].message.content
            parsed = json.loads(result_text)

            return ExtractedData(
                registration_number=parsed.get("registration_number"),
                brand_name=parsed.get("brand_name"),
                product_description=parsed.get("product_description"),
                manufacturer=parsed.get("manufacturer"),
                expiry_date=parsed.get("expiry_date"),
                batch_number=parsed.get("batch_number"),
                net_weight=parsed.get("net_weight"),
            )

        except Exception as e:
            raise RuntimeError(f"Groq extraction failed: {e}") from e

    def _crop_low_confidence_regions(
        self, image_bytes: bytes, ocr_results: list[OCRResult]
    ) -> bytes:
        """
        Crop image to focus on low-confidence regions for Gemini.

        Args:
            image_bytes: Original image bytes
            ocr_results: OCR results with bounding boxes

        Returns:
            Cropped image bytes focusing on problematic areas
        """
        # For now, return original image
        # In production, implement intelligent cropping based on bounding boxes
        return image_bytes

    async def _extract_with_gemini_fallback(
        self, image_bytes: bytes, mime_type: str, raw_text: str
    ) -> ExtractedData:
        """
        Use Gemini 2.5 Flash as fallback for low-confidence or complex images.

        Args:
            image_bytes: Image bytes (potentially cropped)
            mime_type: Image MIME type
            raw_text: Raw text from PaddleOCR for context

        Returns:
            Structured extracted data from Gemini
        """
        if not GEMINI_AVAILABLE or not gemini_client:
            raise RuntimeError("Gemini API is not available")

        # Check cache first
        cache_key = hashlib.md5(image_bytes).hexdigest()
        if cache_key in _gemini_cache:
            cached = _gemini_cache[cache_key]
            return ExtractedData(**cached)

        prompt = f"""You are an expert FDA Philippines product verification assistant.

The initial OCR detected this text (but with low confidence):
{raw_text[:500]}

Analyze this product image carefully and extract ALL visible text and product information.

Extract these specific fields:
- registration_number: FDA Philippines registration number (BR-XXXX, DR-XXXXX, FR-XXXXX, etc.)
- brand_name: PRIMARY brand name only
- product_description: Product type, flavor, or description
- manufacturer: Manufacturer or distributor name
- expiry_date: Expiration date
- batch_number: Batch or lot number
- net_weight: Net weight or volume

IMPORTANT: Extract exact text as visible. Set to null if not visible or unclear.

Return as JSON with these exact field names."""

        try:
            from pydantic import BaseModel, Field

            class GeminiExtractedFields(BaseModel):
                registration_number: str | None = Field(None)
                brand_name: str | None = Field(None)
                product_description: str | None = Field(None)
                manufacturer: str | None = Field(None)
                expiry_date: str | None = Field(None)
                batch_number: str | None = Field(None)
                net_weight: str | None = Field(None)

            # Create image part
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

            # Generate structured content
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiExtractedFields,
                    temperature=0.1,
                ),
            )

            result: GeminiExtractedFields = response.parsed
            extracted = ExtractedData(
                registration_number=result.registration_number,
                brand_name=result.brand_name,
                product_description=result.product_description,
                manufacturer=result.manufacturer,
                expiry_date=result.expiry_date,
                batch_number=result.batch_number,
                net_weight=result.net_weight,
            )

            # Cache result
            _gemini_cache[cache_key] = {
                "registration_number": result.registration_number,
                "brand_name": result.brand_name,
                "product_description": result.product_description,
                "manufacturer": result.manufacturer,
                "expiry_date": result.expiry_date,
                "batch_number": result.batch_number,
                "net_weight": result.net_weight,
            }

            return extracted

        except Exception as e:
            raise RuntimeError(f"Gemini extraction failed: {e}") from e

    def _merge_results(
        self, groq_data: ExtractedData, gemini_data: ExtractedData | None
    ) -> ExtractedData:
        """
        Merge results from Groq and Gemini, prioritizing higher confidence fields.

        Args:
            groq_data: Data extracted by Groq
            gemini_data: Data extracted by Gemini (if used)

        Returns:
            Merged extracted data
        """
        if not gemini_data:
            return groq_data

        # Merge: prefer non-null values, prioritize Gemini for critical fields
        return ExtractedData(
            registration_number=gemini_data.registration_number
            or groq_data.registration_number,
            brand_name=gemini_data.brand_name or groq_data.brand_name,
            product_description=gemini_data.product_description
            or groq_data.product_description,
            manufacturer=gemini_data.manufacturer or groq_data.manufacturer,
            expiry_date=gemini_data.expiry_date or groq_data.expiry_date,
            batch_number=gemini_data.batch_number or groq_data.batch_number,
            net_weight=gemini_data.net_weight or groq_data.net_weight,
        )

    async def extract_product_info(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> tuple[ExtractedData, ProcessingMetadata]:
        """
        Main entry point: Extract product information using hybrid approach.

        Args:
            image_bytes: Image bytes to process
            mime_type: Image MIME type

        Returns:
            Tuple of (extracted_data, processing_metadata)
        """
        metadata = ProcessingMetadata()
        start_time = time.time()


        # Layer 1: Tesseract OCR for fast text extraction
        ocr_start = time.time()
        try:
            ocr_results = self._extract_with_tesseract(image_bytes)
            raw_text = " ".join([r.text for r in ocr_results])
            confidence_report = self._analyze_confidence(ocr_results)

            metadata.paddle_time = time.time() - ocr_start
            metadata.paddle_confidence = confidence_report.average_confidence
            metadata.layers_used.append("Tesseract OCR")


        except Exception as e:
            # Fallback to Gemini if OCR fails completely
            raw_text = ""
            confidence_report = ConfidenceReport(
                average_confidence=0.0,
                critical_fields_confidence={},
                low_confidence_regions=[],
                needs_gemini_fallback=True,
                reason=f"OCR failed: {e}",
            )
            metadata.paddle_time = time.time() - ocr_start

        # Layer 2: Groq for structured extraction
        groq_start = time.time()
        try:
            groq_data = await self._extract_with_groq(
                raw_text, confidence_report.average_confidence
            )
            metadata.groq_time = time.time() - groq_start
            metadata.layers_used.append("Groq Llama 3.1")

        except Exception:
            # If Groq fails, create empty data and force Gemini
            groq_data = ExtractedData()
            metadata.groq_time = time.time() - groq_start
            confidence_report.needs_gemini_fallback = True
            confidence_report.reason = "Groq extraction failed"

        # Layer 3: Gemini fallback (if needed)
        gemini_data = None
        if confidence_report.needs_gemini_fallback:
            gemini_start = time.time()
            try:
                gemini_data = await self._extract_with_gemini_fallback(
                    image_bytes, mime_type, raw_text
                )
                metadata.gemini_time = time.time() - gemini_start
                metadata.gemini_used = True
                metadata.layers_used.append("Gemini 2.5 Flash")
            except Exception:
                metadata.gemini_time = time.time() - gemini_start

        # Merge results
        final_data = self._merge_results(groq_data, gemini_data)

        metadata.total_time = time.time() - start_time


        return final_data, metadata

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return {"cache_size": len(_gemini_cache), "cache_keys": list(_gemini_cache.keys())}

    def clear_cache(self):
        """Clear the Gemini result cache"""
        _gemini_cache.clear()


# Global service instance
_ocr_service_instance: HybridOCRService | None = None


def get_ocr_service() -> HybridOCRService:
    """Get or create the global OCR service instance"""
    global _ocr_service_instance
    if _ocr_service_instance is None:
        _ocr_service_instance = HybridOCRService()
    return _ocr_service_instance
