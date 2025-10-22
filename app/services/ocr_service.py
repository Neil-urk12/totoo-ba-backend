"""
Hybrid OCR Service with Groq Vision + Cerebras LLM + Gemini Fallback
Implements a three-layer approach for efficient and accurate text extraction from product images.
"""

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

from groq import Groq
from loguru import logger

# Import Cerebras SDK
try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = bool(os.getenv("CEREBRAS_API_KEY"))
    cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY")) if CEREBRAS_AVAILABLE else None
except ImportError:
    CEREBRAS_AVAILABLE = False
    cerebras_client = None
except Exception:
    CEREBRAS_AVAILABLE = False
    cerebras_client = None

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
    """Result from OCR extraction"""

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

    groq_vision_time: float = 0.0
    cerebras_time: float = 0.0
    gemini_time: float = 0.0
    total_time: float = 0.0
    layers_used: list[str] = None
    groq_vision_confidence: float = 0.0
    gemini_used: bool = False

    def __post_init__(self):
        if self.layers_used is None:
            self.layers_used = []


class HybridOCRService:
    """
    Hybrid OCR service implementing Groq Vision + Cerebras LLM + Gemini fallback strategy.
    Uses a three-layer approach:
    1. Groq Llama 4 Scout Vision for image OCR
    2. Cerebras Llama for structured field extraction
    3. Gemini 2.5 Flash as fallback for low-confidence results
    """

    def __init__(self):
        """Initialize the hybrid OCR service"""
        pass



    async def _extract_with_groq_vision(self, image_bytes: bytes, mime_type: str) -> list[OCRResult]:
        """
        Extract text using Groq Llama 4 Scout vision model for OCR.

        Args:
            image_bytes: Image bytes to process
            mime_type: Image MIME type

        Returns:
            List of OCR results with simulated bounding boxes and confidence
        """
        logger.debug("Starting Groq Llama 4 Scout vision OCR extraction")

        if not GROQ_AVAILABLE or not groq_client:
            logger.error("Groq API key not configured")
            raise RuntimeError("Groq API is not available")

        # Convert image to base64 for Groq
        import base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{image_base64}"

        prompt = """You are an expert OCR system. Extract ALL visible text from this product image.

For each piece of text you find, provide:
1. The exact text as you see it
2. A confidence score (0.0-1.0) based on text clarity
3. Approximate position (top, middle, bottom of image)

Return a JSON object with an "items" array containing the extracted text:
{
  "items": [
    {
      "text": "exact text found",
      "confidence": 0.95,
      "position": "top|middle|bottom"
    }
  ]
}

Focus on extracting:
- Registration numbers (BR-, DR-, FR-, etc.)
- Brand names
- Product descriptions
- Manufacturer names
- Dates
- Batch numbers
- Net weights/volumes
- Any other visible text

Be thorough and extract even small or partially visible text."""

        try:
            # Temporarily reduce logging level for groq module to prevent image data logging
            import logging
            groq_logger = logging.getLogger("groq")
            original_level = groq_logger.level
            # If the current level is DEBUG (10) or lower, temporarily set to WARNING (30)
            # to prevent request details with image data from being logged
            if original_level <= logging.DEBUG:
                groq_logger.setLevel(logging.WARNING)

            completion = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            # Restore original logging level after the API call
            groq_logger.setLevel(original_level)

            # Parse JSON response
            import json
            result_text = completion.choices[0].message.content
            response_data = json.loads(result_text)

            # Extract items array from response
            extracted_items = response_data.get("items", [])

            # Convert to OCRResult format
            ocr_results = []
            for _i, item in enumerate(extracted_items):
                if isinstance(item, dict):
                    text = item.get("text", "").strip()
                    if text:
                        confidence = float(item.get("confidence", 0.8))
                        position = item.get("position", "middle")

                        # Create simulated bounding box based on position
                        if position == "top":
                            bbox = [[0, 0], [100, 0], [100, 30], [0, 30]]
                        elif position == "bottom":
                            bbox = [[0, 170], [100, 170], [100, 200], [0, 200]]
                        else:  # middle
                            bbox = [[0, 85], [100, 85], [100, 115], [0, 115]]

                        is_low_confidence = confidence < 0.85
                        ocr_results.append(
                            OCRResult(
                                text=text,
                                bbox=bbox,
                                confidence=confidence,
                                is_low_confidence=is_low_confidence,
                            )
                        )

            if ocr_results:
                avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results)
                logger.info(
                    f"Groq vision OCR extracted {len(ocr_results)} text blocks, "
                    f"avg_confidence={avg_confidence:.2f}"
                )
            else:
                logger.warning("Groq vision OCR found no text in image")

            return ocr_results

        except json.JSONDecodeError as e:
            logger.error(f"Groq vision OCR returned invalid JSON: {str(e)}")
            raise RuntimeError(f"Groq vision OCR returned invalid JSON: {e}") from e
        except KeyError as e:
            logger.error(f"Groq vision OCR response missing expected field: {str(e)}")
            raise RuntimeError(f"Groq vision OCR response missing expected field: {e}") from e
        except Exception as e:
            logger.error(f"Groq vision OCR extraction failed: {str(e)}")
            raise RuntimeError(f"Groq vision OCR extraction failed: {e}") from e

    def _analyze_confidence(self, ocr_results: list[OCRResult]) -> ConfidenceReport:
        """
        Analyze confidence scores and determine if Gemini fallback is needed.

        Args:
            ocr_results: Results from OCR extraction

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

        report = ConfidenceReport(
            average_confidence=avg_conf,
            critical_fields_confidence=critical_fields,
            low_confidence_regions=low_conf_regions,
            needs_gemini_fallback=needs_gemini,
            reason=reason,
        )

        logger.info(
            f"Confidence analysis: avg={avg_conf:.2%}, "
            f"gemini_needed={needs_gemini}, reason={reason}"
        )
        return report

    async def _extract_with_groq_llama31(
        self, raw_text: str, confidence_score: float
    ) -> ExtractedData:
        """
        Use Groq Llama 3.1 8B for structured field extraction.

        Args:
            raw_text: Raw text from Groq vision OCR
            confidence_score: Overall confidence from vision OCR

        Returns:
            Structured extracted data
        """
        logger.debug(f"Starting Groq Llama 3.1 8B extraction with {len(raw_text)} chars of text")
        if not GROQ_AVAILABLE or not groq_client:
            logger.error("Groq API key not configured")
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
                model="llama-3.1-8b-instant",  # Using Groq Llama 3.1 8B model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            import json

            result_text = completion.choices[0].message.content
            parsed = json.loads(result_text)

            extracted = ExtractedData(
                registration_number=parsed.get("registration_number"),
                brand_name=parsed.get("brand_name"),
                product_description=parsed.get("product_description"),
                manufacturer=parsed.get("manufacturer"),
                expiry_date=parsed.get("expiry_date"),
                batch_number=parsed.get("batch_number"),
                net_weight=parsed.get("net_weight"),
            )

            logger.info(
                f"Groq Llama 3.1 8B extraction successful: "
                f"reg_num={'✓' if extracted.registration_number else '✗'}, "
                f"brand={'✓' if extracted.brand_name else '✗'}, "
                f"product={'✓' if extracted.product_description else '✗'}"
            )
            return extracted

        except Exception as e:
            logger.error(f"Groq Llama 3.1 8B extraction failed: {str(e)}")
            raise RuntimeError(f"Groq Llama 3.1 8B extraction failed: {e}") from e

    # COMMENTED OUT - Cerebras extraction method (for latency comparison)
    async def _extract_with_cerebras(
        self, raw_text: str, confidence_score: float
    ) -> ExtractedData:
        """
        Use Cerebras Llama for structured field extraction.

        Args:
            raw_text: Raw text from Groq vision OCR
            confidence_score: Overall confidence from vision OCR

        Returns:
            Structured extracted data
        """
        logger.debug(f"Starting Cerebras extraction with {len(raw_text)} chars of text")
        if not CEREBRAS_AVAILABLE or not cerebras_client:
            logger.error("Cerebras API key not configured")
            raise RuntimeError("Cerebras API is not available")

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
            completion = cerebras_client.chat.completions.create(
                model="llama3.1-8b",  # Using Cerebras Llama 3.1 8B model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            # Parse JSON response
            import json

            result_text = completion.choices[0].message.content
            parsed = json.loads(result_text)

            extracted = ExtractedData(
                registration_number=parsed.get("registration_number"),
                brand_name=parsed.get("brand_name"),
                product_description=parsed.get("product_description"),
                manufacturer=parsed.get("manufacturer"),
                expiry_date=parsed.get("expiry_date"),
                batch_number=parsed.get("batch_number"),
                net_weight=parsed.get("net_weight"),
            )

            logger.info(
                f"Cerebras extraction successful: "
                f"reg_num={'✓' if extracted.registration_number else '✗'}, "
                f"brand={'✓' if extracted.brand_name else '✗'}, "
                f"product={'✓' if extracted.product_description else '✗'}"
            )
            return extracted

        except Exception as e:
            logger.error(f"Cerebras extraction failed: {str(e)}")
            raise RuntimeError(f"Cerebras extraction failed: {e}") from e

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
            raw_text: Raw text from Groq vision OCR for context

        Returns:
            Structured extracted data from Gemini
        """
        logger.info("Starting Gemini fallback extraction")
        if not GEMINI_AVAILABLE or not gemini_client:
            logger.error("Gemini API key not configured")
            raise RuntimeError("Gemini API is not available")

        # Check cache first
        cache_key = hashlib.md5(image_bytes).hexdigest()
        if cache_key in _gemini_cache:
            logger.info("Gemini result found in cache")
            cached = _gemini_cache[cache_key]
            return ExtractedData(**cached)

        prompt = f"""You are an expert FDA Philippines product verification assistant.

The initial Groq vision OCR detected this text (but with low confidence):
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

            # Temporarily reduce logging level for genai module to prevent image data logging
            import logging
            genai_logger = logging.getLogger("google")
            original_level = genai_logger.level
            # If the current level is DEBUG (10) or lower, temporarily set to WARNING (30)
            # to prevent request details with image data from being logged
            if original_level <= logging.DEBUG:
                genai_logger.setLevel(logging.WARNING)

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

            # Restore original logging level after the API call
            genai_logger.setLevel(original_level)

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

            logger.info(
                f"Gemini extraction successful: "
                f"reg_num={'✓' if result.registration_number else '✗'}, "
                f"brand={'✓' if result.brand_name else '✗'}, "
                f"product={'✓' if result.product_description else '✗'}"
            )

            return extracted

        except Exception as e:
            logger.error(f"Gemini extraction failed: {str(e)}")
            raise RuntimeError(f"Gemini extraction failed: {e}") from e

    def _merge_results(
        self, groq_data: ExtractedData, gemini_data: ExtractedData | None
    ) -> ExtractedData:
        """
        Merge results from Groq and Gemini, prioritizing higher confidence fields.

        Args:
            groq_data: Data extracted by Groq Llama 3.1 8B
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
        logger.info("Starting hybrid OCR pipeline")
        metadata = ProcessingMetadata()
        start_time = time.time()


        # Layer 1: Groq Llama 4 Scout Vision OCR for text extraction
        ocr_start = time.time()
        try:
            ocr_results = await self._extract_with_groq_vision(image_bytes, mime_type)
            raw_text = " ".join([r.text for r in ocr_results])
            confidence_report = self._analyze_confidence(ocr_results)

            metadata.groq_vision_time = time.time() - ocr_start
            metadata.groq_vision_confidence = confidence_report.average_confidence
            metadata.layers_used.append("Groq Llama 4 Scout Vision")


        except Exception as e:
            # Fallback to Gemini if Groq vision OCR fails completely
            logger.warning(f"Groq vision OCR failed: {str(e)}. Will use Gemini fallback.")
            raw_text = ""
            confidence_report = ConfidenceReport(
                average_confidence=0.0,
                critical_fields_confidence={},
                low_confidence_regions=[],
                needs_gemini_fallback=True,
                reason=f"Groq vision OCR failed: {e}",
            )
            metadata.groq_vision_time = time.time() - ocr_start

        # Layer 2: Groq Llama 3.1 8B for structured extraction (switched back for latency comparison)
        groq_start = time.time()
        try:
            groq_data = await self._extract_with_groq_llama31(
                raw_text, confidence_report.average_confidence
            )
            metadata.cerebras_time = time.time() - groq_start  # Keep same field for consistency
            metadata.layers_used.append("Groq Llama 3.1 8B")

        except Exception as e:
            # If Groq fails, create empty data and force Gemini
            logger.warning(f"Groq extraction failed: {str(e)}. Will use Gemini fallback.")
            groq_data = ExtractedData()
            metadata.cerebras_time = time.time() - groq_start
            confidence_report.needs_gemini_fallback = True
            confidence_report.reason = "Groq extraction failed"
        
        # COMMENTED OUT - Cerebras Layer 2 (for latency comparison)
        # cerebras_start = time.time()
        # try:
        #     cerebras_data = await self._extract_with_cerebras(
        #         raw_text, confidence_report.average_confidence
        #     )
        #     metadata.cerebras_time = time.time() - cerebras_start
        #     metadata.layers_used.append("Cerebras Llama 3.1 8B")
        # except Exception as e:
        #     logger.warning(f"Cerebras extraction failed: {str(e)}. Will use Gemini fallback.")
        #     cerebras_data = ExtractedData()
        #     metadata.cerebras_time = time.time() - cerebras_start
        #     confidence_report.needs_gemini_fallback = True
        #     confidence_report.reason = "Cerebras extraction failed"

        # Layer 3: Gemini fallback (if needed)
        gemini_data = None
        if confidence_report.needs_gemini_fallback:
            logger.info(f"Triggering Gemini fallback: {confidence_report.reason}")
            gemini_start = time.time()
            try:
                gemini_data = await self._extract_with_gemini_fallback(
                    image_bytes, mime_type, raw_text
                )
                metadata.gemini_time = time.time() - gemini_start
                metadata.gemini_used = True
                metadata.layers_used.append("Gemini 2.5 Flash")
            except Exception as e:
                logger.error(f"Gemini fallback also failed: {str(e)}")
                metadata.gemini_time = time.time() - gemini_start

        # Merge results
        final_data = self._merge_results(groq_data, gemini_data)

        metadata.total_time = time.time() - start_time

        logger.success(
            f"Hybrid OCR pipeline complete: "
            f"layers={', '.join(metadata.layers_used)}, "
            f"total_time={metadata.total_time:.2f}s, "
            f"gemini_used={metadata.gemini_used}"
        )

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
