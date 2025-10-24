"""
Hybrid Vision Service with Groq Vision + Groq Llama 3.1 + Groq Llama 4 Maverick Fallback
Implements a three-layer approach for efficient and accurate text extraction from product images.
"""

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

from groq import Groq
from loguru import logger

# Initialize services




try:
    GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if GROQ_AVAILABLE else None
except Exception:
    GROQ_AVAILABLE = False
    groq_client = None




# Simple in-memory cache for Groq fallback results
_groq_fallback_cache: dict[str, dict] = {}


@dataclass
class VisionResult:
    """Result from vision extraction"""

    text: str
    bbox: list[list[float]]  # Bounding box coordinates
    confidence: float
    is_low_confidence: bool


@dataclass
class ConfidenceReport:
    """Confidence analysis report"""

    average_confidence: float
    critical_fields_confidence: dict[str, float]
    low_confidence_regions: list[VisionResult]
    needs_groq_fallback: bool
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
    groq_llama31_time: float = 0.0
    groq_fallback_time: float = 0.0
    total_time: float = 0.0
    layers_used: list[str] = None
    groq_vision_confidence: float = 0.0
    groq_fallback_used: bool = False

    def __post_init__(self):
        if self.layers_used is None:
            self.layers_used = []


class HybridVisionService:
    """
    Hybrid vision service implementing Groq Vision + Groq Llama 3.1 + Groq Llama 4 Maverick fallback strategy.
    Uses a three-layer approach:
    1. Groq Llama 4 Scout Vision for image text extraction
    2. Groq Llama 3.1 8B for structured field extraction
    3. Groq Llama 4 Maverick 17b 128e as fallback for low-confidence results
    """

    def __init__(self):
        """Initialize the hybrid vision service"""
        pass



    async def _extract_with_groq_vision(self, image_bytes: bytes, mime_type: str) -> list[VisionResult]:
        """
        Extract text using Groq Llama 4 Scout vision model.

        Args:
            image_bytes: Image bytes to process
            mime_type: Image MIME type

        Returns:
            List of vision results with simulated bounding boxes and confidence
        """
        logger.debug("Starting Groq Llama 4 Scout vision extraction")

        if not GROQ_AVAILABLE or not groq_client:
            logger.error("Groq API key not configured")
            raise RuntimeError("Groq API is not available")

        # Convert image to base64 for Groq
        import base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{image_base64}"

        prompt = """You are an expert vision system. Extract ALL visible text from this product image.

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

Focus on extracting ALL visible text, especially:
- Registration numbers (BR-, DR-, FR-, etc.)
- Brand names (include FULL brand with sub-brand text)
- Product type/category (e.g., "Tonkatsu Sauce", "Green Tea", "Capsule")
- Product descriptions (flavors, formulations, ingredients)
- Manufacturer names
- Dates, batch numbers, net weights/volumes
- Any promotional text
- Any other visible text in ANY language (English, Japanese, Filipino, etc.)

CRITICAL: Pay special attention to:
- Large/prominent text indicating the product type (e.g., "Tonkatsu Sauce", "Soy Sauce", "Tea")
- Text written in English or romaji even if other languages are present
- Product category labels that describe what the product IS

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

            # Convert to VisionResult format
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
                            VisionResult(
                                text=text,
                                bbox=bbox,
                                confidence=confidence,
                                is_low_confidence=is_low_confidence,
                            )
                        )

            if ocr_results:
                avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results)
                logger.info(
                    f"Groq vision extraction extracted {len(ocr_results)} text blocks, "
                    f"avg_confidence={avg_confidence:.2f}"
                )
            else:
                logger.warning("Groq vision extraction found no text in image")

            return ocr_results

        except json.JSONDecodeError as e:
            logger.error(f"Groq vision extraction returned invalid JSON: {str(e)}")
            raise RuntimeError(f"Groq vision extraction returned invalid JSON: {e}") from e
        except KeyError as e:
            logger.error(f"Groq vision extraction response missing expected field: {str(e)}")
            raise RuntimeError(f"Groq vision extraction response missing expected field: {e}") from e
        except Exception as e:
            logger.error(f"Groq vision extraction failed: {str(e)}")
            raise RuntimeError(f"Groq vision extraction failed: {e}") from e

    def _clean_brand_name(self, brand_name: str) -> str:
        """
        Clean extracted brand name by removing common promotional and dosage text.
        
        Args:
            brand_name: Raw extracted brand name
            
        Returns:
            Cleaned brand name
        """
        if not brand_name:
            return brand_name
            
        import re
        
        # List of patterns to remove (case-insensitive)
        removal_patterns = [
            r'\b\d+\s*mg\b',  # Dosage: "500mg", "500 mg"
            r'\b\d+\s*ml\b',  # Volume: "100ml", "100 ml"
            r'\b\d+\s*g\b',   # Weight: "10g", "10 g"
            r'\bcapsule[s]?\b',  # Product form
            r'\btablet[s]?\b',
            r'\bsyrup\b',
            r'\bsuspension\b',
            r'\bbuy\s+\d+.*?free\b',  # Promos: "BUY 5 GET 1 FREE"
            r'\bsulit\s+\d+\s+days?\b',  # "Sulit 2 Days"
            r'\b\d+\s+capsule[s]?\b',  # "6 Capsules"
            r'\b\d+\s+tablet[s]?\b',
            r'\bpack\b',
            r'^\d+\+\d+\b',  # "5+1"
        ]
        
        cleaned = brand_name
        for pattern in removal_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # If cleaning removed everything, return original
        if not cleaned or len(cleaned) < 2:
            # Try to extract just manufacturer + single word brand
            # E.g., "UNILAB Solmux" -> "Solmux"
            words = brand_name.split()
            if len(words) >= 2:
                # Return last meaningful word that's likely the brand
                for word in reversed(words):
                    if len(word) > 2 and not word.lower() in ['inc', 'corp', 'co']:
                        cleaned = word
                        break
        
        logger.debug(f"Brand name cleaned: '{brand_name}' → '{cleaned}'")
        return cleaned.strip()

    def _analyze_confidence(self, vision_results: list[VisionResult]) -> ConfidenceReport:
        """
        Analyze confidence scores and determine if Groq fallback is needed.

        Args:
            vision_results: Results from vision extraction

        Returns:
            Confidence report with analysis
        """
        if not vision_results:
            return ConfidenceReport(
                average_confidence=0.0,
                critical_fields_confidence={},
                low_confidence_regions=[],
                needs_groq_fallback=True,
                reason="No text detected by vision model",
            )

        # Calculate average confidence
        avg_conf = sum(r.confidence for r in vision_results) / len(vision_results)

        # Identify low confidence regions
        low_conf_regions = [r for r in vision_results if r.is_low_confidence]

        # Check for critical field patterns
        critical_fields = {}
        for result in vision_results:
            text = result.text.upper()
            # Check for registration number patterns
            if any(pattern in text for pattern in ["BR-", "DR-", "FR-", "LTO-"]):
                critical_fields["registration_number"] = result.confidence

        # Determine if Groq fallback is needed
        needs_groq_fallback = False
        reason = "Vision confidence is sufficient"

        if avg_conf < 0.75:
            needs_groq_fallback = True
            reason = f"Average confidence too low: {avg_conf:.2%}"
        elif critical_fields and any(c < 0.85 for c in critical_fields.values()):
            needs_groq_fallback = True
            reason = "Critical field confidence below threshold"
        elif len(low_conf_regions) > len(vision_results) * 0.4:
            needs_groq_fallback = True
            reason = f"Too many low confidence regions: {len(low_conf_regions)}/{len(vision_results)}"

        report = ConfidenceReport(
            average_confidence=avg_conf,
            critical_fields_confidence=critical_fields,
            low_confidence_regions=low_conf_regions,
            needs_groq_fallback=needs_groq_fallback,
            reason=reason,
        )

        logger.info(
            f"Confidence analysis: avg={avg_conf:.2%}, "
            f"groq_fallback_needed={needs_groq_fallback}, reason={reason}"
        )
        return report

    async def _extract_with_groq_llama31(
        self, raw_text: str, confidence_score: float
    ) -> ExtractedData:
        """
        Use Groq Llama 3.1 8B for structured field extraction.

        Args:
            raw_text: Raw text from Groq vision extraction
            confidence_score: Overall confidence from vision extraction

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
2. brand_name: PRIMARY brand name only - the company/product brand (e.g., "Solmux", "Yamamori", "C2")
3. product_description: CRITICAL - The product type/category and key attributes
   - For sauces: "Tonkatsu Sauce", "Teriyaki Sauce", "Soy Sauce"
   - For drugs: "Carbocisteine 500mg Capsule"
   - For beverages: "Apple Green Tea", "Green Apple Juice Tea"
   - For any product: Extract the MAIN product type/flavor/category
4. manufacturer: Company or manufacturer name
5. expiry_date: Expiration or best before date
6. batch_number: Batch, lot, or production code
7. net_weight: Net weight or volume (e.g., "500ml", "220ml", "10 tablets")

CRITICAL INSTRUCTIONS:
- brand_name: Extract ONLY the brand (e.g., "Yamamori", NOT "Yamamori Tonkatsu Sauce")
- product_description: Extract the product TYPE (e.g., "Tonkatsu Sauce", NOT just null)
  → This is the MOST IMPORTANT field - extract what the product IS
  → Look for prominent text describing the product category/type
  → Include sauce types, drink flavors, drug formulations, etc.

What NOT to include in brand_name:
❌ Promotional text (e.g., "BUY 5 GET 1 FREE", "Sulit 2 Days Pack")
❌ Dosage/strength (e.g., "500 mg", "500mg Capsule")
❌ Product type (e.g., "Tonkatsu Sauce", "Capsule", "Green Tea")
❌ Product category (e.g., "Capsule", "Tablet", "Syrup")
❌ Quantity (e.g., "6 Capsules", "100ml")
❌ Generic/active ingredient (e.g., "Carbocisteine", "Paracetamol")

✓ ONLY extract the actual brand/product name (e.g., "Solmux", "Biogesic", "C2 COOL & CLEAN")

EXAMPLES:
- If you see "BUY 5 GET 1 FREE UNILAB Carbocisteine Solmux 500mg Capsule"
  → brand_name: "Solmux"
  → product_description: "Carbocisteine 500mg Capsule"
  → manufacturer: "UNILAB"

- If you see "C2 COOL & CLEAN APPLE GREEN TEA"
  → brand_name: "C2 COOL & CLEAN"
  → product_description: "APPLE GREEN TEA"

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

            # Clean and validate brand name to remove promotional/dosage text
            brand_name = parsed.get("brand_name")
            if brand_name:
                brand_name = self._clean_brand_name(brand_name)

            extracted = ExtractedData(
                registration_number=parsed.get("registration_number"),
                brand_name=brand_name,
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


    def _crop_low_confidence_regions(
        self, image_bytes: bytes, ocr_results: list[VisionResult]
    ) -> bytes:
        """
        Crop image to focus on low-confidence regions for Groq fallback.

        Args:
            image_bytes: Original image bytes
            ocr_results: OCR results with bounding boxes

        Returns:
            Cropped image bytes focusing on problematic areas
        """
        # For now, return original image
        # In production, implement intelligent cropping based on bounding boxes
        return image_bytes

    async def _extract_with_groq_fallback(
        self, image_bytes: bytes, mime_type: str, raw_text: str
    ) -> ExtractedData:
        """
        Use Groq Llama 4 Maverick 17b 128e as fallback for low-confidence or complex images.

        Args:
            image_bytes: Image bytes (potentially cropped)
            mime_type: Image MIME type
            raw_text: Raw text from Groq vision extraction for context

        Returns:
            Structured extracted data from Groq Llama 4 Maverick
        """
        logger.info("Starting Groq Llama 4 Maverick fallback extraction")
        if not GROQ_AVAILABLE or not groq_client:
            logger.error("Groq API key not configured")
            raise RuntimeError("Groq API is not available")

        # Check cache first
        cache_key = hashlib.md5(image_bytes).hexdigest()
        if cache_key in _groq_fallback_cache:
            logger.info("Groq fallback result found in cache")
            cached = _groq_fallback_cache[cache_key]
            return ExtractedData(**cached)

        # Convert image to base64 for Groq
        import base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{image_base64}"

        prompt = f"""You are an expert FDA Philippines product verification assistant.

The initial Groq vision extraction detected this text (but with low confidence):
{raw_text[:500]}

Analyze this product image carefully and extract ALL visible text and product information.

Extract these specific fields:
- registration_number: FDA Philippines registration number (BR-XXXX, DR-XXXXX, FR-XXXXX, etc.)
- brand_name: The company/product brand name ONLY (e.g., "Yamamori", "Solmux")
- product_description: CRITICAL - The product TYPE/CATEGORY (e.g., "Tonkatsu Sauce", "Teriyaki Sauce", "Carbocisteine Capsule")
  → This is THE MOST IMPORTANT field
  → Extract what the product IS (sauce type, drink flavor, drug formulation, etc.)
  → Look for large/prominent text describing the product category
- manufacturer: Manufacturer or distributor name
- expiry_date: Expiration date
- batch_number: Batch or lot number
- net_weight: Net weight or volume

CRITICAL INSTRUCTIONS:
- brand_name: Extract ONLY the brand (e.g., "Yamamori", NOT "Yamamori Tonkatsu Sauce")
- product_description: Extract the product TYPE (e.g., "Tonkatsu Sauce", "Soy Sauce", "Green Tea")
  → DO NOT leave this null - extract the product category/type that's prominently displayed

What NOT to include in brand_name:
❌ Promotional text (e.g., "BUY 5 GET 1 FREE")
❌ Dosage/strength (e.g., "500 mg")
❌ Product type (e.g., "Tonkatsu Sauce", "Capsule")

Return as JSON with these exact field names. Set to null if not visible or unclear."""

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
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
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
            parsed = json.loads(result_text)

            # Clean brand name
            brand_name = parsed.get("brand_name")
            if brand_name:
                brand_name = self._clean_brand_name(brand_name)

            extracted = ExtractedData(
                registration_number=parsed.get("registration_number"),
                brand_name=brand_name,
                product_description=parsed.get("product_description"),
                manufacturer=parsed.get("manufacturer"),
                expiry_date=parsed.get("expiry_date"),
                batch_number=parsed.get("batch_number"),
                net_weight=parsed.get("net_weight"),
            )

            # Cache result
            _groq_fallback_cache[cache_key] = {
                "registration_number": extracted.registration_number,
                "brand_name": extracted.brand_name,
                "product_description": extracted.product_description,
                "manufacturer": extracted.manufacturer,
                "expiry_date": extracted.expiry_date,
                "batch_number": extracted.batch_number,
                "net_weight": extracted.net_weight,
            }

            logger.info(
                f"Groq Llama 4 Maverick extraction successful: "
                f"reg_num={'✓' if extracted.registration_number else '✗'}, "
                f"brand={'✓' if extracted.brand_name else '✗'}, "
                f"product={'✓' if extracted.product_description else '✗'}"
            )

            return extracted

        except Exception as e:
            logger.error(f"Groq Llama 4 Maverick extraction failed: {str(e)}")
            raise RuntimeError(f"Groq Llama 4 Maverick extraction failed: {e}") from e

    def _merge_results(
        self, groq_data: ExtractedData, groq_fallback_data: ExtractedData | None
    ) -> ExtractedData:
        """
        Merge results from Groq and Groq fallback, prioritizing higher confidence fields.

        Args:
            groq_data: Data extracted by Groq Llama 3.1 8B
            groq_fallback_data: Data extracted by Groq Llama 4 Maverick (if used)

        Returns:
            Merged extracted data
        """
        if not groq_fallback_data:
            return groq_data

        # Merge: prefer non-null values, prioritize Groq fallback for critical fields
        merged = ExtractedData(
            registration_number=groq_fallback_data.registration_number
            or groq_data.registration_number,
            brand_name=groq_fallback_data.brand_name or groq_data.brand_name,
            product_description=groq_fallback_data.product_description
            or groq_data.product_description,
            manufacturer=groq_fallback_data.manufacturer or groq_data.manufacturer,
            expiry_date=groq_fallback_data.expiry_date or groq_data.expiry_date,
            batch_number=groq_fallback_data.batch_number or groq_data.batch_number,
            net_weight=groq_fallback_data.net_weight or groq_data.net_weight,
        )
        
        return merged

    def _extract_product_keywords(self, vision_results: list[VisionResult], brand_name: str | None) -> str | None:
        """
        Extract product description keywords from vision results when structured extraction fails.
        
        This is a fallback to help identify product type when the LLM misses it.
        
        Args:
            vision_results: Raw vision extraction results
            brand_name: Extracted brand name (to avoid including it in description)
            
        Returns:
            Likely product description or None
        """
        if not vision_results:
            return None
            
        # Common product type keywords that indicate product description
        product_keywords = [
            # Sauces
            'sauce', 'tonkatsu', 'teriyaki', 'soy', 'worcestershire', 'oyster',
            # Beverages
            'tea', 'juice', 'drink', 'milk', 'coffee', 'soda',
            # Drugs
            'capsule', 'tablet', 'syrup', 'suspension', 'drops',
            # Food
            'chips', 'noodles', 'pasta', 'rice', 'oil',
        ]
        
        # Extract text blocks that likely contain product description
        candidates = []
        for result in vision_results:
            text = result.text.strip()
            text_lower = text.lower()
            
            # Skip if it's the brand name
            if brand_name and brand_name.lower() in text_lower:
                continue
                
            # Check if text contains product keywords
            for keyword in product_keywords:
                if keyword in text_lower:
                    # Found a likely product description
                    candidates.append((text, result.confidence, len(text.split())))
                    break
        
        if not candidates:
            return None
            
        # Sort by confidence and word count (prefer longer, confident descriptions)
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Return the best candidate
        best_description = candidates[0][0]
        logger.debug(f"Extracted product keywords fallback: '{best_description}'")
        return best_description


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


        # Layer 1: Groq Llama 4 Scout Vision for text extraction
        vision_start = time.time()
        vision_results = []  # Initialize to prevent UnboundLocalError if _extract_with_groq_vision fails
        try:
            vision_results = await self._extract_with_groq_vision(image_bytes, mime_type)
            raw_text = " ".join([r.text for r in vision_results])
            confidence_report = self._analyze_confidence(vision_results)

            metadata.groq_vision_time = time.time() - vision_start
            metadata.groq_vision_confidence = confidence_report.average_confidence
            metadata.layers_used.append("Groq Llama 4 Scout Vision")


        except Exception as e:
            # Fallback to Groq Llama 4 Maverick if Groq vision extraction fails completely
            logger.warning(f"Groq vision extraction failed: {str(e)}. Will use Groq Llama 4 Maverick fallback.")
            raw_text = ""
            confidence_report = ConfidenceReport(
                average_confidence=0.0,
                critical_fields_confidence={},
                low_confidence_regions=[],
                needs_groq_fallback=True,
                reason=f"Groq vision extraction failed: {e}",
            )
            metadata.groq_vision_time = time.time() - vision_start

        # Layer 2: Groq Llama 3.1 8B for structured extraction (switched back for latency comparison)
        groq_start = time.time()
        try:
            groq_data = await self._extract_with_groq_llama31(
                raw_text, confidence_report.average_confidence
            )
            metadata.groq_llama31_time = time.time() - groq_start
            metadata.layers_used.append("Groq Llama 3.1 8B")

        except Exception as e:
            # If Groq fails, create empty data and force Groq fallback
            logger.warning(f"Groq extraction failed: {str(e)}. Will use Groq Llama 4 Maverick fallback.")
            groq_data = ExtractedData()
            metadata.groq_llama31_time = time.time() - groq_start
            confidence_report.needs_groq_fallback = True
            confidence_report.reason = "Groq extraction failed"


        # Layer 3: Groq Llama 4 Maverick fallback (if needed)
        groq_fallback_data = None
        if confidence_report.needs_groq_fallback:
            logger.info(f"Triggering Groq Llama 4 Maverick fallback: {confidence_report.reason}")
            groq_fallback_start = time.time()
            try:
                groq_fallback_data = await self._extract_with_groq_fallback(
                    image_bytes, mime_type, raw_text
                )
                metadata.groq_fallback_time = time.time() - groq_fallback_start
                metadata.groq_fallback_used = True
                metadata.layers_used.append("Groq Llama 4 Maverick 17b 128e")
            except Exception as e:
                logger.error(f"Groq Llama 4 Maverick fallback also failed: {str(e)}")
                metadata.groq_fallback_time = time.time() - groq_fallback_start

        # Merge results
        final_data = self._merge_results(groq_data, groq_fallback_data)
        
        # Smart fallback: If product_description is still missing, try to extract from vision results
        if not final_data.product_description and vision_results:
            logger.info("Product description missing, attempting keyword extraction from vision results")
            extracted_keywords = self._extract_product_keywords(vision_results, final_data.brand_name)
            if extracted_keywords:
                final_data.product_description = extracted_keywords
                logger.info(f"Keyword extraction successful: '{extracted_keywords}'")

        metadata.total_time = time.time() - start_time

        logger.success(
            f"Hybrid OCR pipeline complete: "
            f"layers={', '.join(metadata.layers_used)}, "
            f"total_time={metadata.total_time:.2f}s, "
            f"groq_fallback_used={metadata.groq_fallback_used}"
        )

        return final_data, metadata

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return {"cache_size": len(_groq_fallback_cache), "cache_keys": list(_groq_fallback_cache.keys())}

    def clear_cache(self):
        """Clear the Groq fallback result cache"""
        _groq_fallback_cache.clear()


# Global service instance
_vision_service_instance: HybridVisionService | None = None


def get_vision_service() -> HybridVisionService:
    """Get or create the global vision service instance"""
    global _vision_service_instance
    if _vision_service_instance is None:
        _vision_service_instance = HybridVisionService()
    return _vision_service_instance
