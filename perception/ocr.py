"""
Optical Character Recognition (OCR) Reader
==========================================
Extracts text from camera frames (room signs, labels, doors).
Supports EasyOCR with fallback to mock / pattern extraction.
"""

import logging
from typing import Any, Dict, List
import cv2
import numpy as np

logger = logging.getLogger("VisionAssist.OCR")


class OCRReader:
    """
    OCR wrapper for assistive text reading.
    """

    def __init__(self, languages: List[str] = ["en"]):
        self.languages = languages
        self.reader: Any = None
        self._is_mock = False
        self._initialize_reader()

    def _initialize_reader(self) -> None:
        """Initialize EasyOCR or set fallback."""
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(self.languages, gpu=False)
            self._is_mock = False
            logger.info("EasyOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR not available ({e}). Using mock OCR reader.")
            self._is_mock = True

    def warmup(self) -> None:
        """Warm up the OCR model with a blank frame to prevent cold-start delay."""
        dummy = np.full((120, 320, 3), 200, dtype=np.uint8)
        self.read_text(dummy)
        logger.info("OCRReader warmed up and ready.")

    def read_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract visible text segments from frame with glare and low-light robustness.
        
        Returns:
            List of { "text": str, "confidence": float, "bbox": [x1, y1, x2, y2], "is_unclear": bool }
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.debug("Received empty or invalid frame for OCR.")
            return []

        try:
            # Analyze lighting and contrast for glare or heavy underexposure
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            mean_brightness = float(np.mean(gray))
            contrast_std = float(np.std(gray))

            # Low-light (< 40), localized glare hotspot on dark background, or washed-out overexposure
            saturated_ratio = float(np.mean(gray > 220))
            is_low_light = mean_brightness < 40.0
            is_glare_hotspot = (mean_brightness < 70.0 and saturated_ratio > 0.02)
            is_washed_out = (mean_brightness > 220.0 and contrast_std < 22.0)
            is_lighting_degraded = is_low_light or is_glare_hotspot or is_washed_out

            # Early graceful fallback under degraded lighting (glare / low-light)
            if is_lighting_degraded:
                logger.info("OCR scene degraded by glare or low-light; returning assistive retry fallback.")
                return [{
                    "text": "Text appears unclear, please adjust lighting or move closer.",
                    "confidence": 0.20,
                    "bbox": [0, 0, w, h],
                    "is_unclear": True
                }]

            if not self._is_mock and self.reader is not None:
                results = self.reader.readtext(frame)
                extracted = []
                valid_texts = []
                for bbox, text, conf in results:
                    clean_text = text.strip()
                    if conf > 0.25 and len(clean_text) > 0:
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        extracted.append({
                            "text": clean_text,
                            "confidence": round(float(conf), 2),
                            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                            "is_unclear": False
                        })
                        valid_texts.append(clean_text)

                if extracted:
                    # If multiple sign fragments/lines were detected, assemble full sign text
                    if len(extracted) > 1:
                        full_sign_text = " - ".join(valid_texts)
                        avg_conf = round(float(np.mean([item["confidence"] for item in extracted])), 2)
                        min_x = min(item["bbox"][0] for item in extracted)
                        min_y = min(item["bbox"][1] for item in extracted)
                        max_x = max(item["bbox"][2] for item in extracted)
                        max_y = max(item["bbox"][3] for item in extracted)
                        extracted.insert(0, {
                            "text": full_sign_text,
                            "confidence": avg_conf,
                            "bbox": [min_x, min_y, max_x, max_y],
                            "is_unclear": False
                        })
                    return extracted

                return [{
                    "text": "Text appears unclear, please adjust lighting or move closer.",
                    "confidence": 0.20,
                    "bbox": [0, 0, w, h],
                    "is_unclear": True
                }]

            # Clean mock response for developer scaffolding & headless tests
            return [{
                "text": "Room 204 - Computer Science Lab",
                "confidence": 0.95,
                "bbox": [int(w * 0.2), int(h * 0.3), int(w * 0.8), int(h * 0.6)],
                "is_unclear": False
            }]

        except Exception as e:
            logger.error(f"Defensive perception catch in OCRReader: {e}")
            return [{
                "text": "Text unclear.",
                "confidence": 0.0,
                "bbox": [0, 0, 100, 100],
                "is_unclear": True
            }]

