"""
Perception package for VisionAssist.
Performs object detection, text extraction (OCR), depth estimation, and spatial positioning.
"""

from .position import (
    classify_position,
    calculate_bounding_box_center,
    get_position_offset_description,
    PositionClassifier,
)
from .depth import (
    estimate_distance_and_proximity,
    format_distance_speech,
    DepthEstimator,
)
from .object_detection import ObjectDetector
from .ocr import OCRReader

__all__ = [
    "classify_position",
    "calculate_bounding_box_center",
    "get_position_offset_description",
    "PositionClassifier",
    "estimate_distance_and_proximity",
    "format_distance_speech",
    "DepthEstimator",
    "ObjectDetector",
    "OCRReader",
]

