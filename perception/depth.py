"""
Monocular Depth & Proximity Estimator
====================================
Estimates approximate real‑world distance (in meters) and proximity category
from 2D bounding boxes using geometric perspective, apparent object height,
and ground‑plane heuristics.
Conforms strictly to Team Interface Contract:
Returns: (distance_meters: float, proximity_category: str)
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import MODELS_DIR

logger = logging.getLogger("VisionAssist.Perception.Depth")

# Approximate known real‑world heights of objects in meters
REAL_WORLD_HEIGHTS: Dict[str, float] = {
    # Navigational Hazards
    "stairs": 1.50,
    "door": 2.05,
    "chair": 0.85,
    "table": 0.75,
    "dining table": 0.75,
    "couch": 0.85,
    "bed": 0.70,
    "bench": 0.50,
    "fire hydrant": 0.75,
    "stop sign": 2.10,
    "traffic light": 2.80,
    "pole": 3.00,
    # Vehicles & Pedestrians
    "person": 1.70,
    "car": 1.50,
    "bus": 3.20,
    "truck": 3.40,
    "motorcycle": 1.15,
    "bicycle": 1.05,
    "dog": 0.60,
    # Handheld / Everyday Items (suppressed by danger weights)
    "bottle": 0.25,
    "cup": 0.12,
    "laptop": 0.25,
    "cell phone": 0.15,
    "book": 0.22,
    "backpack": 0.45,
    "handbag": 0.30,
    "umbrella": 0.85,
}

DEFAULT_OBJECT_HEIGHT = 0.80  # Default height assumption in meters
DEFAULT_FOCAL_LENGTH_PIXELS = 600.0  # Standard webcam / mobile camera focal length


def format_distance_speech(distance_meters: float) -> str:
    """Formats metric distance into natural spoken phrasing.
    Examples:
      0.8m -> 'less than one metre'
      1.5m -> 'approximately 1.5 metres'
    """
    if distance_meters < 1.0:
        return "less than one metre"
    elif distance_meters <= 1.1:
        return "about one metre"
    elif distance_meters == int(distance_meters):
        return f"approximately {int(distance_meters)} metres away"
    else:
        return f"approximately {distance_meters:.1f} metres"


def estimate_distance_and_proximity(
    object_name: str,
    bbox: List[int],
    frame_height: int,
    frame_width: int,
    focal_length: float = DEFAULT_FOCAL_LENGTH_PIXELS,
) -> Tuple[float, str]:
    """Estimate metric distance in meters and categorical proximity.
    Args:
        object_name: Detected class label (e.g., 'chair', 'stairs')
        bbox: [x1, y1, x2, y2]
        frame_height: Image height in pixels
        frame_width: Image width in pixels
        focal_length: Calibrated camera focal length in pixels
    Returns:
        (distance_meters, proximity_category)
    """
    try:
        if not bbox or len(bbox) < 4 or frame_height <= 0 or frame_width <= 0:
            return (3.0, "Mid-range")

        x1, y1, x2, y2 = bbox[:4]
        box_height = max(1, y2 - y1)

        # 1. Height‑based pinhole projection estimate
        real_h = REAL_WORLD_HEIGHTS.get(object_name.lower(), DEFAULT_OBJECT_HEIGHT)
        distance_from_height = (real_h * focal_length) / float(box_height)

        # 2. Ground‑plane vertical position heuristic (objects lower in the frame are nearer)
        y2_norm = min(1.0, max(0.0, y2 / float(frame_height)))
        ground_dist_estimate = max(0.6, 4.0 * (1.0 - y2_norm) + 0.8)

        # 3. Blend the two estimates
        estimated_distance = 0.65 * distance_from_height + 0.35 * ground_dist_estimate
        estimated_distance = max(0.4, min(12.0, estimated_distance))
        estimated_distance = round(estimated_distance, 1)

        # 4. Categorise proximity
        if estimated_distance < 1.2:
            proximity = "Immediate"
        elif estimated_distance < 2.5:
            proximity = "Near"
        elif estimated_distance < 5.0:
            proximity = "Mid-range"
        else:
            proximity = "Far"

        return (estimated_distance, proximity)
    except Exception as e:
        logger.error(f"Sec.33 Error catch in depth estimation: {e}")
        return (3.0, "Mid-range")


class DepthEstimator:
    """Encapsulated depth estimation module.
    """

    def __init__(self, focal_length: float = DEFAULT_FOCAL_LENGTH_PIXELS):
        self.focal_length = focal_length

    def estimate(
        self,
        object_name: str,
        bbox: List[int],
        frame_height: int,
        frame_width: int,
    ) -> Tuple[float, str]:
        return estimate_distance_and_proximity(
            object_name=object_name,
            bbox=bbox,
            frame_height=frame_height,
            frame_width=frame_width,
            focal_length=self.focal_length,
        )
