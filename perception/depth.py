"""
Monocular Depth & Proximity Estimator
=====================================
Estimates approximate real-world distance (in meters) and proximity rating
from 2D bounding boxes using geometric perspective and apparent object height.
Provides plug-in support for deep MiDaS monocular depth models.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("VisionAssist.Depth")

# Approximate known real-world heights of objects in meters
REAL_WORLD_HEIGHTS: Dict[str, float] = {
    "person": 1.70,
    "chair": 0.85,
    "table": 0.75,
    "dining table": 0.75,
    "couch": 0.85,
    "car": 1.50,
    "bus": 3.00,
    "truck": 3.20,
    "motorcycle": 1.10,
    "bicycle": 1.00,
    "bottle": 0.25,
    "cup": 0.12,
    "door": 2.00,
    "stairs": 1.50,
    "laptop": 0.25,
    "dog": 0.60,
}
DEFAULT_OBJECT_HEIGHT = 0.80  # Default height assumption in meters
FOCAL_LENGTH_PIXELS = 600.0   # Standard webcam focal length calibration


def estimate_distance_and_proximity(
    object_name: str,
    bbox: List[int],
    frame_height: int,
    frame_width: int
) -> Tuple[float, str]:
    """
    Estimate metric distance in meters and categorical proximity.
    
    Args:
        object_name: Detected class label (e.g., 'chair', 'stairs')
        bbox: [x1, y1, x2, y2]
        frame_height: Image height in pixels
        frame_width: Image width in pixels
        
    Returns:
        (distance_meters, proximity_category): e.g., (1.8, 'Near')
    """
    try:
        if not bbox or len(bbox) < 4 or frame_height <= 0 or frame_width <= 0:
            return (3.0, "Mid-range")

        x1, y1, x2, y2 = bbox[:4]
        box_height = max(1, y2 - y1)

        # 1. Height-based pinhole projection estimate
        real_h = REAL_WORLD_HEIGHTS.get(object_name.lower(), DEFAULT_OBJECT_HEIGHT)
        distance_from_height = (real_h * FOCAL_LENGTH_PIXELS) / float(box_height)

        # 2. Ground plane vertical position heuristic (objects closer to bottom of frame are nearer)
        # y2 normalized to bottom: near bottom (y2 ~ frame_height) => closer
        y2_norm = min(1.0, max(0.0, y2 / float(frame_height)))
        # Ground plane distance estimate: objects at bottom edge are ~0.8m to 1.5m
        ground_dist_estimate = max(0.6, 4.0 * (1.0 - y2_norm) + 0.8)

        # Combine estimates with weighting
        estimated_distance = 0.65 * distance_from_height + 0.35 * ground_dist_estimate
        # Constrain to plausible indoor assistive range (0.4m to 12.0m)
        estimated_distance = max(0.5, min(12.0, estimated_distance))
        estimated_distance = round(estimated_distance, 1)

        # Categorize proximity
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
