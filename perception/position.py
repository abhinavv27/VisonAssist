"""
Spatial Position Classifier
==========================
Calculates normalized horizontal & vertical coordinates and classifies
 detections into 'Left', 'Centre', or 'Right' positions relative to the user's field of view.
"""

from typing import List, Tuple
from config import HORIZONTAL_LEFT_BOUNDARY, HORIZONTAL_RIGHT_BOUNDARY


def calculate_bounding_box_center(bbox: List[int]) -> Tuple[float, float]:
    """Calculate center point of a bounding box [x1, y1, x2, y2].

    Returns:
        (x_center, y_center): Raw pixel coordinates.
    """
    try:
        if not bbox or len(bbox) < 4:
            return (320.0, 240.0)
        x1, y1, x2, y2 = bbox[:4]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    except Exception:
        return (320.0, 240.0)


def classify_position(x_center: float, frame_width: int) -> str:
    """Classify horizontal position relative to user FOV.

    Args:
        x_center: Pixel x-coordinate of object center.
        frame_width: Total image width in pixels.

    Returns:
        'Left' | 'Centre' | 'Right'
    """
    try:
        if frame_width <= 0:
            return "Centre"
        norm_x = float(x_center) / float(frame_width)
        if norm_x < HORIZONTAL_LEFT_BOUNDARY:
            return "Left"
        elif norm_x > HORIZONTAL_RIGHT_BOUNDARY:
            return "Right"
        else:
            return "Centre"
    except Exception:
        return "Centre"


def get_position_offset_description(x_center: float, frame_width: int) -> str:
    """Return intuitive phrasing like 'ahead', 'slightly to your left', etc.
    """
    try:
        if frame_width <= 0:
            return "ahead"
        norm_x = float(x_center) / float(frame_width)
        if norm_x < 0.25:
            return "on your far left"
        elif norm_x < HORIZONTAL_LEFT_BOUNDARY:
            return "on your left"
        elif norm_x < 0.45:
            return "slightly to your left"
        elif norm_x <= 0.55:
            return "directly ahead"
        elif norm_x <= HORIZONTAL_RIGHT_BOUNDARY:
            return "slightly to your right"
        elif norm_x <= 0.75:
            return "on your right"
        else:
            return "on your far right"
    except Exception:
        return "ahead"


class PositionClassifier:
    """Class wrapper for position calculations."""

    @staticmethod
    def classify(x_center: float, frame_width: int) -> str:
        return classify_position(x_center, frame_width)

    @staticmethod
    def describe_offset(x_center: float, frame_width: int) -> str:
        return get_position_offset_description(x_center, frame_width)

    @staticmethod
    def get_center(bbox: List[int]) -> Tuple[float, float]:
        return calculate_bounding_box_center(bbox)
