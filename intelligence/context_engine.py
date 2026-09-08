"""
Context Engine
==============
Assembles rich contextual metadata for each detected object:
combines bounding box coordinates, spatial position classification,
monocular depth estimate, and calculated risk score.
"""

from typing import Any, Dict, List
import numpy as np

from perception.position import classify_position, get_position_offset_description
from perception.depth import estimate_distance_and_proximity
from intelligence.risk_engine import RiskEngine
from config import ProductMode


class ContextEngine:
    """
    Fuses perception layers into scored, contextualized entities.
    """

    def __init__(self, risk_engine: RiskEngine = None):
        self.risk_engine = risk_engine or RiskEngine()

    def process_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: tuple,
        mode: ProductMode = ProductMode.OBSTACLE_AWARENESS
    ) -> List[Dict[str, Any]]:
        """
        Transforms raw detections into fully scored context entities.
        
        Returns:
            List of dicts conforming to team contract:
            {
                "object": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],
                "x_center": float,
                "y_center": float,
                "position": str,               # 'Left' | 'Centre' | 'Right'
                "position_desc": str,          # 'slightly to your left', 'ahead'
                "distance": float,             # approximate meters
                "proximity": str,              # 'Immediate' | 'Near' | 'Mid-range' | 'Far'
                "risk_score": float,           # 0 - 100
                "priority": str,              # 'HIGH' | 'MEDIUM' | 'LOW'
                "mode": str
            }
        """
        h, w = frame_shape[:2]
        context_items: List[Dict[str, Any]] = []

        for det in detections:
            obj_name = det.get("object", "object")
            conf = det.get("confidence", 1.0)
            bbox = det.get("bbox", [0, 0, w, h])
            x_center = det.get("x_center", (bbox[0] + bbox[2]) / 2.0)
            y_center = det.get("y_center", (bbox[1] + bbox[3]) / 2.0)

            # 1. Spatial position
            pos = classify_position(x_center, w)
            pos_desc = get_position_offset_description(x_center, w)

            # 2. Distance and proximity
            dist_meters, prox_cat = estimate_distance_and_proximity(obj_name, bbox, h, w)

            # 3. Risk calculation
            risk_eval = self.risk_engine.calculate_risk(
                object_name=obj_name,
                distance_meters=dist_meters,
                position=pos,
                confidence=conf
            )

            context_items.append({
                "object": obj_name,
                "confidence": conf,
                "bbox": bbox,
                "x_center": x_center,
                "y_center": y_center,
                "position": pos,
                "position_desc": pos_desc,
                "distance": dist_meters,
                "proximity": prox_cat,
                "risk_score": risk_eval["risk_score"],
                "priority": risk_eval["priority"],
                "components": risk_eval["components"],
                "mode": mode.value if hasattr(mode, "value") else str(mode)
            })

        # Sort context items by risk score descending (highest threat first)
        context_items.sort(key=lambda x: x["risk_score"], reverse=True)
        return context_items
