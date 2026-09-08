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


class ObjectTrack:
    """Tracks an individual detected object across sequential video frames."""

    def __init__(self, track_id: int, obj_name: str, bbox: List[int], distance: float):
        self.track_id = track_id
        self.obj_name = obj_name
        self.bbox = bbox
        self.distance = distance
        self.prev_distance = distance
        self.delta_distance = 0.0
        self.frames_seen = 1
        self.is_moving = False
        self.last_seen_time = 0.0

    def update(self, bbox: List[int], distance: float):
        self.prev_distance = self.distance
        self.distance = distance
        self.delta_distance = self.distance - self.prev_distance
        self.bbox = bbox
        self.frames_seen += 1
        # If distance shrinks by > 0.15m over consecutive frames, object is closing in
        self.is_moving = self.delta_distance < -0.15


class ContextEngine:
    """
    Fuses perception layers into scored, contextualized entities.
    Tracks objects temporally across frames to compute movement velocity vectors.
    """

    def __init__(self, risk_engine: RiskEngine = None):
        self.risk_engine = risk_engine or RiskEngine()
        self._tracks: Dict[int, ObjectTrack] = {}
        self._next_track_id = 1

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

            # 3. Temporal Tracking & Movement Estimation
            is_moving = False
            matched_track = None
            for track in self._tracks.values():
                if track.obj_name == obj_name:
                    # Centroid distance match
                    prev_xc = (track.bbox[0] + track.bbox[2]) / 2.0
                    prev_yc = (track.bbox[1] + track.bbox[3]) / 2.0
                    if abs(x_center - prev_xc) < (w * 0.20) and abs(y_center - prev_yc) < (h * 0.20):
                        matched_track = track
                        break

            if matched_track:
                matched_track.update(bbox, dist_meters)
                is_moving = matched_track.is_moving
            else:
                new_track = ObjectTrack(self._next_track_id, obj_name, bbox, dist_meters)
                self._tracks[self._next_track_id] = new_track
                self._next_track_id += 1

            # 4. Risk calculation with motion factor
            risk_eval = self.risk_engine.calculate_risk(
                object_name=obj_name,
                distance_meters=dist_meters,
                position=pos,
                is_moving=is_moving,
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
                "is_moving": is_moving,
                "risk_score": risk_eval["risk_score"],
                "priority": risk_eval["priority"],
                "components": risk_eval["components"],
                "mode": mode.value if hasattr(mode, "value") else str(mode)
            })

        # Sort context items by risk score descending (highest threat first)
        context_items.sort(key=lambda x: x["risk_score"], reverse=True)
        return context_items
