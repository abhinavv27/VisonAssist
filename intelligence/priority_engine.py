"""
Priority Engine
===============
Filters out auditory noise and selects the single most critical, mode-relevant
observation to announce. Manages debounce cooldowns to avoid repetitive speech loops.
"""

import time
from typing import Any, Dict, List, Optional
from config import ProductMode, ANNOUNCEMENT_COOLDOWN_SECONDS, CRITICAL_SAFETY_DISTANCE_METERS


class PriorityEngine:
    """
    Selects the single most actionable item to announce to the user.
    """

    def __init__(self, cooldown_seconds: float = ANNOUNCEMENT_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self.last_spoken_object: Optional[str] = None
        self.last_spoken_timestamp: float = 0.0
        self.last_spoken_distance: Optional[float] = None

    def select_top_item(
        self,
        context_items: List[Dict[str, Any]],
        mode: ProductMode = ProductMode.OBSTACLE_AWARENESS,
        ocr_items: Optional[List[Dict[str, Any]]] = None,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Determines what item (if any) should be spoken right now.
        
        Returns:
            The selected context item dict, or None if suppressed / cooled down.
        """
        now = time.time()

        # Mode 3: READ Mode prioritizes OCR text directly
        if mode == ProductMode.READ:
            if ocr_items and len(ocr_items) > 0:
                top_text = ocr_items[0]
                text_content = top_text.get("text", "")
                if force_refresh or text_content != self.last_spoken_object or (now - self.last_spoken_timestamp) > self.cooldown_seconds:
                    self.last_spoken_object = text_content
                    self.last_spoken_timestamp = now
                    return {
                        "type": "ocr",
                        "text": text_content,
                        "confidence": top_text.get("confidence", 1.0),
                        "priority": "HIGH"
                    }
            return None

        # Mode 1: QUICK LOOK Mode returns the primary salient objects (ignoring cooldown)
        if mode == ProductMode.QUICK_LOOK:
            if not context_items:
                return {"type": "empty_scene", "text": "Path appears clear.", "priority": "LOW"}
            return {
                "type": "quick_look",
                "primary": context_items[0],
                "secondary": context_items[1] if len(context_items) > 1 else None,
                "priority": "HIGH"
            }

        # Modes 2 & 5: OBSTACLE AWARENESS & SAFETY ALERT
        if not context_items:
            return None

        candidate = context_items[0]
        obj_name = candidate["object"]
        dist = candidate["distance"]
        priority = candidate["priority"]
        risk_score = candidate["risk_score"]

        # Urgent override: if object is dangerously close, bypass standard cooldown
        is_critical_hazard = (dist <= CRITICAL_SAFETY_DISTANCE_METERS and risk_score >= 50.0)

        # Check debounce rules
        time_elapsed = now - self.last_spoken_timestamp
        is_same_object = (obj_name == self.last_spoken_object)
        distance_changed_significantly = (
            self.last_spoken_distance is not None and abs(dist - self.last_spoken_distance) > 0.8
        )

        if not force_refresh:
            if is_same_object and not is_critical_hazard:
                # Do not repeat the same object unless cooldown expired or distance closed dramatically
                if time_elapsed < self.cooldown_seconds and not distance_changed_significantly:
                    return None
            elif time_elapsed < (self.cooldown_seconds * 0.6) and not is_critical_hazard:
                # Minimum spacing between switching different lower-priority objects
                return None

        # Suppress low-threat background items during obstacle awareness mode
        if priority == "LOW" and not force_refresh:
            return None

        # Candidate approved for announcement
        self.last_spoken_object = obj_name
        self.last_spoken_timestamp = now
        self.last_spoken_distance = dist

        return {
            "type": "obstacle",
            "candidate": candidate,
            "is_critical": is_critical_hazard,
            "priority": priority
        }

    def reset_cooldown(self) -> None:
        """Reset announcement history to allow immediate speech."""
        self.last_spoken_object = None
        self.last_spoken_timestamp = 0.0
        self.last_spoken_distance = None
