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
        force_refresh: bool = False,
        user_query: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Determines what item (if any) should be spoken right now.
        
        Returns:
            The selected context item dict, or None if suppressed / cooled down.
        """
        now = time.time()

        # Mode 4: ASK Mode (Visual Question Answering)
        if mode == ProductMode.ASK:
            query = (user_query or "what is in front of me").strip().lower()

            # Case A: OCR / Sign reading query
            if any(k in query for k in ["read", "sign", "text", "room", "label", "written"]):
                if ocr_items and len(ocr_items) > 0:
                    return {
                        "type": "ask_answer",
                        "text": f"The sign reads: {ocr_items[0].get('text', '')}.",
                        "priority": "HIGH"
                    }
                return {
                    "type": "ask_answer",
                    "text": "I do not see any readable text in this direction.",
                    "priority": "MEDIUM"
                }

            # Case B: Specific object location query (e.g. "Where is the door?", "Is there a chair?")
            for item in context_items:
                obj = item["object"].lower()
                if obj in query or (obj == "chair" and "seat" in query) or (obj == "door" and "doorway" in query):
                    pos_desc = item.get("position_desc", "ahead")
                    dist = item.get("distance", 2.0)
                    return {
                        "type": "ask_answer",
                        "text": f"The {obj} is {pos_desc}, approximately {dist} metres away.",
                        "priority": "HIGH"
                    }

            # Case C: General "What is in front of me?" query
            if not context_items:
                return {
                    "type": "ask_answer",
                    "text": "The path ahead appears clear. No major obstacles detected.",
                    "priority": "LOW"
                }
            primary = context_items[0]
            p_obj = primary["object"]
            p_pos = primary["position_desc"]
            if len(context_items) > 1:
                secondary = context_items[1]
                s_obj = secondary["object"]
                s_pos = secondary["position_desc"]
                return {
                    "type": "ask_answer",
                    "text": f"There is a {p_obj} {p_pos}, and a {s_obj} {s_pos}.",
                    "priority": "HIGH"
                }
            return {
                "type": "ask_answer",
                "text": f"There is a {p_obj} {p_pos}.",
                "priority": "HIGH"
            }

        # Mode 5: SAFETY ALERT Mode (High-urgency immediate collision warning)
        if mode == ProductMode.SAFETY_ALERT:
            if not context_items:
                return None
            candidate = context_items[0]
            dist = candidate["distance"]
            pos = candidate["position"]
            # Trigger alert if obstacle within 1.8m or high risk in center
            if dist <= 1.8 or candidate["priority"] == "HIGH" or pos == "Centre":
                return {
                    "type": "safety_alert",
                    "candidate": candidate,
                    "is_critical": True,
                    "priority": "HIGH"
                }
            return None

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

        # Mode 2: OBSTACLE AWARENESS
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
