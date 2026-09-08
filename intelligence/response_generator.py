"""
Natural Language Response Generator
====================================
Transforms prioritized perception/risk results into concise, clear,
and grammatically natural spoken sentences for visually impaired users.
Conforms strictly to Team Interface Contract:
Returns: { "text": str }
"""

from typing import Any, Dict, Optional


class ResponseGenerator:
    """
    Template-based natural sentence generator.
    Keeps messages succinct to prevent cognitive overload.
    """

    def generate(self, prioritized_item: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """
        Produce a concise spoken response string.
        
        Returns:
            { "text": str }
        """
        if prioritized_item is None:
            return {"text": ""}

        item_type = prioritized_item.get("type")

        # 1. OCR Text Reading
        if item_type == "ocr":
            raw_text = prioritized_item.get("text", "")
            return {"text": f"{raw_text}."}

        # 2. Quick Look Multi-Object Scene Overview
        if item_type == "quick_look":
            primary = prioritized_item.get("primary")
            secondary = prioritized_item.get("secondary")
            if not primary:
                return {"text": "Path appears clear."}

            p_obj = primary["object"]
            p_pos = primary["position_desc"]

            if secondary:
                s_obj = secondary["object"]
                s_pos = secondary["position_desc"]
                return {"text": f"There is a {p_obj} {p_pos}, and a {s_obj} {s_pos}."}
            else:
                return {"text": f"There is a {p_obj} {p_pos}."}

        # 3. Obstacle / Safety Guidance
        if item_type == "obstacle":
            cand = prioritized_item.get("candidate", {})
            obj = cand.get("object", "obstacle")
            dist = cand.get("distance", 2.0)
            pos_desc = cand.get("position_desc", "ahead")
            is_critical = prioritized_item.get("is_critical", False)

            # High-urgency collision warnings
            if is_critical or dist <= 1.0:
                return {"text": f"Warning. {obj.capitalize()} {pos_desc}, less than one metre."}

            # Normal distance description
            if dist < 1.5:
                dist_str = f"approximately {dist} metres"
            elif dist < 3.0:
                dist_str = f"about {dist} metres"
            else:
                dist_str = f"{dist} metres"

            # Contextual phrasing
            if obj in ("stairs", "steps"):
                return {"text": f"Stairs {pos_desc}, {dist_str} away."}
            elif obj in ("car", "bus", "truck", "motorcycle"):
                return {"text": f"Caution, {obj} {pos_desc}, {dist_str}."}
            elif obj in ("door", "doorway"):
                return {"text": f"Door {pos_desc}."}
            else:
                return {"text": f"{obj.capitalize()} {pos_desc}, {dist_str}."}

        # 4. Custom/Fallback text
        custom_text = prioritized_item.get("text", "")
        return {"text": custom_text}
