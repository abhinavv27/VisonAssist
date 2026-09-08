"""
Natural Language Response Generator
====================================
Transforms prioritized perception/risk results into concise, clear,
and grammatically natural spoken sentences for visually impaired users.
Supports bilingual output (English / Hindi) for hackathon demo versatility.
Conforms strictly to Team Interface Contract:
Returns: { "text": str }
"""

from typing import Any, Dict, Optional

# Localization dictionaries for Phase 4 Multilingual Demo
_HINDI_OBJECTS = {
    "chair": "कुर्सी",
    "door": "दरवाजा",
    "doorway": "दरवाजा",
    "stairs": "सीढ़ियाँ",
    "steps": "सीढ़ियाँ",
    "person": "व्यक्ति",
    "car": "गाड़ी",
    "bus": "बस",
    "truck": "ट्रक",
    "bottle": "बोतल",
    "obstacle": "रुकावट",
}

_HINDI_POSITIONS = {
    "directly ahead": "सामने",
    "ahead": "सामने",
    "slightly to your left": "आपकी थोड़ी बाईं ओर",
    "slightly to your right": "आपकी थोड़ी दाईं ओर",
    "on your left": "आपकी बाईं ओर",
    "on your right": "आपकी दाईं ओर",
    "on your far left": "आपकी दूर बाईं ओर",
    "on your far right": "आपकी दूर दाईं ओर",
}


def _translate_to_hindi(english_text: str, cand: dict) -> str:
    """Helper to convert assistive phrases to fluent Hindi."""
    obj_en = cand.get("object", "obstacle").lower()
    pos_en = cand.get("position_desc", "ahead").lower()
    dist = cand.get("distance", 2.0)

    obj_hi = _HINDI_OBJECTS.get(obj_en, obj_en)
    pos_hi = _HINDI_POSITIONS.get(pos_en, "सामने")

    if "warning" in english_text.lower() or dist <= 1.0:
        return f"सावधान। {obj_hi} {pos_hi}, एक मीटर से कम दूरी पर है।"
    if obj_en in ("stairs", "steps"):
        return f"सीढ़ियाँ {pos_hi}, लगभग {dist} मीटर दूर हैं।"
    if obj_en in ("door", "doorway"):
        return f"दरवाजा {pos_hi} है।"
    return f"{obj_hi} {pos_hi}, लगभग {dist} मीटर दूर है।"


class ResponseGenerator:
    """
    Template-based natural sentence generator.
    Keeps messages succinct to prevent cognitive overload.
    """

    def generate(
        self,
        prioritized_item: Optional[Dict[str, Any]],
        language: str = "en"
    ) -> Dict[str, str]:
        """
        Produce a concise spoken response string in English or Hindi.

        Returns:
            { "text": str }
        """
        if prioritized_item is None:
            return {"text": ""}

        item_type = prioritized_item.get("type")

        # 1. Ask Mode Direct Answers
        if item_type == "ask_answer":
            ans = prioritized_item.get("text", "")
            return {"text": ans}

        # 2. Safety Alert Emergency Warnings
        if item_type == "safety_alert":
            cand = prioritized_item.get("candidate", {})
            obj = cand.get("object", "obstacle")
            pos_desc = cand.get("position_desc", "ahead")
            dist = cand.get("distance", 1.0)
            if language == "hi":
                return {
                    "text": _translate_to_hindi(
                        "Warning. Obstacle ahead.",
                        cand
                    )
                }
            if dist <= 1.0:
                msg = (
                    f"Warning. {obj.capitalize()} {pos_desc}, "
                    "less than one metre."
                )
                return {"text": msg}
            return {"text": f"Warning. {obj.capitalize()} {pos_desc}."}

        # 3. OCR Text Reading
        if item_type == "ocr":
            if prioritized_item.get("is_unclear"):
                if language == "hi":
                    msg_hi = "टेक्स्ट स्पष्ट नहीं है, कृपया थोड़ा पास आएँ।"
                    return {"text": msg_hi}
                msg = (
                    "Text appears unclear, please adjust lighting "
                    "or move closer."
                )
                return {"text": msg}
            raw_text = prioritized_item.get("text", "")
            return {"text": f"{raw_text}."}

        # 4. Quick Look Multi-Object Scene Overview
        if item_type == "quick_look":
            primary = prioritized_item.get("primary")
            secondary = prioritized_item.get("secondary")
            if not primary:
                if language == "hi":
                    return {"text": "रास्ता साफ़ है।"}
                return {"text": "Path appears clear."}

            p_obj = primary["object"]
            p_pos = primary["position_desc"]

            if language == "hi":
                p_hi = _HINDI_OBJECTS.get(p_obj.lower(), p_obj)
                p_pos_hi = _HINDI_POSITIONS.get(p_pos.lower(), "सामने")
                if secondary:
                    s_obj = secondary["object"]
                    s_pos = secondary["position_desc"]
                    s_hi = _HINDI_OBJECTS.get(s_obj.lower(), s_obj)
                    s_pos_hi = _HINDI_POSITIONS.get(s_pos.lower(), "सामने")
                    return {
                        "text": f"आगे {p_hi} {p_pos_hi} है, "
                                f"और {s_hi} {s_pos_hi} है।"
                    }
                return {"text": f"आगे {p_hi} {p_pos_hi} है।"}

            if secondary:
                s_obj = secondary["object"]
                s_pos = secondary["position_desc"]
                return {
                    "text": f"There is a {p_obj} {p_pos}, "
                            f"and a {s_obj} {s_pos}."
                }
            return {"text": f"There is a {p_obj} {p_pos}."}

        # 5. Obstacle / Navigational Guidance
        if item_type == "obstacle":
            cand = prioritized_item.get("candidate", {})
            obj = cand.get("object", "obstacle")
            dist = cand.get("distance", 2.0)
            pos_desc = cand.get("position_desc", "ahead")
            is_critical = prioritized_item.get("is_critical", False)

            if language == "hi":
                prefix = "Warning" if (is_critical or dist <= 1.0) else ""
                return {"text": _translate_to_hindi(prefix, cand)}

            # High-urgency collision warnings
            if is_critical or dist <= 1.0:
                msg = (
                    f"Warning. {obj.capitalize()} {pos_desc}, "
                    "less than one metre."
                )
                return {"text": msg}

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
            if obj in ("car", "bus", "truck", "motorcycle"):
                return {"text": f"Caution, {obj} {pos_desc}, {dist_str}."}
            if obj in ("door", "doorway"):
                return {"text": f"Door {pos_desc}."}
            return {"text": f"{obj.capitalize()} {pos_desc}, {dist_str}."}

        # 6. Custom/Fallback text
        custom_text = prioritized_item.get("text", "")
        return {"text": custom_text}
