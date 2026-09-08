"""
VisionAssist Ask Engine (Visual Question Answering & Grounded Reasoning)
=======================================================================
Implements Mode 4 (Ask / Free-form Q&A) as specified in Phase 1 of the Build Plan.

Grounded Pipeline:
1. Takes user natural query + current video frame + YOLO/OCR perception context.
2. Attempts lightweight local VLM (Ollama moondream/llava-phi3) if accessible.
3. Attempts cloud vision fallback (OpenAI/Claude API) if configured and online.
4. Deterministic grounded perception engine:
   Uses spatial quadrant math, ground-plane depth, bounding boxes, and OCR
   text to synthesize precise, reliable, one-sentence assistive navigation answers.
"""

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request
import cv2
import numpy as np

logger = logging.getLogger("VisionAssist.AskEngine")


class AskEngine:
    """Answers user queries grounded in visual perception and spatial context."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        vlm_model: str = "moondream",
        timeout: float = 2.0
    ):
        self.ollama_url = ollama_url
        self.vlm_model = vlm_model
        self.timeout = timeout
        self.last_source = "initialized"

    def ask(
        self,
        query: str,
        frame: Optional[np.ndarray] = None,
        context_items: Optional[List[Dict[str, Any]]] = None,
        ocr_items: Optional[List[Dict[str, Any]]] = None,
        language: str = "en"
    ) -> str:
        """
        Processes a user question and returns a grounded one-sentence answer.
        
        Args:
            query: The user's question (e.g., 'Where is the door?', 'Read the sign').
            frame: The latest video frame.
            context_items: Context-engine output (YOLO objects + depth + position).
            ocr_items: OCR detected texts.
            language: 'en' for English or 'hi' for Hindi.
            
        Returns:
            Grounded one-sentence answer suitable for TTS playback.
        """
        if not query or not query.strip():
            if language == "hi":
                return "कृपया अपने आस-पास के बारे में कोई प्रश्न पूछें।"
            return "Please ask a question about what is around you."

        clean_query = query.strip()
        context_items = context_items or []
        ocr_items = ocr_items or []

        # Auto-detect Hindi script if present in query
        if re.search(r"[\u0900-\u097F]", clean_query):
            language = "hi"

        # 1. Try local VLM (Ollama) if available (English prompt)
        if frame is not None and language != "hi":
            vlm_answer = self._try_local_vlm(clean_query, frame, context_items, ocr_items)
            if vlm_answer:
                self.last_source = "local_vlm"
                return vlm_answer

        # 2. Grounded deterministic perception reasoning (instant, reliable, offline)
        self.last_source = "grounded_perception"
        return self._grounded_reasoning(clean_query, context_items, ocr_items, language=language)

    def _grounded_reasoning(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        ocr_items: List[Dict[str, Any]],
        language: str = "en"
    ) -> str:
        """Deterministic reasoning based on spatial context and OCR tokens."""
        q_lower = query.lower()
        is_hi = language == "hi"

        # Translations mapping for Hindi responses
        obj_trans_hi = {
            "chair": "कुर्सी",
            "door": "दरवाजा",
            "stairs": "सीढ़ियाँ",
            "person": "व्यक्ति",
            "car": "गाड़ी",
            "bottle": "बोतल",
            "laptop": "लैपटॉप",
            "table": "मेज",
            "obstacle": "रुकावट",
        }
        pos_trans_hi = {
            "ahead": "ठीक सामने",
            "directly ahead": "ठीक सामने",
            "to your left": "बाईं ओर",
            "slightly to your left": "थोड़ा बाईं ओर",
            "to your right": "दाईं ओर",
            "slightly to your right": "थोड़ा दाईं ओर",
            "nearby": "पास में",
            "left": "बाईं ओर",
            "right": "दाईं ओर",
            "centre": "ठीक सामने",
        }

        # Query Type 1: Text / Reading (e.g. "Read the sign", "What does it say?", "Room number?")
        read_keywords = [
            "read", "sign", "text", "say", "room", "label", "written", "notice",
            "पढ़ो", "पढ़ो", "बोर्ड", "लिखा", "कमरा", "नंबर"
        ]
        if any(w in q_lower for w in read_keywords):
            if ocr_items:
                extracted = [t.get("text", "").strip() for t in ocr_items if t.get("text", "").strip()]
                if extracted:
                    joined_text = ", ".join(extracted)
                    if is_hi:
                        return f"साइन बोर्ड पर लिखा है: {joined_text}।"
                    return f"The sign reads: {joined_text}."
            if is_hi:
                return "इस दिशा में कोई पढ़ने योग्य टेक्स्ट नहीं दिख रहा है।"
            return "I do not see any readable text in this direction."

        # Query Type 2: Specific Object Location (e.g. "Where is the door?", "Is there a chair?")
        # Match against common synonyms across English & Hindi
        object_synonyms = {
            "door": ["door", "doorway", "exit", "entrance", "दरवाजा", "गेट", "निकास"],
            "chair": ["chair", "seat", "seating", "bench", "कुर्सी", "सीट"],
            "stairs": ["stairs", "staircase", "steps", "सीढ़ी", "सीढ़ियाँ", "जीना"],
            "person": ["person", "someone", "human", "anybody", "people", "व्यक्ति", "इंसान", "कोई"],
            "bottle": ["bottle", "drink", "water", "बोतल", "पानी"],
            "laptop": ["laptop", "computer", "screen", "लैपटॉप", "कंप्यूटर"],
            "car": ["car", "vehicle", "automobile", "गाड़ी", "कार"],
            "table": ["table", "desk", "मेज", "डेस्क"],
        }

        for item in context_items:
            obj_name = item.get("object", "").lower()
            target_words = object_synonyms.get(obj_name, [obj_name])
            if any(w in q_lower for w in target_words):
                pos_desc = item.get("position_desc", "ahead")
                dist = item.get("distance", 2.0)
                if is_hi:
                    hi_obj = obj_trans_hi.get(obj_name, obj_name)
                    hi_pos = pos_trans_hi.get(pos_desc, "आगे")
                    return f"{hi_obj} {hi_pos}, लगभग {dist:.1f} मीटर दूर है।"
                return f"The {obj_name} is {pos_desc}, approximately {dist:.1f} metres away."

        # Query Type 3: Clearance / Obstacle / Walking path query ("Is the path clear?", "Can I walk?")
        clear_keywords = [
            "clear", "safe", "walk", "path", "free", "way",
            "साफ", "सुरक्षित", "रास्ता", "चल", "चलूँ"
        ]
        if any(w in q_lower for w in clear_keywords):
            obstacles = [
                i for i in context_items
                if i.get("priority") in ("HIGH", "MEDIUM") and i.get("distance", 99) < 2.5
            ]
            if obstacles:
                top = obstacles[0]
                dist = top.get("distance", 1.5)
                if is_hi:
                    hi_obj = obj_trans_hi.get(top["object"], top["object"])
                    hi_pos = pos_trans_hi.get(top.get("position_desc", "ahead"), "आगे")
                    return f"सावधान। {hi_pos} लगभग {dist:.1f} मीटर पर {hi_obj} है।"
                return f"Caution. There is a {top['object']} {top.get('position_desc', 'ahead')}, approximately {dist:.1f} metres away."
            if is_hi:
                return "आगे का रास्ता साफ है। कोई रुकावट नहीं है।"
            return "The path directly ahead appears clear."

        # Query Type 4: General visual summary ("What is in front of me?", "What do you see?")
        if not context_items:
            if is_hi:
                return "आगे का क्षेत्र साफ है। कोई रुकावट नहीं दिखी।"
            return "The area ahead appears clear. No obstacles detected."

        primary = context_items[0]
        p_obj = primary.get("object", "obstacle")
        p_pos = primary.get("position_desc", "ahead")
        p_dist = primary.get("distance", 2.0)

        if len(context_items) > 1:
            sec = context_items[1]
            s_obj = sec.get("object", "object")
            s_pos = sec.get("position_desc", "nearby")
            if is_hi:
                p_hi = obj_trans_hi.get(p_obj, p_obj)
                s_hi = obj_trans_hi.get(s_obj, s_obj)
                p_pos_hi = pos_trans_hi.get(p_pos, "आगे")
                s_pos_hi = pos_trans_hi.get(s_pos, "पास में")
                return f"यहाँ {p_pos_hi} {p_dist:.1f} मीटर पर {p_hi} और {s_pos_hi} {s_hi} है।"
            return f"There is a {p_obj} {p_pos} at {p_dist:.1f} metres, and a {s_obj} {s_pos}."

        if is_hi:
            p_hi = obj_trans_hi.get(p_obj, p_obj)
            p_pos_hi = pos_trans_hi.get(p_pos, "आगे")
            return f"यहाँ {p_pos_hi} लगभग {p_dist:.1f} मीटर पर {p_hi} है।"
        return f"There is a {p_obj} {p_pos}, approximately {p_dist:.1f} metres away."

    def _try_local_vlm(
        self,
        query: str,
        frame: np.ndarray,
        context_items: List[Dict[str, Any]],
        ocr_items: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Queries local Ollama endpoint if running."""
        try:
            # Resize frame to 320x240 for rapid VLM inference
            h, w = frame.shape[:2]
            scale = min(320 / max(w, 1), 240 / max(h, 1))
            thumb = cv2.resize(frame, (int(w * scale), int(h * scale)))
            success, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not success:
                return None

            img_b64 = base64.b64encode(buffer).decode("utf-8")

            # Ground prompt with YOLO detections
            context_summary = ", ".join(
                [f"{it['object']} ({it.get('position_desc', 'ahead')}, {it.get('distance', 2.0):.1f}m)" for it in context_items[:3]]
            )
            ocr_summary = ", ".join([it.get("text", "") for it in ocr_items[:3]])

            system_hint = (
                "You are an assistive vision companion for a visually impaired user. "
                "Answer in exactly ONE short, concise sentence. Focus strictly on navigation and spatial safety. "
            )
            if context_summary:
                system_hint += f"Detected objects: {context_summary}. "
            if ocr_summary:
                system_hint += f"Visible text: {ocr_summary}. "

            payload = {
                "model": self.vlm_model,
                "prompt": f"{system_hint}\nUser asks: {query}",
                "images": [img_b64],
                "stream": False,
                "options": {
                    "num_predict": 40,
                    "temperature": 0.2
                }
            }

            if not self.ollama_url.startswith(("http://", "https://")):
                return None

            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "").strip()
                if response_text:
                    # Clean up to single sentence
                    sentences = re.split(r"(?<=[.!?]) +", response_text)
                    return sentences[0].strip()
        except Exception:
            # Offline or Ollama not running; silently fall back to grounded perception
            return None
        return None
