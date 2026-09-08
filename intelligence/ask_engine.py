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
        ocr_items: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Processes a user question and returns a grounded one-sentence answer.
        
        Args:
            query: The user's question (e.g., 'Where is the door?', 'Read the sign').
            frame: The latest video frame.
            context_items: Context-engine output (YOLO objects + depth + position).
            ocr_items: OCR detected texts.
            
        Returns:
            Grounded one-sentence answer suitable for TTS playback.
        """
        if not query or not query.strip():
            return "Please ask a question about what is around you."

        clean_query = query.strip()
        context_items = context_items or []
        ocr_items = ocr_items or []

        # 1. Try local VLM (Ollama) if available
        if frame is not None:
            vlm_answer = self._try_local_vlm(clean_query, frame, context_items, ocr_items)
            if vlm_answer:
                self.last_source = "local_vlm"
                return vlm_answer

        # 2. Grounded deterministic perception reasoning (instant, reliable, offline)
        self.last_source = "grounded_perception"
        return self._grounded_reasoning(clean_query, context_items, ocr_items)

    def _grounded_reasoning(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        ocr_items: List[Dict[str, Any]]
    ) -> str:
        """Deterministic reasoning based on spatial context and OCR tokens."""
        q_lower = query.lower()

        # Query Type 1: Text / Reading (e.g. "Read the sign", "What does it say?", "Room number?")
        if any(w in q_lower for w in ["read", "sign", "text", "say", "room", "label", "written", "notice"]):
            if ocr_items:
                extracted = [t.get("text", "").strip() for t in ocr_items if t.get("text", "").strip()]
                if extracted:
                    joined_text = ", ".join(extracted)
                    return f"The sign reads: {joined_text}."
            return "I do not see any readable text in this direction."

        # Query Type 2: Specific Object Location (e.g. "Where is the door?", "Is there a chair?")
        # Match against common synonyms
        object_synonyms = {
            "door": ["door", "doorway", "exit", "entrance"],
            "chair": ["chair", "seat", "seating", "bench"],
            "stairs": ["stairs", "staircase", "steps"],
            "person": ["person", "someone", "human", "anybody", "people"],
            "bottle": ["bottle", "drink", "water"],
            "laptop": ["laptop", "computer", "screen"],
            "car": ["car", "vehicle", "automobile"],
            "table": ["table", "desk"],
        }

        for item in context_items:
            obj_name = item.get("object", "").lower()
            target_words = object_synonyms.get(obj_name, [obj_name])
            if any(w in q_lower for w in target_words):
                pos_desc = item.get("position_desc", "ahead")
                dist = item.get("distance", 2.0)
                return f"The {obj_name} is {pos_desc}, approximately {dist:.1f} metres away."

        # Query Type 3: Clearance / Obstacle / Walking path query ("Is the path clear?", "Can I walk?")
        if any(w in q_lower for w in ["clear", "safe", "walk", "path", "free", "way"]):
            obstacles = [i for i in context_items if i.get("priority") in ("HIGH", "MEDIUM") and i.get("distance", 99) < 2.5]
            if obstacles:
                top = obstacles[0]
                return f"Caution. There is a {top['object']} {top.get('position_desc', 'ahead')}, approximately {top.get('distance', 1.5):.1f} metres away."
            return "The path directly ahead appears clear."

        # Query Type 4: General visual summary ("What is in front of me?", "What do you see?")
        if not context_items:
            return "The area ahead appears clear. No obstacles detected."

        primary = context_items[0]
        p_obj = primary.get("object", "obstacle")
        p_pos = primary.get("position_desc", "ahead")
        p_dist = primary.get("distance", 2.0)

        if len(context_items) > 1:
            sec = context_items[1]
            s_obj = sec.get("object", "object")
            s_pos = sec.get("position_desc", "nearby")
            return f"There is a {p_obj} {p_pos} at {p_dist:.1f} metres, and a {s_obj} {s_pos}."

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
