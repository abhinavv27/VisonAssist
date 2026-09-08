"""
Drawing & Annotation Utilities
==============================
Renders bounding boxes, risk tags, distance badges, and spoken audio banner
onto OpenCV frames for live visual monitoring.
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Risk priority BGR color mapping
PRIORITY_COLORS = {
    "HIGH": (0, 0, 230),     # Crisp Red
    "MEDIUM": (0, 165, 255), # Amber / Orange
    "LOW": (0, 200, 100),    # Emerald Green
}
DEFAULT_BOX_COLOR = (200, 200, 200)


def draw_visual_annotations(
    frame: np.ndarray,
    context_items: List[Dict[str, Any]],
    current_audio: Optional[str] = None,
    mode_label: str = "OBSTACLE AWARENESS",
    fps: Optional[float] = None
) -> np.ndarray:
    """
    Overlays detection boxes, risk tags, position guidelines, and live audio banner.
    """
    if frame is None:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Subtle spatial FOV divider lines (Left / Centre / Right)
    left_x = int(w * 0.33)
    right_x = int(w * 0.66)
    cv2.line(annotated, (left_x, 0), (left_x, h), (50, 50, 50), 1, cv2.LINE_AA)
    cv2.line(annotated, (right_x, 0), (right_x, h), (50, 50, 50), 1, cv2.LINE_AA)

    # 2. Draw object bounding boxes and risk badges
    for item in context_items:
        bbox = item.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        priority = item.get("priority", "LOW")
        color = PRIORITY_COLORS.get(priority, DEFAULT_BOX_COLOR)

        # Thick box for high priority, thinner for medium/low
        thickness = 3 if priority == "HIGH" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label tag: "STAIRS (2.1m) - HIGH"
        obj_name = item.get("object", "").upper()
        dist = item.get("distance", 0.0)
        risk_score = item.get("risk_score", 0.0)
        label = f"{obj_name} ({dist}m) [{priority}]"

        # Label background pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pill_y1 = max(0, y1 - th - 8)
        pill_y2 = y1
        cv2.rectangle(annotated, (x1, pill_y1), (x1 + tw + 10, pill_y2), color, -1)
        cv2.putText(
            annotated, label, (x1 + 5, pill_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    # 3. Top Mode / Status Bar
    cv2.rectangle(annotated, (0, 0), (w, 32), (20, 24, 30), -1)
    status_text = f"VISIONASSIST | MODE: {mode_label.upper()}"
    if fps:
        status_text += f" | {fps:.1f} FPS"
    cv2.putText(annotated, status_text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 240), 1, cv2.LINE_AA)

    # 4. Bottom Spoken Audio Banner (Matching UI Mockup)
    if current_audio and current_audio.strip():
        audio_bar_h = 42
        cv2.rectangle(annotated, (0, h - audio_bar_h), (w, h), (16, 80, 70), -1)  # Deep teal banner
        audio_msg = f'CURRENT AUDIO: "{current_audio}"'
        cv2.putText(
            annotated, audio_msg, (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 255, 245), 1, cv2.LINE_AA
        )

    return annotated
