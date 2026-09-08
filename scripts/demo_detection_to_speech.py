"""
Detection-to-Speech End-to-End Demo Script
===========================================
Demonstrates the complete closed-loop vision-to-audio pipeline
for the 4 judging scenarios defined in MASTER PROJECT REPORT Section 19 & 31:

Scenario 1: Classroom Navigation (Chair ahead, bottle suppressed)
Scenario 2: Corridor Hazard (Stairs ahead, ~2 metres away)
Scenario 3: Sudden Obstacle Collision Alert (Interruption override)
Scenario 4: OCR Sign Reading (Mode 3: Room 204 Computer Science Lab)
"""

import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import argparse
from typing import List, Dict

def get_webcam_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Capture a frame from the default webcam and resize to target resolution."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Ensure a camera is connected.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame from webcam.")
    return cv2.resize(frame, (width, height))

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the ObjectDetector for live detection
from perception.object_detection import ObjectDetector
from config import ProductMode
from perception.depth import estimate_distance_and_proximity
from perception.position import classify_position, calculate_bounding_box_center, get_position_offset_description
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from audio.tts import TextToSpeechEngine
from audio.audio_stream_server import AudioStreamServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VisionAssist.Demo")


def create_scenario_frame(title: str, objects_to_draw: list, text_to_draw: str = None) -> np.ndarray:
    """Create a synthetic 640x480 test frame representing an indoor scene."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background gradient
    for y in range(480):
        frame[y, :] = (int(30 + y * 0.08), int(35 + y * 0.08), int(40 + y * 0.08))

    # Header
    cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)

    # Draw simulated objects
    for obj in objects_to_draw:
        x1, y1, x2, y2 = obj["bbox"]
        color = (0, 255, 120) if obj["object"] in ["bottle", "book"] else (0, 165, 255)
        if obj.get("is_hazard", False):
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{obj['object']} ({obj.get('confidence', 0.85):.2f})"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if text_to_draw:
        cv2.rectangle(frame, (140, 160), (520, 240), (255, 255, 255), -1)
        cv2.putText(frame, text_to_draw, (155, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    return frame

# ---------------------------------------------------------------------
# Helper to draw live detections on a frame
def draw_detections(frame: np.ndarray, detections: List[Dict]):
    """Overlay bounding boxes and labels onto the frame.
    The function mutates the supplied frame in-place.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['object']} ({det.get('confidence', 0):.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def run_demo(use_webcam: bool = False):
    print("=" * 75)
    print(" VISIONASSIST — DETECTION-TO-SPEECH END-TO-END DEMO ")
    print("=" * 75)

    # Initialize the object detector (may fall back to heuristic mock)
    detector = ObjectDetector()
    detector.warmup()

    # Start audio relay server for phone streaming
    audio_server = AudioStreamServer(port=8088)
    audio_server.start()

    # Initialize engines
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()
    tts = TextToSpeechEngine(mute=False)

    time.sleep(0.5)

    # =========================================================================
    # SCENARIO 1: Classroom Navigation (Chair ahead, Door right, Bottle suppressed)
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SCENARIO 1] Classroom Navigation")
    print(" Scene contains: Chair (center), Door (right), Person (left), Water Bottle (handheld)")
    print(" Expected: Water bottle is suppressed as low-danger noise; Chair is spoken.")
    print("-" * 75)

    priority_engine.reset_cooldown()

    detections_s1 = [
        {
            "object": "bottle",
            "confidence": 0.94,
            "x_center": 560.0,
            "y_center": 380.0,
            "bbox": [530, 320, 590, 440]
        },
        {
            "object": "chair",
            "confidence": 0.88,
            "x_center": 320.0,
            "y_center": 330.0,
            "bbox": [240, 220, 400, 440]
        },
        {
            "object": "door",
            "confidence": 0.82,
            "x_center": 520.0,
            "y_center": 220.0,
            "bbox": [460, 80, 580, 360]
        }
    ]

    if use_webcam:
        frame_s1 = get_webcam_frame()
        detections_s1 = detector.detect(frame_s1)
        # Show live preview with detections
        draw_detections(frame_s1, detections_s1)
        cv2.imshow("Webcam Demo", frame_s1)
        cv2.waitKey(1)
    else:
        frame_s1 = create_scenario_frame("Scenario 1: Classroom Navigation", detections_s1)
    context_items_s1 = context_engine.process_detections(detections_s1, frame_s1.shape, ProductMode.OBSTACLE_AWARENESS)

    print(f" Context Items Computed ({len(context_items_s1)} items):")
    for item in context_items_s1:
        base_danger = item.get("components", {}).get("base_danger", 0.0)
        print(f"   - {item['object'].upper():<8}: BaseDanger={base_danger:.1f}, Dist={item['distance']}m, "
              f"RiskScore={item['risk_score']:.1f}, Priority={item['priority']}")


    prioritized_s1 = priority_engine.select_top_item(context_items_s1, ProductMode.OBSTACLE_AWARENESS)
    if prioritized_s1:
        speech_text = response_gen.generate(prioritized_s1)["text"]
        print(f"\n ==> Priority Engine Selected: {prioritized_s1['candidate']['object']}")
        print(f" ==> Spoken Output: \"{speech_text}\"")
        tts.speak(speech_text, interrupt=prioritized_s1.get("is_critical", False))

    time.sleep(2.0)

    # =========================================================================
    # SCENARIO 2: Corridor Hazard (Stairs ahead)
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SCENARIO 2] Corridor Hazard (Stairs ahead)")
    print(" Scene contains: Person (left), Stairs (center ahead, 2.0m)")
    print(" Expected: High base danger of stairs triggers priority announcement.")
    print("-" * 75)

    priority_engine.reset_cooldown()

    detections_s2 = [
        {
            "object": "stairs",
            "confidence": 0.91,
            "x_center": 320.0,
            "y_center": 310.0,
            "bbox": [200, 180, 440, 440],
            "is_hazard": True
        },
        {
            "object": "person",
            "confidence": 0.78,
            "x_center": 140.0,
            "y_center": 250.0,
            "bbox": [80, 120, 200, 380]
        }
    ]

    if use_webcam:
        frame_s2 = get_webcam_frame()
        detections_s2 = detector.detect(frame_s2)
        draw_detections(frame_s2, detections_s2)
        cv2.imshow("Webcam Demo", frame_s2)
        cv2.waitKey(1)
    else:
        frame_s2 = create_scenario_frame("Scenario 2: Corridor Hazard", detections_s2)
    context_items_s2 = context_engine.process_detections(detections_s2, frame_s2.shape, ProductMode.OBSTACLE_AWARENESS)

    for item in context_items_s2:
        base_danger = item.get("components", {}).get("base_danger", 0.0)
        print(f"   - {item['object'].upper():<8}: BaseDanger={base_danger:.1f}, Dist={item['distance']}m, "
              f"RiskScore={item['risk_score']:.1f}, Priority={item['priority']}")


    prioritized_s2 = priority_engine.select_top_item(context_items_s2, ProductMode.OBSTACLE_AWARENESS)
    if prioritized_s2:
        speech_text = response_gen.generate(prioritized_s2)["text"]
        print(f"\n ==> Priority Engine Selected: {prioritized_s2['candidate']['object']}")
        print(f" ==> Spoken Output: \"{speech_text}\"")
        tts.speak(speech_text, interrupt=prioritized_s2.get("is_critical", False))

    time.sleep(2.0)

    # =========================================================================
    # SCENARIO 3: Sudden Obstacle Collision Alert (Interrupt Mode)
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SCENARIO 3] Sudden Collision Warning (Safety Alert)")
    print(" Scene contains: Obstacle directly ahead (< 1.0m, huge bbox)")
    print(" Expected: Immediate interruption override with collision warning.")
    print("-" * 75)

    priority_engine.reset_cooldown()

    # For scenario 3, we will use live detections when webcam is enabled.
    if use_webcam:
        frame_s3 = get_webcam_frame()
        detections_s3 = detector.detect(frame_s3)
        draw_detections(frame_s3, detections_s3)
        cv2.imshow("Webcam Demo", frame_s3)
        cv2.waitKey(1)
    else:
        detections_s3 = [
            {
                "object": "chair",
                "confidence": 0.95,
                "x_center": 320.0,
                "y_center": 360.0,
                "bbox": [100, 120, 540, 480],  # Very large bounding box
                "is_hazard": True
            }
        ]
        frame_s3 = create_scenario_frame("Scenario 3: Emergency Collision Alert", detections_s3)
    context_items_s3 = context_engine.process_detections(detections_s3, frame_s3.shape, ProductMode.OBSTACLE_AWARENESS)

    for item in context_items_s3:
        print(f"   - {item['object'].upper():<8}: Dist={item['distance']}m, RiskScore={item['risk_score']:.1f}, "
              f"Priority={item['priority']}")

    prioritized_s3 = priority_engine.select_top_item(context_items_s3, ProductMode.OBSTACLE_AWARENESS)
    if prioritized_s3:
        speech_text = response_gen.generate(prioritized_s3)["text"]
        print(f"\n ==> Priority Engine Selected: {prioritized_s3['candidate']['object']} (Critical: {prioritized_s3.get('is_critical')})")
        print(f" ==> Spoken Output: \"{speech_text}\"")
        tts.speak(speech_text, interrupt=prioritized_s3.get("is_critical", False))

    time.sleep(2.0)

    # =========================================================================
    # SCENARIO 4: OCR Sign Reading (Mode 3: Read)
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SCENARIO 4] Mode 3: Room Signage Reading (OCR)")
    print(" Scene contains: Sign reading 'ROOM 204 - COMPUTER SCIENCE LAB'")
    print(" Expected: OCR extracts text and speaks room identifier.")
    print("-" * 75)

    priority_engine.reset_cooldown()

    ocr_items_s4 = [
        {
            "text": "Computer Science Lab, Room 204",
            "confidence": 0.95,
            "bbox": [150, 160, 510, 240],
            "x_center": 330.0,
            "y_center": 200.0
        }
    ]

    # Create frame and select OCR result
    frame_s4 = get_webcam_frame() if use_webcam else create_scenario_frame("Scenario 4: Signage Reading", [], text_to_draw="ROOM 204 - CS LAB")
    if use_webcam:
        # Show preview (no detections to draw for OCR case)
        cv2.imshow("Webcam Demo", frame_s4)
        cv2.waitKey(1)
    prioritized_s4 = priority_engine.select_top_item([], ProductMode.READ, ocr_items=ocr_items_s4)
    if prioritized_s4:
        if prioritized_s4.get("type") == "ocr":
            # OCR result contains 'text' directly
            speech_text = response_gen.generate(prioritized_s4)["text"]
            print(f"\n ==> Priority Engine Selected OCR Text: \"{prioritized_s4['text']}\"")
            print(f" ==> Spoken Output: \"{speech_text}\"")
            tts.speak(speech_text)
        else:
            # Fallback for unexpected types
            speech_text = response_gen.generate(prioritized_s4)["text"]
            print(f"\n ==> Priority Engine Selected: {prioritized_s4.get('candidate', {}).get('object', 'unknown')}")
            print(f" ==> Spoken Output: \"{speech_text}\"")
            tts.speak(speech_text)

    time.sleep(2.5)

    tts.stop()
    audio_server.stop()
    cv2.destroyAllWindows()
    print("\n" + "=" * 75)
    print(" ALL 4 DEMO SCENARIOS COMPLETED SUCCESSFULLY! ")
    print("=" * 75)

# ---------------------------------------------------------------------
# Live video loop implementation
def run_live_demo():
    # OpenCV GUI functions are not available on this build; we rely on imshow to create the window automatically
    """Continuously capture webcam frames, run detection, display the video with bounding boxes,
    and speak the highest-priority detection when it changes.
    Press 'q' to quit the loop.
    """
    detector = ObjectDetector()
    detector.warmup()

    # Start audio relay server for phone streaming (optional but kept for parity)
    audio_server = AudioStreamServer(port=8088)
    audio_server.start()

    # Initialize engines
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()
    tts = TextToSpeechEngine(mute=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Ensure a camera is connected.")

    last_spoken: str | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detector.detect(frame)
        draw_detections(frame, detections)

        # Process detections through the same pipeline used in the demo scenarios
        context_items = context_engine.process_detections(
            detections, frame.shape, ProductMode.OBSTACLE_AWARENESS
        )
        prioritized = priority_engine.select_top_item(
            context_items, ProductMode.OBSTACLE_AWARENESS
        )
        if prioritized:
            obj = prioritized["candidate"]["object"]
            if obj != last_spoken:
                speech_text = response_gen.generate(prioritized)["text"]
                tts.speak(speech_text, interrupt=prioritized.get("is_critical", False))
                last_spoken = obj

        cv2.imshow("Live VisionAssist", frame)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    audio_server.stop()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VisionAssist demo with optional webcam input.")
    parser.add_argument("--webcam", action="store_true", help="Enable live webcam detection instead of synthetic scenarios.")
    parser.add_argument("--live-loop", action="store_true", help="Run a continuous live video loop with detections and TTS.")
    args = parser.parse_args()
    if args.live_loop:
        run_live_demo()
    else:
        run_demo(use_webcam=args.webcam)
