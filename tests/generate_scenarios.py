"""
Scenario Clip & Frame Generator (Phase 2 Calibration Set)
=========================================================
Generates 10 standardized scenario test clips/frames under tests/scenarios/
to calibrate danger weights, test OCR under glare/low-light, and benchmark pass rates:

1. stairs_ahead.png: Descending corridor stairs (Critical/High)
2. doorway_right.png: Open doorway on right (Medium)
3. chair_center.png: Office chair in path at 1.5m (Medium/Hazard)
4. sign_clean.png: Crisp room sign 'ROOM 204' (Read Mode)
5. sign_glare_lowlight.png: Sign with glare/low-light (OCR unclear fallback)
6. close_obstacle.png: Collision obstacle < 1.0m (Mode 5 Safety Alert)
7. bottle_distractor.png: Handheld water bottle on table (Low/Suppressed)
8. person_passing.png: Walking person on left at 2.5m (Medium)
9. car_approaching.png: Vehicle approaching at 3.5m (High)
10. clear_hallway.png: Clear unobstructed corridor (Path clear)
"""

import json
from pathlib import Path
import cv2
import numpy as np

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
SCENARIOS_DIR.mkdir(exist_ok=True)

MANIFEST = [
    {
        "id": "scenario_01",
        "name": "stairs_ahead.png",
        "description": "Corridor staircase descending directly in travel path at 2.0m",
        "expected_object": "stairs",
        "expected_priority": "HIGH",
        "expected_speech_substring": "stairs",
        "min_risk_score": 65.0
    },
    {
        "id": "scenario_02",
        "name": "doorway_right.png",
        "description": "Hallway with open office door on the right at 3.0m",
        "expected_object": "door",
        "expected_priority": "MEDIUM",
        "expected_speech_substring": "door",
        "min_risk_score": 35.0
    },
    {
        "id": "scenario_03",
        "name": "chair_center.png",
        "description": "Classroom chair blocking direct forward path at 1.5m",
        "expected_object": "chair",
        "expected_priority": "MEDIUM",
        "expected_speech_substring": "chair",
        "min_risk_score": 45.0
    },
    {
        "id": "scenario_04",
        "name": "sign_clean.png",
        "description": "Clear high-contrast room sign: ROOM 204 - CS LAB",
        "expected_object": "ocr",
        "expected_priority": "HIGH",
        "expected_speech_substring": "Room 204",
        "ocr_text": "Room 204 - Computer Science Lab"
    },
    {
        "id": "scenario_05",
        "name": "sign_glare_lowlight.png",
        "description": "Degraded low-contrast sign with simulated lighting glare",
        "expected_object": "ocr_unclear",
        "expected_priority": "LOW",
        "expected_speech_substring": "unclear",
        "ocr_text": ""
    },
    {
        "id": "scenario_06",
        "name": "close_obstacle.png",
        "description": "Sudden near-field obstacle directly ahead at 0.8m",
        "expected_object": "obstacle",
        "expected_priority": "HIGH",
        "expected_speech_substring": "Warning",
        "min_risk_score": 75.0
    },
    {
        "id": "scenario_07",
        "name": "bottle_distractor.png",
        "description": "Small handheld water bottle on side table (Non-hazard)",
        "expected_object": "bottle",
        "expected_priority": "LOW",
        "expected_speech_substring": "suppressed",
        "max_risk_score": 35.0
    },
    {
        "id": "scenario_08",
        "name": "person_passing.png",
        "description": "Pedestrian passing on the left at 2.5m",
        "expected_object": "person",
        "expected_priority": "MEDIUM",
        "expected_speech_substring": "person",
        "min_risk_score": 35.0
    },
    {
        "id": "scenario_09",
        "name": "car_approaching.png",
        "description": "Approaching vehicle directly ahead at 3.5m",
        "expected_object": "car",
        "expected_priority": "HIGH",
        "expected_speech_substring": "car",
        "min_risk_score": 65.0
    },
    {
        "id": "scenario_10",
        "name": "clear_hallway.png",
        "description": "Empty hallway corridor with no immediate obstacles",
        "expected_object": "none",
        "expected_priority": "LOW",
        "expected_speech_substring": "clear",
        "max_risk_score": 20.0
    }
]


def generate_all_scenes():
    """Renders 10 calibrated scenario images and writes manifest.json."""
    # 1. Stairs ahead
    img1 = np.full((480, 640, 3), 35, dtype=np.uint8)
    for i in range(8):
        y = 260 + i * 25
        cv2.line(img1, (160 - i * 15, y), (480 + i * 15, y), (200, 200, 200), 4)
    cv2.putText(img1, "STAIRS (DOWN)", (230, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(SCENARIOS_DIR / "stairs_ahead.png"), img1)

    # 2. Doorway right
    img2 = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(img2, (440, 100), (600, 440), (140, 120, 90), -1)
    cv2.rectangle(img2, (440, 100), (600, 440), (20, 20, 20), 4)
    cv2.putText(img2, "DOORWAY", (470, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    cv2.imwrite(str(SCENARIOS_DIR / "doorway_right.png"), img2)

    # 3. Chair center
    img3 = np.full((480, 640, 3), 45, dtype=np.uint8)
    cv2.rectangle(img3, (240, 220), (400, 420), (30, 80, 220), -1)
    cv2.rectangle(img3, (240, 220), (400, 420), (0, 0, 0), 3)
    cv2.putText(img3, "OFFICE CHAIR", (255, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(SCENARIOS_DIR / "chair_center.png"), img3)

    # 4. Sign clean
    img4 = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(img4, (120, 150), (520, 330), (255, 255, 255), -1)
    cv2.rectangle(img4, (120, 150), (520, 330), (15, 23, 42), 6)
    cv2.putText(img4, "ROOM 204", (190, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 23, 42), 4)
    cv2.putText(img4, "COMPUTER SCIENCE LAB", (140, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 58, 138), 2)
    cv2.imwrite(str(SCENARIOS_DIR / "sign_clean.png"), img4)

    # 5. Sign glare/lowlight (simulated heavy flare and underexposure)
    img5 = np.full((480, 640, 3), 25, dtype=np.uint8)
    cv2.rectangle(img5, (140, 160), (500, 320), (60, 60, 60), -1)
    # Bright glare washed-out flare
    cv2.circle(img5, (320, 240), 90, (230, 240, 255), -1)
    cv2.putText(img5, "R O M  ? 4", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
    cv2.imwrite(str(SCENARIOS_DIR / "sign_glare_lowlight.png"), img5)

    # 6. Close obstacle (< 1.0m)
    img6 = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.rectangle(img6, (80, 80), (560, 480), (0, 0, 220), -1)
    cv2.rectangle(img6, (80, 80), (560, 480), (0, 255, 255), 6)
    cv2.putText(img6, "COLLISION HAZARD (0.8m)", (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.imwrite(str(SCENARIOS_DIR / "close_obstacle.png"), img6)

    # 7. Bottle distractor
    img7 = np.full((480, 640, 3), 45, dtype=np.uint8)
    cv2.rectangle(img7, (480, 320), (530, 420), (220, 180, 50), -1)
    cv2.putText(img7, "BOTTLE", (475, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(SCENARIOS_DIR / "bottle_distractor.png"), img7)

    # 8. Person passing
    img8 = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(img8, (100, 140), (220, 420), (180, 100, 80), -1)
    cv2.circle(img8, (160, 110), 30, (200, 180, 160), -1)
    cv2.putText(img8, "PERSON", (125, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(SCENARIOS_DIR / "person_passing.png"), img8)

    # 9. Car approaching
    img9 = np.full((480, 640, 3), 35, dtype=np.uint8)
    cv2.rectangle(img9, (180, 220), (460, 380), (60, 60, 180), -1)
    cv2.circle(img9, (230, 380), 25, (20, 20, 20), -1)
    cv2.circle(img9, (410, 380), 25, (20, 20, 20), -1)
    cv2.putText(img9, "APPROACHING VEHICLE", (190, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(SCENARIOS_DIR / "car_approaching.png"), img9)

    # 10. Clear hallway
    img10 = np.full((480, 640, 3), 45, dtype=np.uint8)
    cv2.line(img10, (200, 200), (0, 480), (120, 120, 120), 2)
    cv2.line(img10, (440, 200), (640, 480), (120, 120, 120), 2)
    cv2.putText(img10, "CLEAR CORRIDOR", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
    cv2.imwrite(str(SCENARIOS_DIR / "clear_hallway.png"), img10)

    # Write JSON manifest
    manifest_path = SCENARIOS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, indent=2)

    print(f"Generated 10 scenarios in {SCENARIOS_DIR}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    generate_all_scenes()
