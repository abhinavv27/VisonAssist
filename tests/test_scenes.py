"""
Test Scene Curations (Dipayan's QA & Demo Asset Generator)
==========================================================
Generates reproducible test scenes for the 6 judging demo scenarios:
1. Classroom with Chair & Door
2. Corridor with Stairs
3. High-Contrast Room Sign (ROOM 204 - COMPUTER SCIENCE LAB)
4. Rapidly Approaching Vehicle (Safety Collision Alert)
5. Multimodal Scene with Minor Distractor (Water Bottle)
"""

from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np

TEST_SCENES_DIR = Path(__file__).resolve().parent / "test_scenes"
TEST_SCENES_DIR.mkdir(exist_ok=True)


def create_classroom_scene() -> np.ndarray:
    """Scene 1: Classroom with door on left, desk, chair in path, and person."""
    img = np.full((480, 640, 3), 45, dtype=np.uint8)

    # Floor & Wall perspective
    cv2.line(img, (0, 360), (640, 360), (80, 80, 80), 2)
    cv2.fillPoly(img, [np.array([(0, 360), (640, 360), (640, 480), (0, 480)])], (35, 35, 35))

    # Door on left
    cv2.rectangle(img, (60, 100), (200, 360), (120, 80, 50), -1)
    cv2.rectangle(img, (60, 100), (200, 360), (200, 160, 100), 2)
    cv2.putText(img, "DOOR", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Chair in centre-left path (approx 1.5m away)
    cv2.rectangle(img, (260, 240), (380, 390), (40, 120, 40), -1)
    cv2.rectangle(img, (260, 240), (380, 390), (80, 220, 80), 2)
    cv2.putText(img, "CHAIR (1.5m)", (275, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Desk on far right with harmless bottle
    cv2.rectangle(img, (480, 280), (630, 420), (70, 70, 90), -1)
    cv2.rectangle(img, (530, 240), (560, 280), (180, 120, 30), -1)
    cv2.putText(img, "BOTTLE", (520, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return img


def create_corridor_stairs_scene() -> np.ndarray:
    """Scene 2: Corridor with stairs ahead (approx 2m away)."""
    img = np.full((480, 640, 3), 30, dtype=np.uint8)

    # Corridor perspective converging to center
    cv2.line(img, (0, 0), (280, 220), (90, 90, 90), 2)
    cv2.line(img, (640, 0), (360, 220), (90, 90, 90), 2)
    cv2.line(img, (0, 480), (260, 350), (90, 90, 90), 2)
    cv2.line(img, (640, 480), (380, 350), (90, 90, 90), 2)

    # Stairs ahead in center path
    for idx, y in enumerate(range(240, 360, 24)):
        cv2.rectangle(img, (240 - idx * 15, y), (400 + idx * 15, y + 20), (140, 140, 150), -1)
        cv2.rectangle(img, (240 - idx * 15, y), (400 + idx * 15, y + 20), (220, 220, 240), 1)

    cv2.putText(img, "STAIRS (2.0m)", (260, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    return img


def create_room_sign_scene() -> np.ndarray:
    """Scene 3: High-contrast room sign: ROOM 204 - COMPUTER SCIENCE LAB."""
    img = np.full((480, 640, 3), 235, dtype=np.uint8)

    # Wall texture border
    cv2.rectangle(img, (40, 80), (600, 400), (20, 20, 20), 4)
    cv2.rectangle(img, (50, 90), (590, 390), (255, 255, 255), -1)

    # Ashok emblem / institutional seal placeholder
    cv2.circle(img, (320, 150), 30, (180, 120, 30), 2)
    cv2.putText(img, "MUJ", (305, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 120, 30), 1)

    # Text signage
    cv2.putText(
        img, "ROOM 204", (170, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (15, 15, 15), 3, cv2.LINE_AA
    )
    cv2.putText(
        img, "COMPUTER SCIENCE LAB", (100, 310),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA
    )
    cv2.putText(
        img, "DEPARTMENT OF ARTIFICIAL INTELLIGENCE", (140, 360),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 90), 1, cv2.LINE_AA
    )
    return img


def create_critical_obstacle_scene() -> np.ndarray:
    """Scene 4: Obstacle directly ahead at < 1.0m (Safety collision alert)."""
    img = np.full((480, 640, 3), 30, dtype=np.uint8)

    # Large prominent obstacle right in front of user's field of view
    cv2.rectangle(img, (140, 180), (500, 480), (40, 40, 180), -1)
    cv2.rectangle(img, (140, 180), (500, 480), (0, 0, 255), 4)
    cv2.putText(
        img, "CRITICAL OBSTACLE (0.8m)", (170, 300),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        img, "IMMEDIATE COLLISION RISK", (185, 340),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA
    )
    return img


def generate_all_scenes() -> Dict[str, str]:
    """Generates and writes test scenes to disk for offline demo reliability."""
    scenes = {
        "classroom": (create_classroom_scene(), "classroom_chair.png"),
        "corridor": (create_corridor_stairs_scene(), "corridor_stairs.png"),
        "room_sign": (create_room_sign_scene(), "room_sign_lab.png"),
        "collision_alert": (create_critical_obstacle_scene(), "critical_obstacle.png")
    }

    written = {}
    for key, (frame, filename) in scenes.items():
        filepath = TEST_SCENES_DIR / filename
        cv2.imwrite(str(filepath), frame)
        written[key] = str(filepath)
    return written


if __name__ == "__main__":
    paths = generate_all_scenes()
    print("Test scenes generated successfully:")
    for name, p in paths.items():
        print(f"  {name:<16}: {p}")
