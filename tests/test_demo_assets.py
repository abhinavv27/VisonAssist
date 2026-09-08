"""Unit tests for Dipayan's test scenes and demo assets."""

import os
from pathlib import Path
import cv2


def test_curated_test_scenes_exist_and_readable():
    """Verify that all 4 curated test scenes are present, readable, and have valid dimensions."""
    scene_dir = Path(__file__).resolve().parent / "test_scenes"
    expected_scenes = [
        "classroom_chair.png",
        "corridor_stairs.png",
        "room_sign_lab.png",
        "critical_obstacle.png",
    ]

    assert scene_dir.is_dir(), f"Test scenes directory missing: {scene_dir}"

    for scene_name in expected_scenes:
        img_path = scene_dir / scene_name
        assert img_path.exists(), f"Missing expected test scene: {scene_name}"

        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to decode image: {img_path}"
        assert img.shape[0] > 0 and img.shape[1] > 0, f"Invalid dimensions for {scene_name}: {img.shape}"
        assert img.shape[2] == 3, f"Expected 3 channels (BGR) for {scene_name}, got {img.shape}"


def test_demo_props_exist():
    """Verify that high-contrast printable OCR props exist."""
    prop_path = Path(__file__).resolve().parent.parent / "demo_props" / "room_sign.html"
    assert prop_path.exists(), f"Room sign HTML demo prop missing: {prop_path}"
    content = prop_path.read_text(encoding="utf-8")
    assert "ROOM 204" in content
    assert "Computer Science Lab" in content
