"""
Tests for Position Math, Depth Estimation, and Risk Scoring Engine.
"""

import pytest
from perception.position import classify_position, calculate_bounding_box_center, get_position_offset_description
from perception.depth import estimate_distance_and_proximity
from intelligence.risk_engine import RiskEngine
from intelligence.context_engine import ContextEngine
from config import ProductMode


def test_classify_position():
    frame_width = 640
    # Left zone (< 0.33 * 640 = 211.2)
    assert classify_position(100.0, frame_width) == "Left"
    # Centre zone (211.2 - 422.4)
    assert classify_position(320.0, frame_width) == "Centre"
    # Right zone (> 422.4)
    assert classify_position(550.0, frame_width) == "Right"


def test_calculate_bounding_box_center():
    bbox = [100, 200, 300, 400]
    xc, yc = calculate_bounding_box_center(bbox)
    assert xc == 200.0
    assert yc == 300.0


def test_risk_engine_stairs_vs_bottle():
    risk_engine = RiskEngine()
    
    # Stairs 2m away in centre: should score HIGH priority
    stairs_eval = risk_engine.calculate_risk("stairs", distance_meters=2.0, position="Centre")
    assert stairs_eval["priority"] == "HIGH"
    assert stairs_eval["risk_score"] >= 70.0

    # Bottle 2m away on right: should score LOW priority
    bottle_eval = risk_engine.calculate_risk("bottle", distance_meters=2.0, position="Right")
    assert bottle_eval["priority"] == "LOW"
    assert bottle_eval["risk_score"] < 40.0


def test_risk_engine_closer_increases_score():
    risk_engine = RiskEngine()
    score_far = risk_engine.calculate_risk("chair", distance_meters=4.0, position="Centre")["risk_score"]
    score_near = risk_engine.calculate_risk("chair", distance_meters=1.0, position="Centre")["risk_score"]
    assert score_near > score_far


def test_context_engine_aggregation():
    engine = ContextEngine()
    raw_detections = [
        {"object": "stairs", "confidence": 0.90, "bbox": [200, 100, 440, 400], "x_center": 320.0, "y_center": 250.0},
        {"object": "bottle", "confidence": 0.85, "bbox": [550, 300, 600, 420], "x_center": 575.0, "y_center": 360.0}
    ]
    context_items = engine.process_detections(raw_detections, frame_shape=(480, 640, 3), mode=ProductMode.OBSTACLE_AWARENESS)
    
    assert len(context_items) == 2
    # Highest risk should be sorted first (stairs before bottle)
    assert context_items[0]["object"] == "stairs"
    assert context_items[0]["position"] == "Centre"
    assert context_items[0]["priority"] == "HIGH"
    assert context_items[1]["object"] == "bottle"
