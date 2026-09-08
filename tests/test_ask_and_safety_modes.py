"""
Tests for Ask Mode (Visual Q&A), Safety Alert Mode, and Temporal Motion Tracking.
"""

import pytest
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from config import ProductMode


def test_ask_mode_where_is_the_door():
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()

    mock_context = [
        {"object": "door", "position": "Right", "position_desc": "on your right", "distance": 2.8, "priority": "MEDIUM", "risk_score": 42.0},
        {"object": "chair", "position": "Centre", "position_desc": "ahead", "distance": 1.5, "priority": "MEDIUM", "risk_score": 55.0}
    ]

    # Query: "Where is the door?"
    item = priority_engine.select_top_item(
        context_items=mock_context,
        mode=ProductMode.ASK,
        user_query="Where is the door?"
    )
    assert item is not None
    assert item["type"] == "ask_answer"
    
    response = response_gen.generate(item)
    assert "door is on your right" in response["text"]
    assert "2.8 metres" in response["text"]


def test_ask_mode_read_sign():
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()

    mock_ocr = [
        {"text": "Room 204 Computer Science Lab", "confidence": 0.95}
    ]

    item = priority_engine.select_top_item(
        context_items=[],
        mode=ProductMode.ASK,
        ocr_items=mock_ocr,
        user_query="Read the sign in front of me"
    )
    assert item is not None
    response = response_gen.generate(item)
    assert "Room 204 Computer Science Lab" in response["text"]


def test_safety_alert_triggers_on_close_obstacle():
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()

    mock_context = [
        {"object": "chair", "position": "Centre", "position_desc": "directly ahead", "distance": 0.8, "priority": "HIGH", "risk_score": 75.0}
    ]

    item = priority_engine.select_top_item(
        context_items=mock_context,
        mode=ProductMode.SAFETY_ALERT
    )
    assert item is not None
    assert item["type"] == "safety_alert"
    assert item["is_critical"] is True

    response = response_gen.generate(item)
    assert "Warning" in response["text"]
    assert "Chair" in response["text"]


def test_temporal_tracking_detects_approaching_obstacle():
    engine = ContextEngine()

    frame_shape = (480, 640, 3)
    # Frame 1: Car at 6.0 meters
    det_f1 = [{"object": "car", "confidence": 0.95, "bbox": [200, 200, 300, 260], "x_center": 250.0, "y_center": 230.0}]
    items_f1 = engine.process_detections(det_f1, frame_shape)
    assert len(items_f1) == 1
    assert items_f1[0]["is_moving"] is False

    # Frame 2: Car approaches rapidly (bounding box expands, distance closes)
    det_f2 = [{"object": "car", "confidence": 0.95, "bbox": [150, 160, 380, 320], "x_center": 265.0, "y_center": 240.0}]
    items_f2 = engine.process_detections(det_f2, frame_shape)
    assert len(items_f2) == 1
    # Should detect movement / approaching threat
    assert items_f2[0]["is_moving"] is True
    # Movement factor increases risk score
    assert items_f2[0]["risk_score"] > items_f1[0]["risk_score"]
