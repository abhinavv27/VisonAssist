"""
Tests for Priority Engine (Filtering, Debounce, and Top Item Selection).
"""

import time
import pytest
from intelligence.priority_engine import PriorityEngine
from config import ProductMode


def test_priority_engine_selects_highest_risk():
    engine = PriorityEngine(cooldown_seconds=3.0)
    
    mock_context = [
        {"object": "stairs", "distance": 2.0, "priority": "HIGH", "risk_score": 82.0, "position_desc": "ahead"},
        {"object": "chair", "distance": 1.5, "priority": "MEDIUM", "risk_score": 52.0, "position_desc": "on your left"},
        {"object": "bottle", "distance": 1.0, "priority": "LOW", "risk_score": 28.0, "position_desc": "on your right"}
    ]
    
    top_item = engine.select_top_item(mock_context, mode=ProductMode.OBSTACLE_AWARENESS, force_refresh=True)
    assert top_item is not None
    assert top_item["candidate"]["object"] == "stairs"


def test_priority_engine_suppresses_low_priority_in_obstacle_mode():
    engine = PriorityEngine(cooldown_seconds=3.0)
    
    mock_context = [
        {"object": "bottle", "distance": 2.0, "priority": "LOW", "risk_score": 18.0, "position_desc": "on your right"}
    ]
    
    top_item = engine.select_top_item(mock_context, mode=ProductMode.OBSTACLE_AWARENESS, force_refresh=False)
    assert top_item is None  # Suppressed because it's low priority


def test_priority_engine_debounce_prevents_repetitive_announcements():
    engine = PriorityEngine(cooldown_seconds=3.0)
    
    mock_context = [
        {"object": "chair", "distance": 2.0, "priority": "MEDIUM", "risk_score": 55.0, "position_desc": "ahead"}
    ]
    
    # First time: announced
    first = engine.select_top_item(mock_context, mode=ProductMode.OBSTACLE_AWARENESS)
    assert first is not None

    # Immediate second frame: suppressed by debounce cooldown
    second = engine.select_top_item(mock_context, mode=ProductMode.OBSTACLE_AWARENESS)
    assert second is None

    # Forced refresh: allows announcement
    forced = engine.select_top_item(mock_context, mode=ProductMode.OBSTACLE_AWARENESS, force_refresh=True)
    assert forced is not None
