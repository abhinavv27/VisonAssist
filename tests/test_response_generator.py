"""
Tests for Natural Language Response Generation.
"""

import pytest
from intelligence.response_generator import ResponseGenerator


def test_response_for_stairs():
    generator = ResponseGenerator()
    item = {
        "type": "obstacle",
        "candidate": {
            "object": "stairs",
            "distance": 2.0,
            "position_desc": "ahead"
        },
        "is_critical": False
    }
    res = generator.generate(item)
    assert "Stairs ahead" in res["text"]
    assert "metres" in res["text"]


def test_response_for_critical_hazard():
    generator = ResponseGenerator()
    item = {
        "type": "obstacle",
        "candidate": {
            "object": "chair",
            "distance": 0.8,
            "position_desc": "directly ahead"
        },
        "is_critical": True
    }
    res = generator.generate(item)
    assert "Warning" in res["text"]
    assert "less than one metre" in res["text"]


def test_response_for_ocr():
    generator = ResponseGenerator()
    item = {
        "type": "ocr",
        "text": "Computer Science Lab, Room 204"
    }
    res = generator.generate(item)
    assert res["text"] == "Computer Science Lab, Room 204."


def test_response_for_quick_look():
    generator = ResponseGenerator()
    item = {
        "type": "quick_look",
        "primary": {"object": "doorway", "position_desc": "ahead"},
        "secondary": {"object": "person", "position_desc": "on your left"}
    }
    res = generator.generate(item)
    assert "There is a doorway ahead, and a person on your left." == res["text"]
