"""
Unit tests for AskEngine (Mode 4 Free-form Visual Q&A)
======================================================
Verifies Phase 1 Task 1:
"Build ask_engine.py: frame + detection/OCR context -> local VLM prompt -> text answer.
Done when: Typed test question returns a grounded one-sentence answer."
"""

import numpy as np
from intelligence.ask_engine import AskEngine


def test_ask_engine_where_is_the_door():
    """Verify grounded location query returns correct position and distance."""
    engine = AskEngine()
    context_items = [
        {
            "object": "door",
            "confidence": 0.89,
            "position": "left",
            "position_desc": "to your left",
            "distance": 3.2,
            "priority": "MEDIUM",
        }
    ]
    answer = engine.ask("Where is the door?", context_items=context_items)
    assert "door" in answer.lower()
    assert "left" in answer.lower()
    assert "3.2" in answer or "3" in answer


def test_ask_engine_read_sign():
    """Verify reading query extracts OCR text and returns concise sentence."""
    engine = AskEngine()
    ocr_items = [
        {"text": "Room 204", "confidence": 0.95},
        {"text": "Computer Science Lab", "confidence": 0.92},
    ]
    answer = engine.ask("What does the sign say?", ocr_items=ocr_items)
    assert "Room 204" in answer
    assert "Computer Science Lab" in answer


def test_ask_engine_general_scene_summary():
    """Verify general 'What is ahead?' query returns grounded one-sentence summary."""
    engine = AskEngine()
    context_items = [
        {
            "object": "chair",
            "confidence": 0.85,
            "position": "centre",
            "position_desc": "ahead",
            "distance": 1.5,
            "priority": "MEDIUM",
        },
        {
            "object": "table",
            "confidence": 0.75,
            "position": "right",
            "position_desc": "to your right",
            "distance": 2.4,
            "priority": "LOW",
        },
    ]
    answer = engine.ask("What is in front of me?", context_items=context_items)
    assert "chair" in answer.lower()
    assert "table" in answer.lower()
    assert answer.endswith(".")


def test_ask_engine_clear_path():
    """Verify clearance query reports safe when no close obstacle exists."""
    engine = AskEngine()
    answer = engine.ask("Is the path clear?", context_items=[])
    assert "clear" in answer.lower()


def test_ask_engine_hindi_door_query():
    """Verify Hindi question or language='hi' yields natural Hindi location answer."""
    engine = AskEngine()
    context_items = [
        {
            "object": "door",
            "confidence": 0.90,
            "position": "centre",
            "position_desc": "ahead",
            "distance": 2.5,
            "priority": "MEDIUM",
        }
    ]
    # Test via language parameter
    ans_param = engine.ask("Where is the door?", context_items=context_items, language="hi")
    assert "दरवाजा" in ans_param
    assert "मीटर" in ans_param

    # Test via Devanagari script auto-detection
    ans_script = engine.ask("दरवाजा कहाँ है?", context_items=context_items)
    assert "दरवाजा" in ans_script
    assert "सामने" in ans_script or "मीटर" in ans_script


def test_ask_engine_hindi_sign_query():
    """Verify Hindi sign reading query outputs Hindi response."""
    engine = AskEngine()
    ocr_items = [{"text": "Room 204", "confidence": 0.96}]
    answer = engine.ask("बोर्ड पर क्या लिखा है?", ocr_items=ocr_items)
    assert "साइन बोर्ड पर लिखा है" in answer
    assert "Room 204" in answer


def test_ask_engine_hindi_clear_path():
    """Verify Hindi clearance query returns safe path indication in Hindi."""
    engine = AskEngine()
    answer = engine.ask("क्या रास्ता साफ है?", context_items=[])
    assert "रास्ता साफ है" in answer

