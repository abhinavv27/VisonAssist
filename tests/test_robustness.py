"""
Robustness & Defensive Perception Test Suite (Phase 2 & Section 33 Checklist)
===========================================================================
Asserts that all perception and intelligence modules gracefully handle corrupted,
empty, None, or extreme input frames without crashing the application.
"""

import numpy as np
import pytest

from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from perception.depth import estimate_distance_and_proximity
from perception.position import classify_position, calculate_bounding_box_center, get_position_offset_description
from intelligence.context_engine import ContextEngine
from intelligence.risk_engine import RiskEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from config import ProductMode
from scripts.calibrate_pipeline import run_calibration


class TestDefensivePerception:
    """Tests Section 33 defensive perception requirements."""

    def test_object_detector_handles_none_and_empty_frames(self):
        detector = ObjectDetector()
        
        # Test None frame
        assert detector.detect(None) == []
        
        # Test zero-size array
        empty_arr = np.array([])
        assert detector.detect(empty_arr) == []

        # Test 1D invalid array
        arr_1d = np.zeros(100, dtype=np.uint8)
        assert detector.detect(arr_1d) == []

        # Test non-array input
        assert detector.detect("not_an_image") == []

    def test_ocr_handles_none_and_empty_frames(self):
        ocr = OCRReader()

        # Test None frame
        assert ocr.read_text(None) == []

        # Test empty array
        assert ocr.read_text(np.array([])) == []

        # Test non-array input
        assert ocr.read_text(12345) == []

    def test_ocr_glare_and_lowlight_graceful_fallback(self):
        ocr = OCRReader()

        # Severe pitch black frame (brightness < 40)
        pitch_black = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ocr.read_text(pitch_black)
        assert len(results) == 1
        assert results[0]["is_unclear"] is True
        assert "unclear" in results[0]["text"].lower()

        # Extreme glare hotspot frame
        glare_frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        glare_frame[200:300, 200:300] = 250
        results_glare = ocr.read_text(glare_frame)
        assert len(results_glare) == 1
        assert results_glare[0]["is_unclear"] is True

    def test_depth_estimation_handles_corrupt_inputs(self):
        # Empty bbox
        dist, prox = estimate_distance_and_proximity("chair", [], 480, 640)
        assert dist == 3.0
        assert prox == "Mid-range"

        # Incomplete bbox
        dist, prox = estimate_distance_and_proximity("chair", [10, 20], 480, 640)
        assert dist == 3.0
        assert prox == "Mid-range"

        # Zero or negative frame dimensions
        dist, prox = estimate_distance_and_proximity("stairs", [10, 10, 50, 50], 0, 0)
        assert dist == 3.0
        assert prox == "Mid-range"

        # Degenerate zero-height bbox
        dist, prox = estimate_distance_and_proximity("car", [100, 200, 300, 200], 480, 640)
        assert isinstance(dist, float)
        assert prox in ["Immediate", "Near", "Mid-range", "Far"]

    def test_position_classification_handles_corrupted_coordinates(self):
        # Frame width <= 0
        assert classify_position(320.0, 0) == "Centre"
        assert classify_position(320.0, -100) == "Centre"

        # Extreme coordinate values
        assert classify_position(-500.0, 640) == "Left"
        assert classify_position(5000.0, 640) == "Right"

        # Offset description bounds
        assert get_position_offset_description(100.0, 0) == "ahead"

        # Bounding box center calculations
        assert calculate_bounding_box_center([]) == (320.0, 240.0)
        assert calculate_bounding_box_center([10, 20]) == (320.0, 240.0)
        assert calculate_bounding_box_center(None) == (320.0, 240.0)


class TestPipelineRobustness:
    """Tests end-to-end pipeline robustness under edge cases."""

    def test_context_engine_empty_detections(self):
        engine = ContextEngine()
        items = engine.process_detections([], (480, 640, 3))
        assert items == []

    def test_priority_engine_empty_and_corrupted_context(self):
        engine = PriorityEngine()
        assert engine.select_top_item([], mode=ProductMode.OBSTACLE_AWARENESS) is None
        assert engine.select_top_item([], mode=ProductMode.SAFETY_ALERT) is None
        
        quick_look = engine.select_top_item([], mode=ProductMode.QUICK_LOOK)
        assert quick_look["type"] == "empty_scene"

    def test_response_generator_null_safety(self):
        gen = ResponseGenerator()
        assert gen.generate(None) == {"text": ""}
        assert gen.generate({}) == {"text": ""}

    def test_full_scenario_calibration_benchmark(self):
        """Phase 2 benchmark: All scenarios achieve >= 70% pass rate in CI environments."""
        success = run_calibration(min_pass_rate=0.7)
        assert success is True
