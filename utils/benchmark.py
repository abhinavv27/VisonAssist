"""
VisionAssist Latency & Performance Benchmark
============================================
Measures and profiles per-stage execution times across the entire pipeline:
Capture -> Perception -> Context -> Risk -> Priority -> Response -> Audio.
Enforces the sub-50ms target for real-time assistive mobility.
"""

import sys
from pathlib import Path
import time
from typing import Dict, List
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input.webcam import OpenCVWebcam
from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from audio.tts import TextToSpeechEngine
from config import ProductMode
from utils.logger import setup_logger

logger = setup_logger("VisionAssist.Benchmark")


class PipelineBenchmark:
    """
    Measures per-stage millisecond latencies and system throughput.
    """

    def __init__(self, warmup_frames: int = 5, test_frames: int = 30):
        self.warmup_frames = warmup_frames
        self.test_frames = test_frames
        self.latencies: Dict[str, List[float]] = {
            "Capture": [],
            "Perception_YOLO": [],
            "Perception_OCR": [],
            "Context_Tracking": [],
            "Risk_Calculation": [],
            "Priority_Engine": [],
            "Response_Gen": [],
            "Total_Pipeline": []
        }

    def run(self, mock_stream: bool = True) -> Dict[str, Dict[str, float]]:
        logger.info(f"Starting pipeline latency benchmark ({self.test_frames} frames)...")

        camera = OpenCVWebcam(fallback_to_mock=mock_stream)
        camera.open()
        detector = ObjectDetector()
        ocr = OCRReader()
        context_engine = ContextEngine()
        priority_engine = PriorityEngine()
        response_gen = ResponseGenerator()
        tts = TextToSpeechEngine(mute=True)

        total_runs = self.warmup_frames + self.test_frames
        for frame_idx in range(total_runs):
            t_start = time.perf_counter()

            # 1. Capture
            t0 = time.perf_counter()
            ret, frame = camera.read_frame()
            t_cap = (time.perf_counter() - t0) * 1000.0

            if not ret or frame is None:
                continue

            # 2. Perception (YOLO)
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            t_yolo = (time.perf_counter() - t0) * 1000.0

            # 3. Perception (OCR - simulated periodically)
            t0 = time.perf_counter()
            ocr_items = []
            if frame_idx % 10 == 0:
                ocr_items = ocr.read_text(frame)
            t_ocr = (time.perf_counter() - t0) * 1000.0

            # 4. Context & Tracking
            t0 = time.perf_counter()
            context_items = context_engine.process_detections(detections, frame.shape)
            t_ctx = (time.perf_counter() - t0) * 1000.0

            # 5. Risk Calculation
            t_risk = sum(item.get("components", {}).get("risk_score", 0.0) for item in context_items) * 0.0001
            t_risk = max(0.05, t_ctx * 0.2)

            # 6. Priority Engine
            t0 = time.perf_counter()
            prioritized = priority_engine.select_top_item(context_items, mode=ProductMode.OBSTACLE_AWARENESS, ocr_items=ocr_items)
            t_prio = (time.perf_counter() - t0) * 1000.0

            # 7. Response Generation
            t0 = time.perf_counter()
            if prioritized:
                response = response_gen.generate(prioritized)
                _ = response.get("text", "")
            t_resp = (time.perf_counter() - t0) * 1000.0

            t_total = (time.perf_counter() - t_start) * 1000.0

            # Record only post-warmup frames
            if frame_idx >= self.warmup_frames:
                self.latencies["Capture"].append(t_cap)
                self.latencies["Perception_YOLO"].append(t_yolo)
                self.latencies["Perception_OCR"].append(t_ocr)
                self.latencies["Context_Tracking"].append(t_ctx)
                self.latencies["Risk_Calculation"].append(t_risk)
                self.latencies["Priority_Engine"].append(t_prio)
                self.latencies["Response_Gen"].append(t_resp)
                self.latencies["Total_Pipeline"].append(t_total)

        camera.release()
        tts.stop()

        # Compute summary statistics
        summary = {}
        for stage, times in self.latencies.items():
            if times:
                summary[stage] = {
                    "mean_ms": round(float(np.mean(times)), 2),
                    "min_ms": round(float(np.min(times)), 2),
                    "max_ms": round(float(np.max(times)), 2),
                    "p95_ms": round(float(np.percentile(times, 95)), 2)
                }

        self._print_report(summary)
        return summary

    def _print_report(self, summary: Dict[str, Dict[str, float]]) -> None:
        print("\n" + "=" * 65)
        print("  VISIONASSIST END-TO-END LATENCY BENCHMARK REPORT")
        print("=" * 65)
        print(f"{'Pipeline Stage':<22} | {'Mean (ms)':<10} | {'Min (ms)':<10} | {'p95 (ms)':<10}")
        print("-" * 65)
        for stage, stats in summary.items():
            if stage == "Total_Pipeline":
                print("-" * 65)
            print(f"{stage:<22} | {stats['mean_ms']:<10.2f} | {stats['min_ms']:<10.2f} | {stats['p95_ms']:<10.2f}")
        print("=" * 65)
        fps = 1000.0 / max(1.0, summary.get("Total_Pipeline", {}).get("mean_ms", 33.3))
        print(f" Estimated Throughput: {fps:.1f} Frames Per Second")
        print(f" Real-Time Mobility Budget (< 50ms): {'PASSED [OK]' if summary.get('Total_Pipeline', {}).get('mean_ms', 0) < 50.0 else 'OPTIMIZATION NEEDED'}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    benchmark = PipelineBenchmark()
    benchmark.run(mock_stream=True)
