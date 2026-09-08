"""
VisionAssist Pipeline Latency & Hardware Profiler
=================================================
Benchmarks per-stage execution times and end-to-end latency against the
Master Project Report Section 20 SLA (< 40ms latency, >= 25 FPS).

Stages measured:
1. Frame Preprocessing
2. YOLOv8 Object Detection
3. Context Engine (Depth + 3D Geometry + Risk Engine)
4. Priority Engine (Suppression + Cooldown Filtering)
5. Response Generator (Speech Utterance Formulation)
6. TTS Dispatch (Non-blocking queue enqueue)
7. End-to-End Total Pipeline
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from audio.tts import TextToSpeechEngine
from tests.test_scenes import create_corridor_stairs_scene, create_classroom_scene
from config import ProductMode


def percentile(data: List[float], p: float) -> float:
    """Computes the p-th percentile of a list of floats."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(round(p * (len(sorted_data) - 1)))
    return sorted_data[idx]


def run_latency_profile(
    iterations: int = 30,
    warmup: int = 5,
    save_json: str = None
) -> Dict[str, Any]:
    """Runs high-resolution timing benchmarks on the pipeline."""
    print("=" * 65)
    print("  👁️  VisionAssist Pipeline Latency & Performance Profiler")
    print(f"  Iterations: {iterations} | Warmup: {warmup} | Platform: {sys.platform}")
    print("=" * 65)

    # Initialize components
    detector = ObjectDetector()
    ocr = OCRReader()
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()
    tts = TextToSpeechEngine(mute=True)

    # Prepare representative test scenes
    scenes = [create_classroom_scene(), create_corridor_stairs_scene()]

    # Stage storage in milliseconds
    timings: Dict[str, List[float]] = {
        "1_Preprocessing": [],
        "2_YOLO_Detection": [],
        "3_Context_Engine": [],
        "4_Priority_Engine": [],
        "5_Response_Gen": [],
        "6_TTS_Dispatch": [],
        "Total_End_to_End": [],
    }

    total_runs = warmup + iterations
    print(f"Executing {total_runs} passes across perception & intelligence pipelines...\n")

    for i in range(total_runs):
        frame = scenes[i % len(scenes)]
        is_warmup = i < warmup

        t_start = time.perf_counter_ns()

        # Stage 1: Preprocessing
        t0 = time.perf_counter_ns()
        h, w = frame.shape[:2]
        _ = (h, w)
        t1 = time.perf_counter_ns()

        # Stage 2: YOLO Detection
        detections = detector.detect(frame)
        t2 = time.perf_counter_ns()

        # Stage 3: Context Engine
        context_items = context_engine.process_detections(
            detections=detections,
            frame_shape=frame.shape,
            mode=ProductMode.OBSTACLE_AWARENESS
        )
        t3 = time.perf_counter_ns()

        # Stage 4: Priority Engine
        prioritized = priority_engine.select_top_item(
            context_items=context_items,
            mode=ProductMode.OBSTACLE_AWARENESS,
            force_refresh=True
        )
        t4 = time.perf_counter_ns()

        # Stage 5: Response Generation
        speech_text = ""
        if prioritized:
            resp = response_gen.generate(prioritized, language="en")
            speech_text = resp.get("text", "")
        t5 = time.perf_counter_ns()

        # Stage 6: TTS Dispatch
        if speech_text:
            tts.speak(speech_text, interrupt=True)
        t6 = time.perf_counter_ns()

        total_ms = (t6 - t_start) / 1_000_000.0

        if not is_warmup:
            timings["1_Preprocessing"].append((t1 - t0) / 1_000_000.0)
            timings["2_YOLO_Detection"].append((t2 - t1) / 1_000_000.0)
            timings["3_Context_Engine"].append((t3 - t2) / 1_000_000.0)
            timings["4_Priority_Engine"].append((t4 - t3) / 1_000_000.0)
            timings["5_Response_Gen"].append((t5 - t4) / 1_000_000.0)
            timings["6_TTS_Dispatch"].append((t6 - t5) / 1_000_000.0)
            timings["Total_End_to_End"].append(total_ms)

    tts.stop()

    # Calculate summary metrics
    results: Dict[str, Dict[str, float]] = {}
    for stage, vals in timings.items():
        results[stage] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p95": float(percentile(vals, 0.95)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    total_mean = results["Total_End_to_End"]["mean"]
    fps = 1000.0 / total_mean if total_mean > 0 else 0.0

    # Print Formatted Report Table
    print(f"{'Pipeline Stage':<24} | {'Mean (ms)':<10} | {'Median':<10} | {'p95 (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 72)
    for stage, stats in results.items():
        if stage == "Total_End_to_End":
            print("-" * 72)
        print(
            f"{stage:<24} | "
            f"{stats['mean']:<10.3f} | "
            f"{stats['median']:<10.3f} | "
            f"{stats['p95']:<10.3f} | "
            f"{stats['max']:<10.3f}"
        )
    print("=" * 72)
    print(f"  Mean End-to-End Latency : {total_mean:.2f} ms")
    print(f"  Effective Processing Rate: {fps:.1f} FPS")

    # SLA Verification against Section 20 of Master Report
    target_sla_ms = 40.0
    is_compliant = total_mean <= target_sla_ms
    if is_compliant:
        status_str = f"PASS (<= {target_sla_ms}ms target)"
        print(f"  Section 20 SLA Status    : \033[92m{status_str}\033[0m")
    else:
        status_str = f"WARN (> {target_sla_ms}ms target on current CPU)"
        print(f"  Section 20 SLA Status    : \033[93m{status_str}\033[0m")
    print("=" * 72 + "\n")

    output_payload = {
        "iterations": iterations,
        "results": results,
        "effective_fps": round(fps, 1),
        "target_sla_ms": target_sla_ms,
        "compliant": is_compliant
    }

    if save_json:
        os.makedirs(os.path.dirname(os.path.abspath(save_json)), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"Saved latency profile data to: {save_json}")

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionAssist Pipeline Latency Profiler")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup passes")
    parser.add_argument("--save-json", type=str, default=None, help="Path to save output JSON metrics")
    args = parser.parse_args()

    run_latency_profile(
        iterations=args.iterations,
        warmup=args.warmup,
        save_json=args.save_json
    )
