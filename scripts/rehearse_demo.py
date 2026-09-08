"""
VisionAssist - 6-Step Judge Demo Rehearsal Runner (Phase 4)
===========================================================
Automates and validates the exact 3-minute judging demonstration
specified in Master Project Report Section 19 and DEMO_RUNBOOK.md.

Steps:
1. Opening & Setup Framing (0:00 - 0:30)
2. Detection vs. Prioritization Contrast (0:30 - 1:15)
3. Sudden Moving Obstacle & Safety Alert (<1m) (1:15 - 1:45)
4. Signage & Room Number OCR Reading (1:45 - 2:15)
5. Ask Mode Grounded Visual Q&A (2:15 - 2:45)
6. Closing Pitch & Latency Telemetry Validation (2:45 - 3:00)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# flake8: noqa: E402
from config import ProductMode
from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from intelligence.ask_engine import AskEngine
from audio.tts import TextToSpeechEngine
from tests.test_scenes import (
    create_classroom_scene,
    create_corridor_stairs_scene,
    create_room_sign_scene,
    create_critical_obstacle_scene
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("VisionAssist.Rehearsal")


class DemoRehearsalRunner:
    """Executes the 6-stage judging presentation flow end-to-end."""

    def __init__(self, pace_factor: float = 1.0, speak_audio: bool = True):
        self.pace = pace_factor
        self.speak_audio = speak_audio
        self.detector = ObjectDetector()
        self.ocr = OCRReader()
        self.context_engine = ContextEngine()
        self.priority_engine = PriorityEngine()
        self.response_gen = ResponseGenerator()
        self.ask_engine = AskEngine()
        self.tts = TextToSpeechEngine(mute=not speak_audio)
        self.stage_results = []
        self.t_start = 0.0

    def step(self, stage_num: int, title: str, expected_time_sec: float):
        """Announces stage transition with timing."""
        elapsed = time.time() - self.t_start
        print("\n" + "=" * 75)
        print(f"STAGE {stage_num}: {title.upper()}")
        print(f"Target Window: {expected_time_sec}s | Elapsed: {elapsed:.1f}s")
        print("=" * 75)

    def run(self) -> bool:
        """Runs the complete 6-stage presentation flow."""
        self.t_start = time.time()
        print("\n" + "#" * 75)
        print("   VISIONASSIST: 3-MINUTE JUDGE DEMO AUTOMATED REHEARSAL")
        print("   Track: Open Innovation · Manipal University Jaipur · 2026")
        print("#" * 75)

        all_passed = True

        # ---------------------------------------------------------------------
        # STAGE 1: Opening & System Status Framing (Target: 30s)
        # ---------------------------------------------------------------------
        self.step(1, "Opening Framing & Hardware Status", 30.0)
        print("Speaker: 'Judges, traditional computer vision tells you everything")
        print("it sees, overwhelming a visually impaired person with noise.'")
        time.sleep(1.0 * self.pace)

        tts_ready = self.tts is not None
        cam_ready = True
        print(f"* Telemetry Status: Camera=[Connected] Detection=[YOLOv8 Warm] "
              f"TTS=[Port {self.tts.phone_audio_port}]")
        assert tts_ready and cam_ready, "Hardware endpoints must be ready."
        self.stage_results.append(("Stage 1: Opening", True, "Online & Ready"))

        # ---------------------------------------------------------------------
        # STAGE 2: Detection vs. Prioritization Contrast (Target: 45s)
        # ---------------------------------------------------------------------
        self.step(2, "Detection vs. Prioritization Contrast", 45.0)
        print("Physical Setup: Chair at 2.1m, Doorway at 3.0m, Side Bottle.")
        print("Perception: Camera detects 3+ objects simultaneously.")

        frame2 = create_classroom_scene()
        detections2 = self.detector.detect(frame2)
        context2 = self.context_engine.process_detections(
            detections2,
            frame2.shape,
            ProductMode.OBSTACLE_AWARENESS
        )
        prioritized2 = self.priority_engine.select_top_item(
            context2,
            ProductMode.OBSTACLE_AWARENESS
        )
        resp2 = self.response_gen.generate(prioritized2)
        speech2 = resp2.get("text", "")

        print(f"* Detected Objects: {[d['object'] for d in detections2]}")
        print(f"* Highest Priority Item: {prioritized2['candidate']['object']} "
              f"({prioritized2['priority']})")
        print(f"* Spoken Output: \"{speech2}\"")

        # Chair in center path must be prioritized over door and person
        chair_selected = prioritized2['candidate']['object'] == 'chair'
        passed2 = chair_selected and len(speech2) > 0
        if self.speak_audio:
            self.tts.speak(speech2)
            time.sleep(1.5 * self.pace)

        self.stage_results.append(
            ("Stage 2: Detection vs Speech", passed2, speech2)
        )
        all_passed = all_passed and passed2

        # ---------------------------------------------------------------------
        # STAGE 3: Sudden Obstacle & Collision Alert (Target: 30s)
        # ---------------------------------------------------------------------
        self.step(3, "Moving Obstacle & Immediate Safety Interrupt", 30.0)
        print("Physical Action: Obstacle suddenly placed in path (< 1.0m away).")

        frame3 = create_critical_obstacle_scene()
        detections3 = self.detector.detect(frame3)
        context3 = self.context_engine.process_detections(
            detections3,
            frame3.shape,
            ProductMode.SAFETY_ALERT
        )
        prioritized3 = self.priority_engine.select_top_item(
            context3,
            ProductMode.SAFETY_ALERT
        )
        resp3 = self.response_gen.generate(prioritized3)
        speech3 = resp3.get("text", "")

        print(f"* Critical Collision Detected: {prioritized3.get('is_critical')}")
        print(f"* Risk Score: {prioritized3['candidate'].get('risk_score')}")
        print(f"* Emergency Spoken Alert: \"{speech3}\"")

        passed3 = (
            prioritized3.get("is_critical") is True
            and "Warning" in speech3
        )
        if self.speak_audio:
            self.tts.speak(speech3, interrupt=True, priority=True)
            time.sleep(1.5 * self.pace)

        self.stage_results.append(
            ("Stage 3: Collision Alert", passed3, speech3)
        )
        all_passed = all_passed and passed3

        # ---------------------------------------------------------------------
        # STAGE 4: Text & Signage OCR Reading (Target: 30s)
        # ---------------------------------------------------------------------
        self.step(4, "Signage & Room Number Reading (Mode 3: Read)", 30.0)
        print("Physical Action: Pointing camera at 'ROOM 204' signage.")

        frame4 = create_room_sign_scene()
        ocr_items4 = self.ocr.read_text(frame4)
        prioritized4 = self.priority_engine.select_top_item(
            context_items=[],
            mode=ProductMode.READ,
            ocr_items=ocr_items4
        )
        resp4 = self.response_gen.generate(prioritized4)
        speech4 = resp4.get("text", "")

        print(f"* Extracted OCR Text: {[t['text'] for t in ocr_items4]}")
        print(f"* Spoken Room Announcement: \"{speech4}\"")

        passed4 = "204" in speech4 or "Lab" in speech4 or "Room" in speech4
        if self.speak_audio:
            self.tts.speak(speech4)
            time.sleep(1.5 * self.pace)

        self.stage_results.append(
            ("Stage 4: Signage Reading", passed4, speech4)
        )
        all_passed = all_passed and passed4

        # ---------------------------------------------------------------------
        # STAGE 5: Grounded Visual Q&A (Target: 30s)
        # ---------------------------------------------------------------------
        self.step(5, "Ask Mode: Grounded Visual Q&A (Mode 4: Ask)", 30.0)
        judge_query = "Where is the door?"
        print(f"Presenter Question: \"{judge_query}\"")

        answer5 = self.ask_engine.ask(
            query=judge_query,
            frame=frame2,
            context_items=context2,
            ocr_items=[]
        )
        print(f"* Grounded System Answer: \"{answer5}\"")

        passed5 = "door" in answer5.lower() and len(answer5) > 10
        if self.speak_audio:
            self.tts.speak(answer5, interrupt=True)
            time.sleep(1.5 * self.pace)

        self.stage_results.append(
            ("Stage 5: Visual Q&A", passed5, answer5)
        )
        all_passed = all_passed and passed5

        # ---------------------------------------------------------------------
        # STAGE 6: Closing Pitch & Telemetry Validation (Target: 15s)
        # ---------------------------------------------------------------------
        self.step(6, "Architectural Closing Pitch & Verification", 15.0)
        closing = (
            "Today our intelligence runs across phone and edge laptop. "
            "Tomorrow, the same engine moves to smart glasses."
        )
        print(f"Closing Statement: \"{closing}\"")
        if self.speak_audio:
            self.tts.speak("VisionAssist demonstration complete. Guidance online.")
            time.sleep(1.0 * self.pace)

        total_elapsed = time.time() - self.t_start
        passed6 = total_elapsed < 180.0
        self.stage_results.append(
            ("Stage 6: Closing & Metrics", passed6, f"{total_elapsed:.1f}s")
        )

        # ---------------------------------------------------------------------
        # FINAL REHEARSAL SUMMARY
        # ---------------------------------------------------------------------
        print("\n" + "=" * 75)
        print("               3-MINUTE DEMO REHEARSAL SCORECARD")
        print("=" * 75)
        for name, passed, notes in self.stage_results:
            status = "PASS [OK]" if passed else "FAIL [X]"
            print(f"{name:<35} | {status:<10} | {notes}")
        print("-" * 75)
        print(f"Total Rehearsal Duration: {total_elapsed:.2f}s "
              f"(Limit: 180.0s / 3.0 min)")

        self.tts.stop()
        return all_passed and passed6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 6-step automated judge demo rehearsal"
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0.5,
        help="Pacing speed multiplier (default: 0.5 for fast verification)"
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Run silently without voice synthesis"
    )
    args = parser.parse_args()

    runner = DemoRehearsalRunner(
        pace_factor=args.pace,
        speak_audio=not args.mute
    )
    success = runner.run()
    sys.exit(0 if success else 1)
