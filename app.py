"""
VisionAssist - Main Application Entrypoint
==========================================
Launches the VisionAssist assistive perception loop.
Supports both CLI / OpenCV desktop mode and the Streamlit monitoring dashboard.

Usage:
    py app.py                 # Run live OpenCV loop with webcam & voice guidance
    py app.py --mock          # Run with simulated camera feed (no webcam needed)
    py app.py --streamlit     # Launch the dark modern Streamlit dashboard
    py app.py --mode read     # Run directly in OCR reading mode
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2

from config import ProductMode, DEFAULT_MODE, CAMERA_WIDTH, CAMERA_HEIGHT
from input.webcam import OpenCVWebcam
from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from audio.tts import TextToSpeechEngine
from utils.logger import setup_logger
from utils.drawing import draw_visual_annotations

logger = setup_logger("VisionAssist.App")


def run_cli_loop(mode: ProductMode, use_mock: bool = False, no_gui: bool = False):
    """
    Core closed-loop execution:
    Camera -> Perception -> Context/Risk -> Priority -> Response Gen -> TTS
    """
    logger.info("Initializing VisionAssist MVP v0 Pipeline...")

    # 1. Input layer
    camera = OpenCVWebcam(fallback_to_mock=use_mock)
    if not camera.open():
        logger.error("Could not open camera stream. Exiting.")
        return

    # 2. Perception layer
    detector = ObjectDetector()
    ocr = OCRReader()

    # 3. Intelligence layer
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()

    # 4. Audio layer
    tts = TextToSpeechEngine()

    current_audio_text = "VisionAssist online."
    tts.speak(current_audio_text)

    logger.info("=" * 60)
    logger.info(f"VisionAssist running in mode: {mode.name}")
    logger.info("Keyboard shortcuts in OpenCV window:")
    logger.info("  'q' - Quit")
    logger.info("  '1' - Quick Look Mode")
    logger.info("  '2' - Obstacle Awareness Mode")
    logger.info("  '3' - Read OCR Mode")
    logger.info("  'r' - Reset audio debounce cooldown")
    logger.info("=" * 60)

    prev_time = time.time()
    active_mode = mode

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time

            # Perception
            detections = detector.detect(frame)
            ocr_items = []
            if active_mode == ProductMode.READ:
                ocr_items = ocr.read_text(frame)

            # Context & Risk reasoning
            context_items = context_engine.process_detections(
                detections=detections,
                frame_shape=frame.shape,
                mode=active_mode
            )

            # Priority selection
            prioritized = priority_engine.select_top_item(
                context_items=context_items,
                mode=active_mode,
                ocr_items=ocr_items
            )

            # Generate natural sentence & speak
            if prioritized:
                res = response_gen.generate(prioritized)
                text_to_speak = res.get("text", "")
                if text_to_speak:
                    current_audio_text = text_to_speak
                    logger.info(f"==> PRIORITY SPEECH: {text_to_speak}")
                    tts.speak(text_to_speak, interrupt=prioritized.get("is_critical", False))

            # Display GUI window
            if not no_gui:
                annotated = draw_visual_annotations(
                    frame=frame,
                    context_items=context_items,
                    current_audio=current_audio_text,
                    mode_label=active_mode.name,
                    fps=fps
                )
                cv2.imshow("VisionAssist - Live Monitoring Feed", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("1"):
                    active_mode = ProductMode.QUICK_LOOK
                    logger.info("Switched to Mode 1: Quick Look")
                    priority_engine.reset_cooldown()
                elif key == ord("2"):
                    active_mode = ProductMode.OBSTACLE_AWARENESS
                    logger.info("Switched to Mode 2: Obstacle Awareness")
                elif key == ord("3"):
                    active_mode = ProductMode.READ
                    logger.info("Switched to Mode 3: Read (OCR)")
                elif key == ord("r"):
                    priority_engine.reset_cooldown()
                    logger.info("Audio cooldown reset.")
            else:
                time.sleep(0.03)

    except KeyboardInterrupt:
        logger.info("Terminated by user interrupt.")
    finally:
        logger.info("Cleaning up resources...")
        camera.release()
        tts.stop()
        if not no_gui:
            cv2.destroyAllWindows()
        logger.info("VisionAssist shutdown complete.")


def launch_streamlit():
    """Launch the Streamlit web dashboard."""
    dashboard_path = Path(__file__).resolve().parent / "ui" / "dashboard.py"
    logger.info(f"Launching Streamlit dashboard: {dashboard_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="VisionAssist Accessibility Platform")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit UI dashboard")
    parser.add_argument("--mock", action="store_true", help="Use simulated test frames (no webcam required)")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless terminal mode without OpenCV window")
    parser.add_argument(
        "--mode",
        choices=["quick_look", "obstacle", "read", "ask", "safety_alert"],
        default="obstacle",
        help="Initial operating mode"
    )

    args = parser.parse_args()

    if args.streamlit:
        launch_streamlit()
    else:
        mode_map = {
            "quick_look": ProductMode.QUICK_LOOK,
            "obstacle": ProductMode.OBSTACLE_AWARENESS,
            "read": ProductMode.READ,
            "ask": ProductMode.ASK,
            "safety_alert": ProductMode.SAFETY_ALERT,
        }
        selected_mode = mode_map.get(args.mode, DEFAULT_MODE)
        run_cli_loop(mode=selected_mode, use_mock=args.mock, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
