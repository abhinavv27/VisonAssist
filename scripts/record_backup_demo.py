"""
VisionAssist - Standalone Backup Demo Video Generator (Phase 4)
==============================================================
Renders a clean, standalone 60-90 second presentation video covering
all 5 modes with real-time HUD telemetry, spatial bounding boxes,
priority matrices, and high-contrast audio caption banners.
Ensures zero stage risk during live judging if Wi-Fi or lighting fails.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# flake8: noqa: E402
import cv2
import numpy as np

from config import CAMERA_WIDTH, CAMERA_HEIGHT, ProductMode
from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from intelligence.ask_engine import AskEngine
from utils.drawing import draw_visual_annotations
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
logger = logging.getLogger("VisionAssist.BackupRecorder")


def render_banner_overlay(
    frame: np.ndarray,
    audio_text: str,
    mode_name: str,
    stage_name: str,
    fps: float
) -> np.ndarray:
    """Renders high-contrast judge presentation banner atop frame."""
    h, w = frame.shape[:2]

    # Top banner card
    cv2.rectangle(frame, (0, 0), (w, 65), (16, 24, 40), -1)
    cv2.line(frame, (0, 65), (w, 65), (56, 189, 248), 2)

    # Audio text
    audio_str = f"CURRENT AUDIO: \"{audio_text}\"" if audio_text else ""
    cv2.putText(
        frame,
        audio_str[:62],
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (94, 234, 212),
        2,
        cv2.LINE_AA
    )

    # Sub-status
    sub_str = f"MODE: {mode_name} | STAGE: {stage_name} | FPS: {fps:.1f}"
    cv2.putText(
        frame,
        sub_str,
        (15, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (148, 163, 184),
        1,
        cv2.LINE_AA
    )

    # Watermark
    wm = "VISIONASSIST STAGE BACKUP"
    cv2.putText(
        frame,
        wm,
        (w - 220, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (56, 189, 248),
        1,
        cv2.LINE_AA
    )
    return frame


def generate_backup_demo(
    output_path: Path,
    total_seconds: float = 60.0,
    fps: int = 20
) -> bool:
    """Generates the multi-stage visual demonstration video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Rendering backup demo video to: {output_path}")

    detector = ObjectDetector()
    ocr = OCRReader()
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()
    ask_engine = AskEngine()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (CAMERA_WIDTH, CAMERA_HEIGHT)
    )

    if not out.isOpened():
        logger.warning("Could not open mp4v codec; falling back to XVID...")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (CAMERA_WIDTH, CAMERA_HEIGHT)
        )

    # 5 Demo Sections corresponding to the 5 Product Modes
    sections = [
        {
            "mode": ProductMode.QUICK_LOOK,
            "stage": "1/5: Quick Look Scene Overview",
            "frame_fn": create_classroom_scene,
            "weight": 0.20,
            "force_refresh": True
        },
        {
            "mode": ProductMode.OBSTACLE_AWARENESS,
            "stage": "2/5: Continuous Obstacle Filtering",
            "frame_fn": create_corridor_stairs_scene,
            "weight": 0.20,
            "force_refresh": False
        },
        {
            "mode": ProductMode.READ,
            "stage": "3/5: Room Signage OCR Extraction",
            "frame_fn": create_room_sign_scene,
            "weight": 0.20,
            "force_refresh": True
        },
        {
            "mode": ProductMode.ASK,
            "stage": "4/5: Grounded Visual Q&A",
            "frame_fn": create_classroom_scene,
            "weight": 0.20,
            "query": "Where is the door?",
            "force_refresh": True
        },
        {
            "mode": ProductMode.SAFETY_ALERT,
            "stage": "5/5: Critical Collision Emergency Interrupt",
            "frame_fn": create_critical_obstacle_scene,
            "weight": 0.20,
            "force_refresh": True
        }
    ]

    total_frames = int(total_seconds * fps)
    current_audio = "VisionAssist online. Initializing sensors..."

    for sec in sections:
        sec_frames = int(total_frames * sec["weight"])
        raw_frame = sec["frame_fn"]()
        detections = detector.detect(raw_frame)
        context_items = context_engine.process_detections(
            detections,
            raw_frame.shape,
            sec["mode"]
        )
        ocr_items = []
        if sec["mode"] in (ProductMode.READ, ProductMode.ASK):
            ocr_items = ocr.read_text(raw_frame)

        # Determine speech text for this segment
        if sec["mode"] == ProductMode.ASK:
            current_audio = ask_engine.ask(
                query=sec.get("query", "Where is door?"),
                frame=raw_frame,
                context_items=context_items,
                ocr_items=ocr_items
            )
        else:
            prioritized = priority_engine.select_top_item(
                context_items=context_items,
                mode=sec["mode"],
                ocr_items=ocr_items,
                force_refresh=sec.get("force_refresh", False)
            )
            if prioritized:
                resp = response_gen.generate(prioritized)
                current_audio = resp.get("text", current_audio)

        # Draw visual annotations
        annotated = draw_visual_annotations(
            frame=raw_frame,
            context_items=context_items,
            current_audio=current_audio,
            mode_label=sec["mode"].name,
            fps=float(fps)
        )

        annotated = render_banner_overlay(
            frame=annotated,
            audio_text=current_audio,
            mode_name=sec["mode"].name,
            stage_name=sec["stage"],
            fps=float(fps)
        )

        for _ in range(sec_frames):
            out.write(annotated)

    out.release()
    logger.info(f"Rendered {total_frames} frames ({total_seconds}s) cleanly.")
    return output_path.exists() and output_path.stat().st_size > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record backup demo video")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Demo length in seconds (default: 30s for quick generation)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second"
    )
    parser.add_argument(
        "--output",
        default="demo/backup_demo.mp4",
        help="Target MP4 path"
    )
    args = parser.parse_args()

    target_path = Path(args.output).resolve()
    success = generate_backup_demo(
        output_path=target_path,
        total_seconds=args.duration,
        fps=args.fps
    )
    print(f"Backup Video Generation: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
