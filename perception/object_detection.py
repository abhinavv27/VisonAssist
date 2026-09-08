"""
Object Detection Engine (YOLOv8 Wrapper)
========================================
Runs real-time object detection on video frames.
Conforms strictly to Team Interface Contract:
Returns: List[{ "object": str, "confidence": float, "x_center": float, "y_center": float, "bbox": [x1, y1, x2, y2] }]
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np

from config import YOLO_MODEL_NAME, YOLO_CONFIDENCE_THRESHOLD, MODELS_DIR

logger = logging.getLogger("VisionAssist.Perception.ObjectDetection")


class ObjectDetector:
    """YOLOv8 Object Detector with automatic model resolution,
    warmup acceleration, and graceful offline/mock fallback.
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        conf_thresh: float = YOLO_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.device = device
        self.model: Any = None
        self._is_mock = False
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load Ultralytics YOLOv8 from local models directory or cache.
        """
        try:
            from ultralytics import YOLO

            # Check if model exists in models/ folder first, otherwise let Ultralytics resolve
            local_model_path = MODELS_DIR / self.model_name
            target_path = str(local_model_path) if local_model_path.exists() else self.model_name

            logger.info(f"Loading YOLOv8 model from {target_path}...")
            self.model = YOLO(target_path)
            self._is_mock = False
            logger.info("YOLOv8 model initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load ultralytics YOLO ({e}). Operating in heuristic/mock detection mode.")
            self._is_mock = True

    @property
    def is_live(self) -> bool:
        """True if running real YOLOv8 model, False if running heuristic mock.
        """
        return not self._is_mock and self.model is not None

    def warmup(self, frame_size: tuple = (480, 640, 3)) -> float:
        """
        Pre-warms PyTorch inference kernels with a dummy frame to eliminate
        first-frame latency spikes during live user navigation.

        Returns:
            Warmup latency in milliseconds.
        """
        dummy_frame = np.zeros(frame_size, dtype=np.uint8)
        start_time = time.perf_counter()
        self.detect(dummy_frame)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"YOLOv8 warmup completed in {latency_ms:.1f} ms.")
        return latency_ms

    def benchmark(self, num_frames: int = 15, frame_size: tuple = (480, 640, 3)) -> Dict[str, float]:
        """
        Benchmarks inference speed (FPS and latency in milliseconds).
        """
        dummy_frame = np.zeros(frame_size, dtype=np.uint8)
        # Warmup first
        self.detect(dummy_frame)

        latencies = []
        for _ in range(num_frames):
            t0 = time.perf_counter()
            self.detect(dummy_frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = sum(latencies) / len(latencies)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        return {
            "avg_latency_ms": round(avg_latency, 2),
            "fps": round(fps, 1),
            "num_frames": num_frames,
            "is_live_model": self.is_live,
        }

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on image frame.

        Returns:
            List of dicts conforming to team interface:
            [{
                "object": str,
                "confidence": float,
                "x_center": float,
                "y_center": float,
                "bbox": [x1, y1, x2, y2]
            }]
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            logger.debug("Invalid or empty frame passed to ObjectDetector.")
            return []

        try:
            h, w = frame.shape[:2]
            if not self._is_mock and self.model is not None:
                try:
                    kwargs = {"verbose": False, "conf": self.conf_thresh}
                    if self.device:
                        kwargs["device"] = self.device
                    results = self.model(frame, **kwargs)
                    detections: List[Dict[str, Any]] = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = r.names.get(cls_id, f"obj_{cls_id}").lower()
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            # Special handling for critical mobility hazards (vehicles)
                            is_vehicle = cls_name in ("car", "bus", "truck", "motorcycle")
                            thresh = 0.25 if is_vehicle else self.conf_thresh
                            if conf < thresh:
                                continue
                            # Clamp bounding box coordinates to image boundaries
                            x1 = max(0, min(w - 1, x1))
                            y1 = max(0, min(h - 1, y1))
                            x2 = max(0, min(w - 1, x2))
                            y2 = max(0, min(h - 1, y2))

                            x_center = (x1 + x2) / 2.0
                            y_center = (y1 + y2) / 2.0
                            detections.append({
                                "object": cls_name,
                                "confidence": round(conf, 2),
                                "x_center": round(x_center, 1),
                                "y_center": round(y_center, 1),
                                "bbox": [x1, y1, x2, y2],
                            })
                    # Check for stairs hazard (steps/stairways have distinctive periodic horizontal edges)
                    if not any(d["object"] == "stairs" for d in detections):
                        stair_det = self._detect_stairs(frame)
                        if stair_det:
                            detections.append(stair_det)
                    return detections
                except Exception as e:
                    logger.error(f"YOLO inference error: {e}")
            # Fallback heuristic / mock detector for developer scaffolding & offline tests
            return self._heuristic_detect(frame)
        except Exception as e:
            logger.error(f"Sec.33 Error catch in ObjectDetector: {e}")
            return []

    def _detect_stairs(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detects stair steps via periodic horizontal edge structure."""
        try:
            import cv2
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[int(h * 0.20):, :]
            edges = cv2.Canny(roi, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=int(w * 0.15), maxLineGap=20)
            if lines is not None and len(lines) >= 12:
                lines = lines.reshape(-1, 4)
                h_lines = [l for l in lines if abs(l[1] - l[3]) <= 6 and abs(l[0] - l[2]) >= int(w * 0.16)]
                if len(h_lines) >= 8:
                    xs = [min(l[0], l[2]) for l in h_lines] + [max(l[0], l[2]) for l in h_lines]
                    ys = [min(l[1], l[3]) + int(h * 0.20) for l in h_lines] + [max(l[1], l[3]) + int(h * 0.20) for l in h_lines]
                    x1 = max(0, min(xs))
                    x2 = min(w - 1, max(xs))
                    y1 = max(0, min(ys))
                    y2 = min(h - 1, max(ys))
                    conf = min(0.95, 0.65 + len(h_lines) * 0.005)
                    return {
                        "object": "stairs",
                        "confidence": round(float(conf), 2),
                        "x_center": round(float((x1 + x2) / 2.0), 1),
                        "y_center": round(float((y1 + y2) / 2.0), 1),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
        except Exception as e:
            logger.debug(f"Stairs detection error: {e}")
        return None

    def _heuristic_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates detections based on simulated frames or mock indoor layout.
        Allows immediate testing of upstream context and priority engines.
        """
        h, w = frame.shape[:2]
        return [
            {
                "object": "door",
                "confidence": 0.88,
                "x_center": float(w * 0.75),
                "y_center": float(h * 0.50),
                "bbox": [int(w * 0.65), int(h * 0.20), int(w * 0.88), int(h * 0.80)],
            },
            {
                "object": "chair",
                "confidence": 0.79,
                "x_center": float(w * 0.48),
                "y_center": float(h * 0.65),
                "bbox": [int(w * 0.38), int(h * 0.50), int(w * 0.58), int(h * 0.82)],
            },
            {
                "object": "person",
                "confidence": 0.92,
                "x_center": float(w * 0.20),
                "y_center": float(h * 0.45),
                "bbox": [int(w * 0.10), int(h * 0.15), int(w * 0.30), int(h * 0.78)],
            },
        ]
