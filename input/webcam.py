"""
OpenCV Webcam Capture
=====================
Implements BaseCamera for standard laptop and USB webcams.
Includes robust fallback to synthetic frames for headless testing.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from .camera_interface import BaseCamera
from config import CAMERA_WIDTH, CAMERA_HEIGHT, DEFAULT_CAMERA_INDEX


class OpenCVWebcam(BaseCamera):
    """
    Standard laptop webcam capture via OpenCV cv2.VideoCapture.
    """

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX, fallback_to_mock: bool = True):
        self.camera_index = camera_index
        self.fallback_to_mock = fallback_to_mock
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_mock = False
        self._frame_count = 0

    def open(self) -> bool:
        """Open physical webcam device with timeout & fallback."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self._is_mock = False
                return True
        except Exception:
            pass

        if self.fallback_to_mock:
            self._is_mock = True
            return True

        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from webcam or return synthetic animated test scene."""
        if self._is_mock:
            return True, self._generate_mock_frame()

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame

        if self.fallback_to_mock:
            self._is_mock = True
            return True, self._generate_mock_frame()

        return False, None

    def _generate_mock_frame(self) -> np.ndarray:
        """
        Generate synthetic indoor test scene (corridor, door, chair, person)
        useful for offline automated testing and demo validation without webcam.
        """
        self._frame_count += 1
        frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 40, dtype=np.uint8)

        # Floor and ceiling lines
        cv2.line(frame, (0, 380), (CAMERA_WIDTH, 380), (80, 80, 80), 2)
        cv2.line(frame, (0, 100), (CAMERA_WIDTH, 100), (60, 60, 60), 1)

        # Simulated Door on the right
        cv2.rectangle(frame, (450, 120), (590, 380), (30, 90, 180), 2)
        cv2.putText(frame, "DOOR", (480, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 90, 180), 2)

        # Simulated Room Sign
        cv2.rectangle(frame, (460, 140), (580, 180), (255, 255, 255), -1)
        cv2.putText(frame, "LAB 204", (470, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Simulated Chair moving/centered
        offset_x = int(np.sin(self._frame_count * 0.05) * 40)
        chair_x = 240 + offset_x
        cv2.rectangle(frame, (chair_x, 260), (chair_x + 100, 380), (50, 180, 50), 2)
        cv2.putText(frame, "CHAIR", (chair_x + 15, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 180, 50), 2)

        # Watermark banner
        cv2.putText(frame, "[SIMULATED SENSOR FEED]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return frame

    def release(self) -> None:
        """Release video capture device."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self._is_mock = False

    def is_opened(self) -> bool:
        return self._is_mock or (self.cap is not None and self.cap.isOpened())

    @property
    def resolution(self) -> Tuple[int, int]:
        return (CAMERA_WIDTH, CAMERA_HEIGHT)
