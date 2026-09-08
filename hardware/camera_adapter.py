"""
Hardware Camera Adapter
=======================
Connects custom hardware camera modules (Raspberry Pi Camera Module,
ESP32-CAM, USB UVC) to the standardized BaseCamera input interface.
"""

from typing import Optional, Tuple
import numpy as np
from input.camera_interface import BaseCamera


class HardwareCameraAdapter(BaseCamera):
    """
    Adapter for embedded camera modules (Pi Camera, ESP32-CAM, stereo lenses).
    """

    def __init__(self, device_path: str = "/dev/video0"):
        self.device_path = device_path
        self._is_active = False

    def open(self) -> bool:
        # Hardware driver hook for embedded deployment (e.g. picamera2 / v4l2)
        self._is_active = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._is_active:
            return False, None
        # Placeholder for hardware driver DMA buffer / frame grab
        return False, None

    def release(self) -> None:
        self._is_active = False

    def is_opened(self) -> bool:
        return self._is_active

    @property
    def resolution(self) -> Tuple[int, int]:
        return (640, 480)
