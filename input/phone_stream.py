"""
Phone Stream Camera Adapter
===========================
Connects to smartphone camera feeds streamed via HTTP/RTSP
(e.g., 'IP Webcam' on Android or RTSP streamer on iOS).
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from .camera_interface import BaseCamera
from config import CAMERA_WIDTH, CAMERA_HEIGHT


class PhoneStreamCamera(BaseCamera):
    """
    Connects to smartphone video stream (MJPEG / RTSP / HTTP URL).
    """

    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open network stream connection."""
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            return self.cap.isOpened()
        except Exception:
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame
        return False, None

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (CAMERA_WIDTH, CAMERA_HEIGHT)
