"""
Phone Stream Camera Adapter
===========================
Connects to smartphone camera feeds streamed via HTTP/RTSP
(e.g., 'IP Webcam' on Android or RTSP streamer on iOS).
Features auto-reconnection and network drop resilience for live demos.
"""

import logging
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from .camera_interface import BaseCamera
from config import CAMERA_WIDTH, CAMERA_HEIGHT

logger = logging.getLogger("VisionAssist.Input")


class PhoneStreamCamera(BaseCamera):
    """
    Connects to smartphone video stream (MJPEG / RTSP / HTTP URL).
    Automatically reconnects if Wi-Fi packets drop or stream disconnects.
    """

    def __init__(self, stream_url: str, auto_reconnect: bool = True, reconnect_cooldown: float = 2.0):
        self.stream_url = stream_url
        self.auto_reconnect = auto_reconnect
        self.reconnect_cooldown = reconnect_cooldown
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_reconnect_attempt = 0.0
        self._is_connected = False
        self._reconnect_count = 0

    def open(self) -> bool:
        """Open network stream connection."""
        self._last_reconnect_attempt = time.time()
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.stream_url)
            self._is_connected = self.cap.isOpened()
            if self._is_connected:
                logger.info(f"Connected to Phone Camera Stream at {self.stream_url}")
            else:
                logger.warning(f"Could not open stream at {self.stream_url}. Will retry automatically.")
            return self._is_connected
        except Exception as e:
            logger.error(f"Error opening stream: {e}")
            self._is_connected = False
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads frame from phone stream. If disconnected, attempts non-blocking reconnect
        and yields a status standby frame so upstream pipelines do not stall.
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._is_connected = True
                return True, frame

        # Stream disconnected or frame read failed
        self._is_connected = False
        now = time.time()
        if self.auto_reconnect and (now - self._last_reconnect_attempt) > self.reconnect_cooldown:
            self._reconnect_count += 1
            logger.warning(f"Phone stream disconnected. Reconnect attempt #{self._reconnect_count}...")
            self.open()

        # Return standby visual frame during reconnection
        return True, self._generate_standby_frame()

    def _generate_standby_frame(self) -> np.ndarray:
        """Generates visual status frame while searching for phone stream."""
        frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 20, dtype=np.uint8)
        # Pulsing amber notification banner
        cv2.rectangle(frame, (40, 180), (CAMERA_WIDTH - 40, 300), (30, 40, 60), -1)
        cv2.rectangle(frame, (40, 180), (CAMERA_WIDTH - 40, 300), (0, 165, 255), 2)
        cv2.putText(
            frame, "SEARCHING FOR PHONE STREAM...",
            (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"URL: {self.stream_url}",
            (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 190, 200), 1, cv2.LINE_AA
        )
        return frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
        self._is_connected = False

    def is_opened(self) -> bool:
        return self._is_connected or self.auto_reconnect

    @property
    def resolution(self) -> Tuple[int, int]:
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (CAMERA_WIDTH, CAMERA_HEIGHT)
