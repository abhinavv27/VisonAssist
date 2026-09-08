"""
Input layer for VisionAssist.
Provides hardware-agnostic camera and video streaming capture interfaces.
"""

from .camera_interface import BaseCamera
from .webcam import OpenCVWebcam
from .phone_stream import PhoneStreamCamera

__all__ = ["BaseCamera", "OpenCVWebcam", "PhoneStreamCamera"]
