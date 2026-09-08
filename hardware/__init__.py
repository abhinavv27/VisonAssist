"""
Hardware Adapter Layer for VisionAssist.
Isolates hardware sensors (ESP32, Raspberry Pi, Ultrasonic, IMU, Buttons)
so new devices can be integrated without touching the reasoning or priority layers.
"""

from .camera_adapter import HardwareCameraAdapter
from .ultrasonic_adapter import UltrasonicAdapter
from .imu_adapter import IMUAdapter
from .button_adapter import ButtonAdapter

__all__ = [
    "HardwareCameraAdapter",
    "UltrasonicAdapter",
    "IMUAdapter",
    "ButtonAdapter"
]
