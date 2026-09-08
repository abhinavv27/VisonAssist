"""
Hardware Adapter Layer for VisionAssist.
Isolates hardware sensors (ESP32, Raspberry Pi, Ultrasonic, IMU, Buttons)
so new devices can be integrated without touching the reasoning or priority layers.
"""

from .camera_adapter import HardwareCameraAdapter
from .ultrasonic_adapter import UltrasonicAdapter, read_sensor_node
from .imu_adapter import IMUAdapter
from .button_adapter import ButtonAdapter
from .esp8266_adapter import ESP8266CompanionAdapter

__all__ = [
    "HardwareCameraAdapter",
    "UltrasonicAdapter",
    "read_sensor_node",
    "IMUAdapter",
    "ButtonAdapter",
    "ESP8266CompanionAdapter",
]

