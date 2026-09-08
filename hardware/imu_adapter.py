"""
IMU (Inertial Measurement Unit) Adapter
=======================================
Reads accelerometer, gyroscope, and compass data (e.g. MPU-6050 / BNO055).
Tracks user head orientation and movement vector.
"""

from typing import Dict, Optional


class IMUAdapter:
    """
    Interfaces with physical IMU sensor over I2C / SPI.
    """

    def __init__(self, i2c_bus: int = 1, address: int = 0x68):
        self.i2c_bus = i2c_bus
        self.address = address
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def read_orientation(self) -> Optional[Dict[str, float]]:
        """
        Returns orientation dictionary:
        { "pitch": float, "roll": float, "yaw": float, "is_walking": bool }
        """
        if not self._connected:
            return None
        return {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "is_walking": False}

    def disconnect(self) -> None:
        self._connected = False
