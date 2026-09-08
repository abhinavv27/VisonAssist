"""
Ultrasonic Sensor Adapter
=========================
Hardware driver interface for ultrasonic rangefinders (e.g. HC-SR04).
Provides ground-truth distance measurements to complement monocular vision.
"""

from typing import Optional


class UltrasonicAdapter:
    """
    Interfaces with physical ultrasonic hardware over GPIO / Serial.
    """

    def __init__(self, trigger_pin: int = 23, echo_pin: int = 24):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self._connected = False

    def connect(self) -> bool:
        """Initialize GPIO pins or serial communication."""
        self._connected = True
        return True

    def read_distance_meters(self) -> Optional[float]:
        """
        Poll sensor for closest obstacle distance.
        Returns distance in meters, or None if sensor read timeout.
        """
        if not self._connected:
            return None
        # Stub for future hardware reading (HC-SR04 pulseIn calculation)
        return None

    def disconnect(self) -> None:
        self._connected = False
