"""
Ultrasonic Sensor Adapter & ESP32 Sensor Node Client
===================================================
Hardware driver interface for ultrasonic rangefinders (e.g. HC-SR04).
Connects to either direct GPIO/serial or polls the ESP32 Wi-Fi JSON sensor node
(http://<esp32-ip>/status) as specified in Hardware Bring-Up Guide Section 5.
Returns: { "distance_cm": float, "button_pressed": bool }
"""

import logging
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger("VisionAssist.Hardware.Ultrasonic")

# Default ESP32 sensor node URL
DEFAULT_ESP32_SENSOR_URL = "http://192.168.1.43/status"


def read_sensor_node(sensor_url: str = DEFAULT_ESP32_SENSOR_URL, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
    """
    Polls the ESP32 Wi-Fi sensor node endpoint.
    Conforms directly to Hardware Bring-Up Guide Section 5.3:
    Returns:
        {"distance_cm": float, "button_pressed": bool} or None if unreachable.
    """
    try:
        r = requests.get(sensor_url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data
    except requests.RequestException:
        # Per report Section 8, context engine treats None as "no extra evidence"
        return None


class UltrasonicAdapter:
    """
    Adapter interfacing with physical ultrasonic hardware or the ESP32 Wi-Fi node.
    Provides metric distance measurements to complement monocular vision.
    """

    def __init__(self, sensor_url: str = DEFAULT_ESP32_SENSOR_URL, trigger_pin: int = 5, echo_pin: int = 18):
        self.sensor_url = sensor_url
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self._connected = False
        self._last_distance_meters: Optional[float] = None
        self._last_button_pressed: bool = False

    def connect(self) -> bool:
        """Verify network connectivity to ESP32 node or initialize hardware pins."""
        data = read_sensor_node(self.sensor_url, timeout=1.0)
        if data is not None:
            self._connected = True
            logger.info(f"Connected to ESP32 sensor node at {self.sensor_url}")
            return True
        else:
            logger.warning(f"Could not reach ESP32 sensor node at {self.sensor_url}. Operating in disconnected mode.")
            self._connected = False
            return False

    def read_distance_meters(self, timeout: float = 0.5) -> Optional[float]:
        """
        Poll sensor for obstacle distance in meters.
        Returns distance in meters (e.g. 1.5), or None if unavailable/timeout.
        """
        data = read_sensor_node(self.sensor_url, timeout=timeout)
        if data and "distance_cm" in data:
            dist_cm = float(data["distance_cm"])
            if dist_cm > 0:
                dist_m = round(dist_cm / 100.0, 2)
                self._last_distance_meters = dist_m
                return dist_m
        return None

    def is_button_pressed(self, timeout: float = 0.5) -> bool:
        """
        Check if physical trigger button on sensor node was pressed.
        Useful for triggering Mode 1 (Quick Look) or Mode 3 (Read) without laptop keyboard.
        """
        data = read_sensor_node(self.sensor_url, timeout=timeout)
        if data and "button_pressed" in data:
            self._last_button_pressed = bool(data["button_pressed"])
            return self._last_button_pressed
        return False

    def disconnect(self) -> None:
        self._connected = False
