"""
Tactile Trigger Button Adapter
==============================
Interfaces with physical wearable buttons (trigger switches, capacitive touch).
Triggers on-demand modes (e.g., Quick Look on single press, Read on double press).
"""

from typing import Callable, Optional


class ButtonAdapter:
    """
    GPIO push-button handler for wearable glasses frame.
    """

    def __init__(self, pin: int = 17):
        self.pin = pin
        self._on_press_callback: Optional[Callable[[], None]] = None

    def register_callback(self, callback: Callable[[], None]) -> None:
        self._on_press_callback = callback

    def simulate_press(self) -> None:
        """Simulate hardware button event in development."""
        if self._on_press_callback:
            self._on_press_callback()
