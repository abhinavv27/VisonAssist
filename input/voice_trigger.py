"""
Voice Trigger Module (Hands-Free Wake Word Detection)
====================================================
Listens for the wake phrase "Vision, look" on a background thread.
Invokes the provided callback (e.g. Mode 1 Quick Look or Ask Mode)
without requiring physical keyboard interaction.
Includes graceful fallback if SpeechRecognition or microphone is unavailable.
"""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("VisionAssist.VoiceTrigger")

try:
    import speech_recognition as sr
except ImportError:
    sr = None


class VoiceTriggerListener:
    """
    Hands-free acoustic wake word listener.
    Listens asynchronously for 'Vision, look' or 'Vision' triggers.
    """

    def __init__(
        self,
        callback: Optional[Callable[[], None]] = None,
        wake_phrase: str = "vision, look",
        poll_interval: float = 0.5
    ):
        self.callback = callback
        self.wake_phrase = wake_phrase.lower()
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self.is_active = False
        self.last_triggered_time = 0.0

    def start(self) -> bool:
        """Starts background listening thread."""
        if self.is_active:
            return True

        if sr is None:
            logger.info(
                "SpeechRecognition not installed; voice trigger in standby. "
                "Keyboard shortcuts & UI triggers active."
            )
            return False

        self._stop_event.clear()
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self._listener_thread.start()
        self.is_active = True
        logger.info(f"VoiceTrigger active. Wake phrase: '{self.wake_phrase}'")
        return True

    def _listen_loop(self) -> None:
        """Background loop reading audio from microphone."""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        try:
            mic = sr.Microphone()
        except Exception as e:
            logger.warning(f"Microphone unavailable for voice trigger: {e}")
            self.is_active = False
            return

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

        while not self._stop_event.is_set():
            try:
                with mic as source:
                    audio = recognizer.listen(
                        source,
                        timeout=self.poll_interval,
                        phrase_time_limit=3.0
                    )
                text = recognizer.recognize_google(audio).lower()
                logger.info(f"[VOICE TRIGGER DETECTED]: '{text}'")

                if (
                    "vision" in text
                    and ("look" in text or "see" in text or "read" in text)
                ):
                    now = time.time()
                    if now - self.last_triggered_time > 2.0:
                        self.last_triggered_time = now
                        logger.info(">>> WAKE PHRASE TRIGGERED! <<<")
                        if self.callback:
                            self.callback()
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                pass
            except Exception as e:
                logger.debug(f"Voice trigger loop note: {e}")
                time.sleep(0.1)

    def trigger_manually(self) -> bool:
        """Simulates wake word trigger (used by UI buttons & testing)."""
        logger.info(f"Manual wake trigger fired for '{self.wake_phrase}'")
        self.last_triggered_time = time.time()
        if self.callback:
            self.callback()
            return True
        return False

    def stop(self) -> None:
        """Stops background listener."""
        self._stop_event.set()
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.0)
            self._listener_thread = None
        self.is_active = False
