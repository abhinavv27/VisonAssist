"""
Text-To-Speech (TTS) Engine
===========================
Thread-safe, non-blocking offline audio engine.
Supports native Windows SAPI voice synthesis (zero dependencies)
with graceful fallback to pyttsx3 or terminal narration.
Conforms strictly to Team Interface Contract:
Receives: text -> Produces: spoken audio.
"""

import logging
import queue
import threading
import time
from typing import Optional

from config import TTS_RATE, TTS_VOLUME

logger = logging.getLogger("VisionAssist.Audio")


class TextToSpeechEngine:
    """
    Asynchronous Speech Synthesizer.
    Pushes speech jobs to a background queue so camera processing is never blocked.
    """

    def __init__(self, rate: int = TTS_RATE, volume: float = TTS_VOLUME, mute: bool = False):
        self.rate = rate
        self.volume = volume
        self.mute = mute
        self._speech_queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._current_utterance: Optional[str] = None
        self._speech_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._speech_thread.start()

    def speak(self, text: str, interrupt: bool = False) -> None:
        """
        Enqueues text to be spoken.
        
        Args:
            text: Sentence to speak.
            interrupt: If True, clears backlog and speaks immediately.
        """
        if not text or not text.strip() or self.mute:
            return

        if interrupt:
            # Clear pending items in queue
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    break

        try:
            self._speech_queue.put_nowait(text.strip())
        except queue.Full:
            logger.warning("Speech queue full, dropping announcement.")

    def _worker_loop(self) -> None:
        """Background thread handling TTS synthesis."""
        # Try native Windows SAPI first
        sapi_voice = None
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            sapi_voice.Volume = int(self.volume * 100)
            logger.info("Native Windows SAPI voice engine initialized.")
        except Exception as e:
            logger.debug(f"Windows SAPI not available: {e}. Checking pyttsx3...")

        # Fallback to pyttsx3
        pyttsx3_engine = None
        if sapi_voice is None:
            try:
                import pyttsx3
                pyttsx3_engine = pyttsx3.init()
                pyttsx3_engine.setProperty("rate", self.rate)
                pyttsx3_engine.setProperty("volume", self.volume)
                logger.info("pyttsx3 engine initialized.")
            except Exception as e:
                logger.warning(f"Audio TTS fallback to terminal logging: {e}")

        while not self._stop_event.is_set():
            try:
                text = self._speech_queue.get(timeout=0.2)
                self._current_utterance = text
                logger.info(f"[TTS AUDIO]: \"{text}\"")

                if sapi_voice is not None:
                    # SAPI speech flag 0 = synchronous within this worker thread
                    sapi_voice.Speak(text, 0)
                elif pyttsx3_engine is not None:
                    pyttsx3_engine.say(text)
                    pyttsx3_engine.runAndWait()
                else:
                    # Simulated audio delay
                    time.sleep(min(2.0, max(0.5, len(text.split()) * 0.25)))

                self._current_utterance = None
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error during speech synthesis: {e}")
                self._current_utterance = None

    @property
    def current_utterance(self) -> Optional[str]:
        """Returns currently speaking or last spoken sentence."""
        return self._current_utterance

    def stop(self) -> None:
        """Stops the background speech worker thread."""
        self._stop_event.set()
        if self._speech_thread.is_alive():
            self._speech_thread.join(timeout=1.0)
