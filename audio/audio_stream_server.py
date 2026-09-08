"""
Audio Stream Server for Phone Audio Endpoint
============================================
Provides an HTTP server for streaming TTS speech to any smartphone
browser or earphones over the local Wi-Fi / hotspot network.
"""

from http.server import HTTPServer
import logging
import threading
from typing import Optional
from audio.tts import _PhoneAudioHTTPHandler

logger = logging.getLogger("VisionAssist.AudioServer")


class AudioStreamServer:
    """Manages audio relay HTTP server for wearable phone audio."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8088):  # nosec B104
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False

    def start(self) -> None:
        """Starts the audio relay HTTP server on a daemon thread."""
        if self.is_running:
            return
        try:
            self.server = HTTPServer(
                (self.host, self.port),
                _PhoneAudioHTTPHandler
            )
            self.thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.thread.start()
            self.is_running = True
            logger.info(
                f"AudioStreamServer running at "
                f"http://{self.host}:{self.port}/phone-audio"
            )
        except Exception as e:
            logger.debug(f"AudioStreamServer port {self.port} notice: {e}")

    def stop(self) -> None:
        """Stops the audio relay HTTP server."""
        if self.server and self.is_running:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                logger.debug(f"Error stopping AudioStreamServer: {e}")
            self.is_running = False
