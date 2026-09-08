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
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from config import TTS_RATE, TTS_VOLUME

logger = logging.getLogger("VisionAssist.Audio")

# Shared state for phone audio streaming
_LATEST_SPEECH = {"id": 0, "text": "", "timestamp": 0.0}
_SPEECH_LOCK = threading.Lock()


class _PhoneAudioHTTPHandler(BaseHTTPRequestHandler):
    """Serves audio stream API and browser audio client to smartphone."""

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        pass

    def do_GET(self):
        if self.path == "/api/speech":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with _SPEECH_LOCK:
                data = json.dumps(_LATEST_SPEECH).encode("utf-8")
            self.wfile.write(data)
        elif self.path in ("/", "/phone-audio"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionAssist · Phone Audio Client</title>
    <style>
        body { background: #0A0D14; color: #ECEAE4; font-family: sans-serif; text-align: center; padding: 30px; }
        .card { background: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 24px; max-width: 400px; margin: auto; }
        .status { color: #38BDF8; font-weight: bold; margin-bottom: 16px; }
        .utterance { font-size: 1.2rem; color: #5EEAD4; min-height: 60px; margin: 20px 0; font-weight: 600; }
        button { background: #2563EB; color: white; border: none; padding: 14px 28px; border-radius: 8px; font-size: 1rem; font-weight: bold; cursor: pointer; }
    </style>
</head>
<body>
    <div class="card">
        <h2>👁️ VisionAssist</h2>
        <div class="status">● Phone Audio Output Connected</div>
        <button id="enableBtn" onclick="enableAudio()">🔊 Tap to Enable Audio</button>
        <div class="utterance" id="textDisplay">Awaiting guidance...</div>
        <p style="color: #8B949E; font-size: 0.85rem;">Audio guidance will play automatically through this phone's speaker / earphones.</p>
    </div>
    <script>
        let lastId = 0;
        let audioEnabled = false;

        function enableAudio() {
            audioEnabled = true;
            document.getElementById('enableBtn').style.display = 'none';
            speak("VisionAssist phone audio enabled. System ready.");
            setInterval(pollSpeech, 400);
        }

        function speak(text) {
            if (!window.speechSynthesis || !text) return;
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.rate = 1.05;
            window.speechSynthesis.speak(u);
            document.getElementById('textDisplay').innerText = '"' + text + '"';
        }

        async function pollSpeech() {
            try {
                const res = await fetch('/api/speech');
                const data = await res.json();
                if (data.id > lastId && data.text) {
                    lastId = data.id;
                    speak(data.text);
                }
            } catch (e) {}
        }
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class TextToSpeechEngine:
    """
    Asynchronous Speech Synthesizer.
    Pushes speech jobs to a background queue and broadcasts to Phone Audio Client.
    """

    def __init__(
        self,
        rate: int = TTS_RATE,
        volume: float = TTS_VOLUME,
        mute: bool = False,
        phone_audio_host: Optional[str] = None,
        phone_audio_port: int = 8088
    ):
        self.rate = rate
        self.volume = volume
        self.mute = mute
        self.phone_audio_host = phone_audio_host or "0.0.0.0"  # nosec B104
        self.phone_audio_port = phone_audio_port
        self._speech_queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._current_utterance: Optional[str] = None
        self._utterance_counter = 0

        # Background synthesis worker
        self._speech_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._speech_thread.start()

        # Background phone audio relay HTTP server
        self._start_phone_server()

    def _start_phone_server(self):
        def _run_server():
            try:
                # Binds to local Wi-Fi / LAN interface so phone can receive audio
                server = HTTPServer((self.phone_audio_host, self.phone_audio_port), _PhoneAudioHTTPHandler)
                logger.info(f"Phone Audio Relay live at http://{self.phone_audio_host}:{self.phone_audio_port}/phone-audio")
                server.serve_forever()
            except Exception as e:
                logger.debug(f"Could not bind phone audio server on port {self.phone_audio_port}: {e}")

        server_thread = threading.Thread(target=_run_server, daemon=True)
        server_thread.start()

    def speak(self, text: str, interrupt: bool = False) -> None:
        """
        Enqueues text to be spoken and broadcasts to phone audio client.
        
        Args:
            text: Sentence to speak.
            interrupt: If True, clears backlog and speaks immediately.
        """
        if not text or not text.strip() or self.mute:
            return

        cleaned_text = text.strip()

        global _LATEST_SPEECH
        with _SPEECH_LOCK:
            self._utterance_counter += 1
            _LATEST_SPEECH = {
                "id": self._utterance_counter,
                "text": cleaned_text,
                "timestamp": time.time()
            }

        if interrupt:
            # Clear pending items in queue
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    break

        try:
            self._speech_queue.put_nowait(cleaned_text)
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
