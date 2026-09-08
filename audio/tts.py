"""
Text-To-Speech (TTS) Engine
===========================
Thread-safe, non-blocking offline audio engine.
Supports native Windows SAPI voice synthesis (zero dependencies)
with graceful fallback to pyttsx3 or terminal narration.
Conforms strictly to Team Interface Contract:
Receives: text -> Produces: spoken audio.
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from config import TTS_RATE, TTS_VOLUME

logger = logging.getLogger("VisionAssist.Audio")

# Shared state for phone audio streaming
_LATEST_SPEECH = {"id": 0, "text": "", "timestamp": 0.0, "priority": False, "is_urgent": False}
_SPEECH_LOCK = threading.Lock()


def get_latest_speech() -> dict:
    """Returns a thread-safe copy of the most recently queued speech payload."""
    with _SPEECH_LOCK:
        return dict(_LATEST_SPEECH)


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
    <title>VisionAssist · Phone Audio & Haptic Client</title>
    <style>
        body { background: #0A0D14; color: #ECEAE4; font-family: -apple-system, BlinkMacSystemFont, sans-serif; text-align: center; padding: 24px 16px; margin: 0; }
        .card { background: #161B22; border: 1px solid #30363D; border-radius: 16px; padding: 24px; max-width: 420px; margin: auto; box-shadow: 0 8px 24px rgba(0,0,0,0.5); transition: background 0.3s; }
        .card.urgent { background: #450A0A; border-color: #EF4444; animation: flash 0.5s infinite alternate; }
        @keyframes flash { from { border-color: #EF4444; box-shadow: 0 0 20px rgba(239,68,68,0.6); } to { border-color: #7F1D1D; } }
        .status { color: #38BDF8; font-weight: 700; font-size: 0.95rem; margin-bottom: 14px; }
        .utterance { font-size: 1.25rem; color: #5EEAD4; min-height: 70px; margin: 18px 0; font-weight: 700; line-height: 1.4; }
        .btn-row { display: flex; gap: 10px; justify-content: center; margin-bottom: 16px; }
        button { background: #2563EB; color: white; border: none; padding: 12px 20px; border-radius: 10px; font-size: 0.95rem; font-weight: 700; cursor: pointer; transition: transform 0.1s; }
        button:active { transform: scale(0.96); }
        .btn-test { background: #374151; font-size: 0.85rem; padding: 10px 16px; }
        .telemetry { font-size: 0.8rem; color: #8B949E; margin-top: 14px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="card" id="mainCard">
        <h2 style="margin: 4px 0 12px 0;">👁️ VisionAssist</h2>
        <div class="status" id="statusText">● Phone Audio Bridge Connected</div>
        <div class="btn-row">
            <button id="enableBtn" onclick="enableAudio()">🔊 Enable Phone Audio</button>
            <button class="btn-test" id="testBtn" onclick="playTestTone()">🔔 Test Speaker</button>
        </div>
        <div class="utterance" id="textDisplay">Awaiting guidance...</div>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 6px;">Audio plays directly into this phone's speaker or Bluetooth earphones.</p>
        <div class="telemetry" id="telemetryText">Latency: ~20ms | Polling: Active</div>
    </div>
    <script>
        let lastId = 0;
        let audioEnabled = false;
        let audioCtx = null;

        function getAudioContext() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            return audioCtx;
        }

        function playTestTone(freq = 880, dur = 0.15) {
            try {
                const ctx = getAudioContext();
                if (ctx.state === 'suspended') ctx.resume();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(freq, ctx.currentTime);
                gain.gain.setValueAtTime(0.2, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + dur);
            } catch (e) {}
        }

        function enableAudio() {
            audioEnabled = true;
            getAudioContext();
            document.getElementById('enableBtn').style.display = 'none';
            document.getElementById('statusText').innerText = '● Live Audio Active (Earphones / Speaker)';
            document.getElementById('statusText').style.color = '#10B981';
            speak("VisionAssist audio active. Guidance online.", false);
            setInterval(pollSpeech, 300);
        }

        function speak(text, isUrgent) {
            if (!window.speechSynthesis || !text) return;
            window.speechSynthesis.cancel();
            
            const card = document.getElementById('mainCard');
            if (isUrgent) {
                card.classList.add('urgent');
                playTestTone(1200, 0.25);
                if (navigator.vibrate) navigator.vibrate([200, 100, 250]);
            } else {
                card.classList.remove('urgent');
            }

            const u = new SpeechSynthesisUtterance(text);
            u.rate = 1.05;
            window.speechSynthesis.speak(u);
            document.getElementById('textDisplay').innerText = '"' + text + '"';
        }

        async function pollSpeech() {
            try {
                const t0 = performance.now();
                const res = await fetch('/api/speech');
                const data = await res.json();
                const ping = Math.round(performance.now() - t0);
                document.getElementById('telemetryText').innerText = `Latency: ${ping}ms | Status: Online`;
                if (data.id > lastId && data.text) {
                    lastId = data.id;
                    speak(data.text, data.is_urgent || data.priority === 'HIGH');
                }
            } catch (e) {
                document.getElementById('telemetryText').innerText = 'Bridge: Reconnecting...';
            }
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
        self._interrupt_event = threading.Event()
        self._current_utterance: Optional[str] = None
        self._last_spoken_utterance: Optional[str] = None
        self._last_interrupted_utterance: Optional[str] = None
        self._utterance_counter = 0
        self._sapi_voice = None

        self._phone_server: Optional[HTTPServer] = None

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
                self._phone_server = server
                logger.info(f"Phone Audio Relay live at http://{self.phone_audio_host}:{self.phone_audio_port}/phone-audio")
                server.serve_forever()
            except Exception as e:
                logger.debug(f"Could not bind phone audio server on port {self.phone_audio_port}: {e}")

        server_thread = threading.Thread(target=_run_server, daemon=True)
        server_thread.start()

    def speak(self, text: str, interrupt: bool = False, priority: bool = False) -> None:
        """
        Enqueues text to be spoken and broadcasts to phone audio client.
        
        Args:
            text: Sentence to speak.
            interrupt: If True, clears backlog and interrupts current speech.
            priority: If True, treats as Mode-5 emergency alert (cuts off mid-sentence).
        """
        if not text or not text.strip():
            return

        cleaned_text = text.strip()
        is_urgent = interrupt or priority

        with _SPEECH_LOCK:
            self._utterance_counter += 1
            _LATEST_SPEECH.clear()
            _LATEST_SPEECH.update({
                "id": self._utterance_counter,
                "text": cleaned_text,
                "timestamp": time.time(),
                "priority": priority,
                "is_urgent": is_urgent
            })

        self._last_spoken_utterance = cleaned_text

        if is_urgent:
            # Signal background worker to cut off current utterance mid-speech
            if self._current_utterance or self._last_spoken_utterance:
                self._last_interrupted_utterance = self._current_utterance or self._last_spoken_utterance
                logger.info(f"[TTS INTERRUPT]: Cut off ongoing speech \"{self._last_interrupted_utterance}\" for \"{cleaned_text}\"")

            self._interrupt_event.set()
            self._current_utterance = cleaned_text
            self._last_spoken_utterance = cleaned_text

            # Purge SAPI speech instantly
            if self._sapi_voice is not None:
                try:
                    # 2 = SVSFPurgeBeforeSpeak
                    self._sapi_voice.Speak("", 2)
                except Exception as e:
                    logger.debug(f"SAPI purge error: {e}")

            # Clear all pending backlog in queue
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    break
        else:
            self._current_utterance = cleaned_text

        try:
            self._speech_queue.put_nowait(cleaned_text)
        except queue.Full:
            logger.warning("Speech queue full, dropping announcement.")

    def _worker_loop(self) -> None:
        """Background thread handling TTS synthesis."""
        sapi_voice = None
        is_headless_test = bool(
            os.environ.get("PYTEST_CURRENT_TEST")
            or os.environ.get("CI")
            or os.environ.get("GITHUB_ACTIONS")
        )

        # Try native Windows SAPI first (only when interactive and not in automated CI/test runners)
        if not is_headless_test and sys.platform == "win32":
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sapi_voice.Volume = int(self.volume * 100)
                self._sapi_voice = sapi_voice
                logger.info("Native Windows SAPI voice engine initialized.")
            except Exception as e:
                logger.debug(f"Windows SAPI not available: {e}. Checking pyttsx3...")

        # Fallback to pyttsx3 only when not in headless test
        pyttsx3_engine = None
        if sapi_voice is None and not is_headless_test:
            try:
                import pyttsx3
                pyttsx3_engine = pyttsx3.init()
                pyttsx3_engine.setProperty("rate", self.rate)
                pyttsx3_engine.setProperty("volume", self.volume)
                logger.info("pyttsx3 engine initialized.")
            except Exception as e:
                logger.warning(f"Audio TTS fallback to terminal logging: {e}")

        try:
            while not self._stop_event.is_set():
                try:
                    text = self._speech_queue.get(timeout=0.2)
                    self._current_utterance = text
                    self._last_spoken_utterance = text
                    self._interrupt_event.clear()
                    logger.info(f"[TTS AUDIO]: \"{text}\"")

                    word_count = len(text.split())
                    sim_duration = min(2.5, max(0.6, word_count * 0.15))

                    if not self.mute:
                        if sapi_voice is not None:
                            try:
                                # SAPI flag 1 = SVSFlagsAsync (non-blocking, device-safe)
                                sapi_voice.Speak(text, 1)
                            except Exception as e:
                                logger.debug(f"SAPI speech note: {e}")
                        elif pyttsx3_engine is not None:
                            try:
                                pyttsx3_engine.say(text)
                                pyttsx3_engine.runAndWait()
                            except Exception as e:
                                logger.debug(f"pyttsx3 speech note: {e}")

                    # Maintain realistic speech delivery duration checking for priority interrupt
                    elapsed = 0.0
                    while elapsed < sim_duration and not self._interrupt_event.is_set():
                        time.sleep(0.05)
                        elapsed += 0.05

                    self._last_spoken_utterance = text
                    if not self._interrupt_event.is_set():
                        self._current_utterance = None
                    self._speech_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error during speech synthesis: {e}")
                    self._current_utterance = None
        finally:
            if sapi_voice is not None:
                del sapi_voice
                self._sapi_voice = None
                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    @property
    def current_utterance(self) -> Optional[str]:
        """Returns currently speaking or last spoken sentence."""
        return self._current_utterance or self._last_spoken_utterance

    @property
    def last_interrupted_utterance(self) -> Optional[str]:
        """Returns the utterance cut off mid-speech by an emergency interrupt."""
        return self._last_interrupted_utterance

    def stop(self) -> None:
        """Stops the background speech worker thread and closes phone server."""
        self._stop_event.set()
        if self._phone_server:
            try:
                self._phone_server.shutdown()
                self._phone_server.server_close()
            except Exception as e:
                logger.debug(f"Error closing phone server: {e}")
            self._phone_server = None

        if self._speech_thread.is_alive():
            self._speech_thread.join(timeout=1.0)
