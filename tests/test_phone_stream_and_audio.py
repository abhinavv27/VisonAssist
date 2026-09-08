"""
Phone Stream & Audio Return Pipeline Test Suite (Phase 3)
=========================================================
Validates the wireless smartphone bridge:
- PhoneStreamCamera URL normalization, non-blocking socket probing, HUD.
- Threaded zero-latency frame grabber and clean release.
- AudioStreamServer & handler endpoints (/phone-audio, /api/speech).
- Emergency priority alert broadcast with haptic & urgency metadata.
"""

import json
import time
import urllib.request

from config import CAMERA_WIDTH, CAMERA_HEIGHT
from input.phone_stream import (
    PhoneStreamCamera,
    normalize_stream_url,
    probe_stream_connectivity
)
from audio.tts import TextToSpeechEngine, _LATEST_SPEECH, _SPEECH_LOCK
from audio.audio_stream_server import AudioStreamServer


class TestPhoneStreamAdapter:
    """Tests PhoneStreamCamera network handling and low-latency buffering."""

    def test_url_normalization(self):
        # Plain IP:port
        target = "http://192.168.43.1:8080/video"
        assert normalize_stream_url("192.168.43.1:8080") == target

        # http:// without /video
        target2 = "http://192.168.1.100:8080/video"
        assert normalize_stream_url("http://192.168.1.100:8080") == target2

        # Already full video URL
        target3 = "http://10.0.0.5:8080/video"
        assert normalize_stream_url("http://10.0.0.5:8080/video") == target3

        # RTSP stream URL
        target4 = "rtsp://192.168.1.20:8554/live"
        assert normalize_stream_url(target4) == target4

    def test_probe_stream_connectivity_unreachable(self):
        # Unreachable local IP returns False quickly without hanging
        t0 = time.perf_counter()
        reachable = probe_stream_connectivity(
            "http://192.0.2.1:8080/video",
            timeout=0.3
        )
        duration = time.perf_counter() - t0
        assert reachable is False
        assert duration < 1.0  # Must be fast and non-blocking

    def test_standby_frame_generation_when_stream_offline(self):
        cam = PhoneStreamCamera(
            stream_url="http://192.0.2.1:8080/video",
            auto_reconnect=True,
            reconnect_cooldown=10.0,
            enable_threading=False,
            probe_first=True
        )
        cam.open()

        ret, frame = cam.read_frame()
        assert ret is True
        assert frame is not None
        assert frame.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        assert cam.resolution == (CAMERA_WIDTH, CAMERA_HEIGHT)
        assert cam.is_opened() is True

        cam.release()
        assert cam.cap is None

    def test_threaded_camera_lifecycle(self):
        cam = PhoneStreamCamera(
            stream_url="http://192.0.2.1:8080/video",
            enable_threading=True,
            probe_first=True
        )
        cam.open()
        ret, frame = cam.read_frame()
        assert ret is True
        assert frame.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        cam.release()


class TestPhoneAudioRelay:
    """Tests phone browser audio endpoint and broadcast synchronization."""

    def test_phone_audio_http_endpoints(self):
        # Spin up a test audio stream server on port 8089
        test_port = 8089
        server = AudioStreamServer(host="127.0.0.1", port=test_port)
        server.start()
        time.sleep(0.3)

        try:
            # 1. Test /api/speech returns valid JSON
            api_url = f"http://127.0.0.1:{test_port}/api/speech"
            with urllib.request.urlopen(api_url, timeout=2.0) as resp:  # nosec
                assert resp.status == 200
                ct = resp.headers.get("Content-Type")
                assert ct == "application/json"
                data = json.loads(resp.read().decode("utf-8"))
                assert "id" in data
                assert "text" in data

            # 2. Test /phone-audio returns accessible HTML
            ui_url = f"http://127.0.0.1:{test_port}/phone-audio"
            with urllib.request.urlopen(ui_url, timeout=2.0) as resp:  # nosec
                assert resp.status == 200
                html = resp.read().decode("utf-8")
                assert "VisionAssist" in html
                assert "Phone Audio" in html
                assert "SpeechSynthesisUtterance" in html

        finally:
            server.stop()

    def test_priority_emergency_broadcast_metadata(self):
        tts = TextToSpeechEngine(mute=True, phone_audio_port=8091)
        time.sleep(0.2)
        try:
            alert_msg = "Warning. Collision hazard detected."
            tts.speak(alert_msg, interrupt=True, priority=True)
            time.sleep(0.2)

            with _SPEECH_LOCK:
                assert _LATEST_SPEECH["text"] == alert_msg
                assert _LATEST_SPEECH["is_urgent"] is True
                assert _LATEST_SPEECH["priority"] is True
        finally:
            tts.stop()
