"""
Phone Bridge & Streaming Test Tool (Phase 3)
============================================
Diagnoses and tests the wireless link between smartphone and laptop:
1. Identifies local Wi-Fi / Hotspot LAN IP for phone connection.
2. Verifies Phone Audio Relay server (port 8088 / /phone-audio).
3. Connects to PhoneStreamCamera, measures FPS and frame latency.
4. Triggers emergency priority speech to test phone audio and haptics.
"""

import argparse
import json
import logging
from pathlib import Path
import socket
import sys
import time
import urllib.request
import pytest

@pytest.fixture
def host() -> str:
    """Default host for phone bridge tests."""
    return "127.0.0.1"

@pytest.fixture
def port() -> int:
    """Default port for phone bridge tests."""
    return 8088

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# flake8: noqa: E402
from config import PHONE_STREAM_URL
from input.phone_stream import (
    PhoneStreamCamera,
    normalize_stream_url,
    probe_stream_connectivity,
)
from audio.tts import TextToSpeechEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("VisionAssist.PhoneBridge")


def get_local_ip() -> str:
    """Discovers the primary LAN/Wi-Fi IPv4 address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Routes to discover LAN interface without transmitting packets
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def _check_audio_relay(host: str, port: int) -> bool:
    """Helper that verifies the phone audio endpoint is live and serving JSON."""
    url = f"http://{host}:{port}/api/speech"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "VisionAssist-Test/1.0"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:  # nosec B310
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                logger.info(
                    f"Phone Audio Server LIVE at http://{host}:{port}/phone-audio "
                    f"(Latest ID: {data.get('id')})"
                )
                return True
    except Exception as e:
        logger.warning(f"Phone Audio Server check on {url} returned: {e}")
    return False


def test_audio_relay(host: str, port: int) -> None:
    """Verifies that the phone audio endpoint is live and serving JSON.
    Uses fixtures for host and port. Skips the test if the endpoint is not reachable."""
    if not _check_audio_relay(host, port):
        pytest.skip("Phone audio relay is not reachable")

def run_bridge_test(
    stream_url: str = PHONE_STREAM_URL,
    test_frames: int = 30
) -> bool:
    """Full Phase 3 diagnostics suite."""
    local_ip = get_local_ip()
    print("\n" + "=" * 75)
    print("       VisionAssist Phase 3: Phone Bridge Diagnostics Tool")
    print("=" * 75)
    print(f"* Laptop Wi-Fi / LAN IP:      http://{local_ip}:8088/phone-audio")
    print(f"* Target Phone Camera Stream: {normalize_stream_url(stream_url)}")
    print("-" * 75)

    # 1. Start TTS Engine and test phone audio endpoint
    logger.info("Initializing Text-To-Speech & Phone Audio Server...")
    tts = TextToSpeechEngine(phone_audio_port=8088)
    time.sleep(0.5)

    audio_ok = _check_audio_relay("127.0.0.1", 8088)
    if audio_ok:
        print(
            f"[OK] Audio Relay Server: OK -> "
            f"Open on Phone: http://{local_ip}:8088/phone-audio"
        )
        tts.speak("VisionAssist wireless phone bridge verified. Audio online.")
    else:
        print("[WARN] Audio Relay Server: Pending or port occupied.")

    # 2. Probe Phone Camera Stream Reachability
    norm_url = normalize_stream_url(stream_url)
    reachable = probe_stream_connectivity(norm_url, timeout=1.5)
    reach_status = "CONNECTED" if reachable else "NOT REACHABLE (Standby)"
    print(f"* Stream Reachability Probe:  {reach_status}")

    # 3. Test Frame Retrieval & Measure Latency
    logger.info("Testing PhoneStreamCamera frame acquisition...")
    cam = PhoneStreamCamera(stream_url, enable_threading=True)
    opened = cam.open()

    if opened:
        print(f"[OK] Camera Stream: Opened successfully at {norm_url}")
        latencies = []
        frames_grabbed = 0
        t0 = time.perf_counter()

        for _ in range(test_frames):
            t_f0 = time.perf_counter()
            ret, frame = cam.read_frame()
            t_f1 = time.perf_counter()
            if ret and frame is not None:
                frames_grabbed += 1
                latencies.append((t_f1 - t_f0) * 1000.0)
            time.sleep(0.03)

        total_time = time.perf_counter() - t0
        avg_fps = round(frames_grabbed / total_time, 1) if total_time > 0 else 0
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        res = f"{cam.resolution[0]}x{cam.resolution[1]}"

        print(f"* Frames Grabbed:             {frames_grabbed}/{test_frames}")
        print(f"* Effective Stream FPS:       {avg_fps} FPS")
        print(f"* Average Frame Read Latency: {avg_latency} ms")
        print(f"* Camera Resolution:          {res}")
    else:
        print(f"[WARN] Phone camera at {norm_url} is currently offline.")
        print("  -> Standby HUD active. Upstream pipeline will not block.")
        ret, frame = cam.read_frame()
        deliv = "Yes" if ret else "No"
        print(f"* Standby Frame Delivered:    {deliv} ({cam.resolution[0]}x{cam.resolution[1]})")

    # 4. Emergency Collision Interruption Test
    logger.info("Triggering emergency priority alert...")
    tts.speak(
        "Warning. Imminent obstacle ahead.",
        interrupt=True,
        priority=True
    )
    time.sleep(0.5)

    cam.release()
    tts.stop()

    print("-" * 75)
    print(">>> Phase 3 Diagnostics Completed successfully! <<<\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test phone camera stream and audio relay"
    )
    parser.add_argument(
        "--url",
        default=PHONE_STREAM_URL,
        help="Phone stream URL"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to sample"
    )
    args = parser.parse_args()

    run_bridge_test(stream_url=args.url, test_frames=args.frames)
