"""Small local web boundary for browser camera frames.

The browser sends JPEG frames to this process. OpenCV decodes them back to
BGR arrays before the existing ObjectDetector sees them, so the inference
path stays the same as the desktop and Streamlit applications.
"""

import base64
import binascii
import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from perception.object_detection import ObjectDetector
from config import ProductMode
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator


logger = logging.getLogger("VisionAssist.WebServer")
PROJECT_ROOT = Path(__file__).resolve().parent
WEB_PAGE = PROJECT_ROOT / "visionassist.html"
MEDIA_DIR = PROJECT_ROOT / "tests" / "test_scenes"
MEDIA_FILES = {
    "corridor_stairs.png",
    "room_sign_lab.png",
    "critical_obstacle.png",
}
MAX_FRAME_BYTES = 8 * 1024 * 1024


class VisionAssistHandler(BaseHTTPRequestHandler):
    """Serve the local page and a JSON detection endpoint."""

    detector = ObjectDetector()
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    response_generator = ResponseGenerator()

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_FRAME_BYTES:
            raise ValueError("Request body must be between 1 byte and 8 MB")
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    @staticmethod
    def _decode_frame(payload: Dict[str, Any]) -> np.ndarray:
        encoded = payload.get("image", "")
        if not isinstance(encoded, str):
            raise ValueError("image must be a base64 string")
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("image is not valid base64") from exc
        if not raw or len(raw) > MAX_FRAME_BYTES:
            raise ValueError("decoded image is empty or larger than 8 MB")
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("image is not a readable JPEG or PNG")
        return frame

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(200, {"ok": True, "model_live": self.detector.is_live})
            return
        if self.path.startswith("/media/"):
            media_name = self.path.removeprefix("/media/")
            if media_name not in MEDIA_FILES:
                self._send_json(404, {"error": "media not found"})
                return
            media_path = MEDIA_DIR / media_name
            try:
                body = media_path.read_bytes()
            except OSError:
                self._send_json(404, {"error": "media not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(media_name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/visionassist.html"):
            try:
                body = WEB_PAGE.read_bytes()
            except OSError:
                self._send_json(404, {"error": "visionassist.html not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/detect":
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = self._read_json()
            frame = self._decode_frame(payload)
            mode_value = payload.get("mode", ProductMode.OBSTACLE_AWARENESS.value)
            try:
                mode = ProductMode(mode_value)
            except ValueError:
                mode = ProductMode.OBSTACLE_AWARENESS
            detections = self.detector.detect(frame)
            height, width = frame.shape[:2]
            context_items = self.context_engine.process_detections(
                detections=detections,
                frame_shape=frame.shape,
                mode=mode,
            )
            prioritized = self.priority_engine.select_top_item(
                context_items=context_items,
                mode=mode,
            )
            speech_text = ""
            speech_severity = ""
            speech_object = ""
            if prioritized:
                speech_text = self.response_generator.generate(
                    prioritized,
                    language="en",
                ).get("text", "")
                speech_severity = prioritized.get("priority", "LOW")
                candidate = prioritized.get("candidate", {})
                speech_object = candidate.get("object", prioritized.get("type", "alert"))
            self._send_json(200, {
                "ok": True,
                "model_live": self.detector.is_live,
                "width": width,
                "height": height,
                "detections": detections,
                "context_items": context_items,
                "speech_text": speech_text,
                "speech_severity": speech_severity,
                "speech_object": speech_object,
                "mode": mode.value,
            })
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception:
            logger.exception("Web inference request failed")
            self._send_json(500, {"ok": False, "error": "inference request failed"})

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def run_web_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the local browser integration server until interrupted."""
    server = ThreadingHTTPServer((host, port), VisionAssistHandler)
    logger.info("VisionAssist web app listening at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Web server stopped by user")
    finally:
        server.server_close()
