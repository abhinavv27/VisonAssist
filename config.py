"""
VisionAssist - Central Configuration
====================================
Defines system thresholds, base danger weights, spatial coordinates,
and operating mode settings for the VisionAssist platform.
"""

from enum import Enum
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Application Modes
class ProductMode(str, Enum):
    QUICK_LOOK = "quick_look"           # Mode 1: On-demand snapshot of scene
    OBSTACLE_AWARENESS = "obstacle"     # Mode 2: Continuous background obstacle guidance
    READ = "read"                       # Mode 3: Text & signage reading via OCR
    ASK = "ask"                         # Mode 4: Conversational Q&A on visual frame
    SAFETY_ALERT = "safety_alert"       # Mode 5: Urgent collision warning interruption

# Default Mode
DEFAULT_MODE = ProductMode.OBSTACLE_AWARENESS

# Camera & Phone Streaming Settings
DEFAULT_CAMERA_INDEX = 0
PHONE_STREAM_URL = "http://192.168.43.1:8080/video"  # Default IP Webcam / RTSP stream address
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# Spatial Position Thresholds (Normalized 0.0 to 1.0)
HORIZONTAL_LEFT_BOUNDARY = 0.33
HORIZONTAL_RIGHT_BOUNDARY = 0.66

# Risk Engine Danger Weights (Base Danger: 0 - 100)
# Calibrated for pedestrian & indoor mobility safety
OBJECT_DANGER_WEIGHTS = {
    # High Threat / Critical Mobility Hazards (70 - 100)
    "stairs": 85,
    "car": 90,
    "bus": 95,
    "truck": 95,
    "motorcycle": 85,
    "bicycle": 70,
    "traffic light": 75,
    "hazard": 75,
    "obstacle": 60,
    "fire hydrant": 50,
    "stop sign": 60,
    
    # Moderate Mobility Obstacles (30 - 65)
    "person": 45,
    "chair": 36,
    "couch": 45,
    "bench": 45,
    "table": 40,
    "dining table": 40,
    "bed": 35,
    "door": 35,
    "pole": 60,
    "dog": 40,
    
    # Low Threat / Suppressed by Default (0 - 25)
    "bottle": 10,
    "cup": 10,
    "cell phone": 5,
    "laptop": 15,
    "mouse": 5,
    "keyboard": 5,
    "book": 10,
    "clock": 5,
    "vase": 15,
    "backpack": 20,
    "handbag": 15,
    "umbrella": 20,
}
DEFAULT_OBJECT_DANGER = 25

# Risk Score Thresholds
RISK_THRESHOLD_HIGH = 65.0      # Immediate spoken alert (Stairs, Car, critical hazards)
RISK_THRESHOLD_MEDIUM = 35.0    # Spoken if no high-risk object present (Chair, Door)
RISK_THRESHOLD_LOW = 20.0       # Suppressed unless in Quick Look mode

# Priority Engine Debounce / Cooldown (seconds)
ANNOUNCEMENT_COOLDOWN_SECONDS = 3.5
CRITICAL_SAFETY_DISTANCE_METERS = 1.2

# Text-To-Speech Settings
TTS_RATE = 175  # Words per minute
TTS_VOLUME = 1.0
VOICE_NAME_HINT = "David"  # Or "Zira" / default system voice

# YOLO Model Settings
YOLO_MODEL_NAME = "yolov8m.pt"  # upgraded model for higher accuracy
YOLO_CONFIDENCE_THRESHOLD = 0.30  # final confidence for general objects
YOLO_INFERENCE_CONFIDENCE = 0.25  # preserve low-confidence vehicle candidates for filtering
YOLO_IMAGE_SIZE = 640
DETECTION_INTERVAL_FRAMES = 2  # run YOLO every other camera frame for smoother display FPS
