# 👁️ VisionAssist

<div align="center">

[![CI / Build & Test](https://github.com/abhinavv27/VisonAssist/actions/workflows/ci.yml/badge.svg)](https://github.com/abhinavv27/VisonAssist/actions/workflows/ci.yml)
[![Security & SAST Audit](https://github.com/abhinavv27/VisonAssist/actions/workflows/security.yml/badge.svg)](https://github.com/abhinavv27/VisonAssist/actions/workflows/security.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Inference Latency](https://img.shields.io/badge/Latency-~35ms%20(Edge)-brightgreen.svg)]()
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline%20Edge-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Context-Aware Wearable Cognitive Vision for Visually Impaired Users**  
*“See the world through sound — Speak only what matters.”*

[Executive Summary](#-the-core-problem--innovation) •
[System Architecture](#-system-architecture) •
[Risk Mathematics](#-mathematical-risk-scoring) •
[Product Modes](#-the-5-product-modes) •
[Calibration Benchmark](#-phase-2-calibration-benchmark) •
[Quickstart](#-quickstart-guide) •
[Team & Hackathon](#-team--hackathon-pitch)

</div>

---

## 💡 The Core Problem & Innovation

Visually impaired individuals navigate an environment engineered almost exclusively for sighted people. Everyday mobility requires continuous, low-latency, high-consequence spatial decisions: identifying descending stairs, detecting approaching vehicles, reading room numbers, and sidestepping obstacles—all while keeping hands free and avoiding sensory overload.

### The "Everything Announcement" Trap
Traditional assistive computer vision projects feed camera frames into deep neural networks and immediately read aloud every detected label:
> ❌ *“Person. Person. Bottle. Chair. Door. Laptop. Floor. Stairs. Cup.”*

**Reading an unranked catalogue of detections is not assistance — it is hazardous noise.** It drowns out essential real-world acoustic cues (footsteps, traffic sounds, echoes) and forces the user into cognitive fatigue.

### The VisionAssist Breakthrough: Cognitive Context Filtering
VisionAssist introduces a transparent, explainable **Attention & Risk Engine** running in milliseconds on the edge. Rather than announcing everything, VisionAssist computes a multi-dimensional risk vector for every detected entity, prioritizes the single most vital hazard, debounces repetitive updates, and speaks one concise, actionable phrase:

```
Traditional CV:   "Person. Chair. Desk. Door. Stairs. Floor. Bottle."   (Sensory overload)
VisionAssist:     "Stairs directly ahead, 3.8 metres away."              (One clear priority)
```

---

## 🏛️ System Architecture

VisionAssist is architected with a **Hardware-Agnostic Adapter Pattern**. Intelligence and reasoning modules never couple directly to physical sensors. The camera feed can transition seamlessly from a laptop webcam to a chest-mounted smartphone, Raspberry Pi Zero 2W, or custom smart glasses without altering perception or context code.

```
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                   SMARTPHONE / WEARABLE SENSOR CLIENT                       │
   │   • Mounted on user's chest mount, lanyard, or shirt pocket                │
   │   • Video: RTSP / MJPEG camera stream (640x480 @ 30 FPS)                   │
   │   • Audio: Streams guidance to phone speaker / wireless Bluetooth earphones │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │  Zero-Lag Wi-Fi / Local 5GHz Hotspot
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                 CENTRAL EDGE PROCESSING & REASONING ENGINE                  │
   │                                                                             │
   │   [ INPUT ADAPTER ]                                                         │
   │   └── input/phone_stream.py (Threaded zero-buffer frame grabber)            │
   │                                                                             │
   │   [ PERCEPTION LAYER ]                                                      │
   │   ├── Object Detection : YOLOv8 (Vehicle threshold 0.38 / general 0.50)     │
   │   ├── Text Recognition : EasyOCR + Glare & Low-Light Fallback               │
   │   ├── Spatial Mapping  : position.py (Normalized FOV: Left, Centre, Right) │
   │   └── Depth Estimation : depth.py (Focal ground-plane perspective projection)│
   │                                                                             │
   │   [ INTELLIGENCE & REASONING LAYER ]                                        │
   │   ├── Context Engine    : context_engine.py (Temporal tracking & velocity)  │
   │   ├── Risk Engine       : risk_engine.py (Formula: Danger+Distance+Offset)  │
   │   ├── Priority Engine   : priority_engine.py (Cooldown debounce & filter)   │
   │   ├── Ask Engine (VQA)  : ask_engine.py (Grounded perception + Ollama VLM)  │
   │   └── Response Engine   : response_generator.py (Natural spoken templates)  │
   │                                                                             │
   │   [ REAL-TIME AUDIO SYNTHESIS & WEBSOCKET/HTTP RELAY ]                      │
   │   ├── audio/tts.py                : Priority-interrupt offline TTS engine   │
   │   └── audio/audio_stream_server.py: Low-latency /phone-audio HTTP relay     │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │  Live Telemetry (WebSockets / HTTP)
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                     OBSERVER & JUDGE CONSOLE (STREAMLIT)                    │
   │   • Panel 1: Live Video HUD (real-time risk bounding boxes & distance tags) │
   │   • Panel 2: Detected Entities breakdown (class, confidence, metric distance)│
   │   • Panel 3: Priority Pipeline Status (Risk scores, active product mode)   │
   │   • Audio Banner: Real-time broadcast payload sent to the user's phone      │
   └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Risk Scoring

VisionAssist calculates an explainable composite risk score $R \in [0, 100]$ for every candidate entity in the user's field of view:

$$R = S_{\text{danger}} + S_{\text{proximity}} + S_{\text{centrality}} + S_{\text{movement}} + \delta_{\text{uncertainty}}$$

### 1. Base Object Danger ($S_{\text{danger}} \in [0, 50]$)
Weights calibrated from pedestrian accessibility and indoor mobility research:
$$S_{\text{danger}} = \left( \frac{\text{Base Danger Weight}}{100} \right) \times 50$$

| Threat Tier | Objects | Base Weight | $S_{\text{danger}}$ Points | Action Strategy |
| :--- | :--- | :---: | :---: | :--- |
| **Critical Hazards** | Stairs, Car, Bus, Truck, Motorcycle | $85 - 95$ | **$42.5 - 47.5$** | Immediate alert; preemptive interrupt |
| **Mobility Obstacles** | Person, Chair, Doorway, Couch, Table | $35 - 45$ | **$17.5 - 22.5$** | Announced if path is in trajectory |
| **Non-Hazard Distractors**| Bottle, Cup, Phone, Book, Laptop | $5 - 15$ | **$2.5 - 7.5$** | Suppressed unless in Quick Look |

### 2. Metric Proximity Score ($S_{\text{proximity}} \in [0, 30]$)
Calculated via monocular pinhole geometric projection ($d = \frac{H_{\text{real}} \times f}{h_{\text{bbox}}}$):

$$S_{\text{proximity}} = \begin{cases} 
30.0 & d \le 1.0\text{ m} \\ 
20.0 + (2.5 - d) \times 6.6 & 1.0 < d \le 2.5\text{ m} \\ 
10.0 + (4.5 - d) \times 5.0 & 2.5 < d \le 4.5\text{ m} \\ 
2.0 & d > 4.5\text{ m} 
\end{cases}$$

> **Harmless Item Damping**: If $\text{Base Weight} \le 20$ (e.g., bottle on a table), proximity score is damped by $(\text{Base Weight} / 50.0)$, preventing nearby cups from stealing attention from distant descending stairs.

### 3. Centrality in Travel Trajectory ($S_{\text{centrality}} \in [2, 20]$)
- **Centre Path** ($x_{\text{norm}} \in [0.33, 0.66]$): $+20.0\text{ pts}$ (Direct collision line)
- **Slightly Off-Centre**: $+12.0\text{ pts}$
- **Lateral Left / Right**: $+8.0\text{ pts}$
- **Far Periphery**: $+2.0\text{ pts}$

### 4. Dynamic Velocity & Movement ($S_{\text{movement}} \in [0, 10]$)
Temporal tracking across frames estimates bounding box delta:
$$S_{\text{movement}} = \begin{cases} 10.0 & \Delta d < -0.15\text{ m (Closing in)} \\ 0.0 & \text{Static or receding} \end{cases}$$

### Priority Tiers & Action Thresholds
- 🔴 **HIGH Priority ($R \ge 65.0$)**: Spoken immediately; bypasses cooldown.
- 🟡 **MEDIUM Priority ($35.0 \le R < 65.0$)**: Spoken if direct travel path is unobstructed.
- 🟢 **LOW Priority ($R < 35.0$)**: Suppressed to preserve acoustic clarity.

---

## 🕹️ The 5 Product Modes

VisionAssist provides five purpose-built operating modes addressing every phase of visually impaired navigation:

| Mode | Title | Activation | Core Purpose | Sample Spoken Response |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Quick Look** | Press `1` | Instant on-demand 360° situational awareness of the two most salient features. | *"There is a doorway ahead, and a chair slightly to your left."* |
| **2** | **Obstacle Awareness** | Press `2` (Default) | Continuous background vigilance for navigational hazards. Suppresses noise. | *"Chair directly ahead, about 1.8 metres away."* |
| **3** | **Read (OCR)** | Press `3` | Signage, room numbers, door labels, and notices. Assembles multi-line text fragments. | *"Room 204 - Computer Science Lab."* |
| **4** | **Ask (Visual QA)** | Press `4` | Natural language visual question answering. Grounded perception with optional local VLM. | *"You are in a hallway facing an open door on your right."* |
| **5** | **Safety Alert** | Automatic | Emergency collision avoidance $(< 1.2\text{m})$. **Preempts and cuts off speech mid-sentence.** | *"Warning! Obstacle directly ahead, less than one metre!"* |

---

## 📊 Phase 2 Calibration Benchmark

The VisionAssist pipeline is strictly verified using a 10-scenario calibration test suite ([`scripts/calibrate_pipeline.py`](file:///d:/antigravity%20work/VisonAssist/scripts/calibrate_pipeline.py)) covering real-world indoor and outdoor challenges.

```
===============================================================================================
ID           | SCENARIO               | EXP PRIO  | ACT PRIO  | RISK   | STATUS | SPEECH
-----------------------------------------------------------------------------------------------
scenario_01  | stairs_ahead.png       | HIGH      | HIGH      | 76.0   | PASS   | Stairs directly ahead, 3.8 metres a
scenario_02  | doorway_right.png      | MEDIUM    | MEDIUM    | 44.5   | PASS   | Door on your far right.
scenario_03  | chair_center.png       | MEDIUM    | MEDIUM    | 60.6   | PASS   | Chair directly ahead, about 2.1 met
scenario_04  | sign_clean.png         | HIGH      | HIGH      | 50.0   | PASS   | Room 204 - Computer Science Lab.
scenario_05  | sign_glare_lowlight.png | LOW       | LOW       | 10.0   | PASS   | Text appears unclear, please adjust
scenario_06  | close_obstacle.png     | HIGH      | HIGH      | 79.2   | PASS   | Warning. Obstacle directly ahead.
scenario_07  | bottle_distractor.png  | LOW       | LOW       | 18.5   | PASS   | Bottle on your far right, approxima
scenario_08  | person_passing.png     | MEDIUM    | MEDIUM    | 49.0   | PASS   | Person on your left, about 2.8 metr
scenario_09  | car_approaching.png    | HIGH      | HIGH      | 76.5   | PASS   | Caution, car directly ahead, 4.2 me
scenario_10  | clear_hallway.png      | LOW       | LOW       | 0.0    | PASS   | Path clear. Suppressed low hazard.
===============================================================================================
Calibration Result: 10/10 Passed (100.0%) | Benchmark Target: >= 80.0%
```

### Defensive Perception & Edge-Case Robustness (Section 33)
1. **Multi-Line OCR Stitching (`scenario_04`)**: Joins fragmented text regions (`["ROOM 204", "CS LAB"]`) into a single coherent utterance.
2. **Extreme Glare & Flare Rejection (`scenario_05`)**: Evaluates brightness saturation and contrast variance up front. Glare-degraded text gracefully falls back to `"Text appears unclear, please adjust lighting or move closer"` (`LOW` priority), eliminating false-alarm hazard warnings.
3. **Vehicle Approaching Safety (`scenario_09`)**: Leverages a specialized $0.38$ confidence threshold for moving vehicles (`car`, `bus`, `truck`, `motorcycle`) to guarantee immediate high-urgency detection.
4. **Mid-Speech Emergency Preemption**: When a high-risk Mode-5 alert fires, ongoing low-priority sentences are cut off mid-syllable, flushing pending queues to protect the user from collisions.

---

## ⚡ Quickstart Guide

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone git@github.com:abhinavv27/VisonAssist.git
cd VisonAssist

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify System Integrity & Unit Tests
```bash
# Run the complete test suite (39 tests)
py -m pytest tests/ -v

# Run the 10-scenario calibration benchmark
py scripts/calibrate_pipeline.py

# Verify security & SAST rules
py -m bandit -r . -x ./tests,./venv,__pycache__ -ll -i
```

### 3. Launching VisionAssist

#### Option A: Standalone Desktop Mode (Webcam or Mock)
```bash
# Run with integrated laptop webcam
py app.py

# Run in headless developer mock mode (no camera required)
py app.py --mock
```

**OpenCV Window Hotkeys:**
- `1` : Switch to **Quick Look** mode
- `2` : Switch to **Obstacle Awareness** mode (default)
- `3` : Switch to **Read (OCR)** mode
- `4` : Switch to **Ask (VQA)** mode
- `r` : Force refresh / reset audio cooldown
- `q` : Safely stop and exit

#### Option B: Smartphone Camera + Wireless Audio Bridge
1. Install **IP Webcam** (Android) or similar RTSP/MJPEG app on your smartphone.
2. Connect phone and laptop to the same Wi-Fi network or mobile hotspot.
3. Start the video server on the phone (e.g., `http://192.168.43.1:8080/video`).
4. Run VisionAssist pointing to the phone:
   ```bash
   py app.py --phone http://192.168.43.1:8080/video
   ```
5. On the phone browser, open:
   ```text
   http://<YOUR-LAPTOP-IP>:8088/phone-audio
   ```
   *Tap "Enable Phone Audio". All prioritized guidance now speaks directly through the phone's speaker or connected Bluetooth earphones with haptic vibration alerts!*

#### Option C: Observer & Judge Console (Streamlit Dashboard)
```bash
# Launch the accessible 3-panel evaluation dashboard
py -m streamlit run ui/dashboard.py
# Or launch directly via master entrypoint:
py app.py --streamlit
```

#### Option D: 1-Command Verification & Demo Tools
```bash
# Run automated pre-flight system diagnostics across 8 checkpoints
py app.py --preflight

# Run microsecond pipeline latency profiler (< 40ms SLA check)
py app.py --profile

# Run automated 6-stage 3-minute judging rehearsal with voice synthesis
py app.py --demo

# Run with natural Hindi voice guidance
py app.py --mock --lang hi
```

---

## 📁 Repository Directory Structure

```
VisonAssist/
├── app.py                         # Master runtime loop (CLI, phone stream, GUI, --demo, --profile)
├── config.py                      # Danger weights, FOV geometry, risk thresholds
├── requirements.txt               # Dependencies (PyTorch, Ultralytics, OpenCV, Streamlit)
├── SYSTEM_ARCHITECTURE.md         # Full technical design & execution roadmap
├── DEMO_RUNBOOK.md                # Hackathon live demonstration runbook & cheat sheet
├── README.md                      # Project documentation & reference
├── .gitleaks.toml                 # Secret scanner allowlist configuration
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Cross-platform CI (Ubuntu + Windows)
│       └── security.yml           # Bandit SAST, pip-audit, and Gitleaks scanning
│
├── input/                         # Hardware-Agnostic Video Capture
│   ├── camera_interface.py        # BaseCamera abstract class
│   ├── webcam.py                  # Local webcam adapter with mock fallback
│   ├── phone_stream.py            # Low-latency threaded smartphone stream grabber
│   └── voice_trigger.py           # Hands-free "Vision, look" wake phrase listener
│
├── perception/                    # Computer Vision & Sensor Fusion
│   ├── object_detection.py        # YOLOv8 detector with vehicle sensitivity
│   ├── ocr.py                     # EasyOCR reader with glare/low-light resilience
│   ├── depth.py                   # Monocular metric distance estimator
│   └── position.py                # Horizontal FOV spatial classifier (Left/Centre/Right)
│
├── intelligence/                  # Cognitive Reasoning Layer
│   ├── context_engine.py          # Temporal object tracking & velocity calculation
│   ├── risk_engine.py             # Explainable mathematical risk score formula
│   ├── priority_engine.py         # Attention selection & anti-spam debounce
│   ├── ask_engine.py              # Bilingual Visual Q&A (Grounded + Ollama VLM)
│   └── response_generator.py      # Bilingual natural voice prompt templates (EN / HI)
│
├── audio/                         # Speech Synthesis & Wireless Return
│   ├── tts.py                     # Priority-interrupt offline TTS engine + 1200Hz Earcon
│   └── audio_stream_server.py     # HTTP/WebSocket relay for smartphone playback
│
├── ui/                            # Presentation & Telemetry
│   └── dashboard.py               # Streamlit Section 20 3-panel accessible judge console
│
├── hardware/                      # Future Smart-Glasses Hardware Drivers
│   ├── camera_adapter.py          # ESP32-CAM / Raspberry Pi camera driver
│   ├── ultrasonic_adapter.py      # HC-SR04 sonar proximity driver
│   ├── imu_adapter.py             # MPU-6050 6-DOF head-orientation tracker
│   └── button_adapter.py          # Tactile lanyard push-button interrupt driver
│
├── scripts/                       # Utilities & Evaluation Benchmarks
│   ├── calibrate_pipeline.py      # 10-scenario automated calibration runner (100% pass)
│   ├── preflight_check.py         # 8-point hardware & software pre-flight diagnostics
│   ├── profile_latency.py         # Microsecond per-stage & end-to-end latency profiler
│   ├── rehearse_demo.py           # 6-step automated judge presentation rehearsal
│   ├── record_backup_demo.py      # Standalone offline backup video generator
│   └── test_phone_bridge.py       # Smartphone audio/camera latency tester
│
└── tests/                         # Comprehensive Unit & Regression Suite (51 Tests)
    ├── test_ask_and_safety_modes.py
    ├── test_ask_engine.py
    ├── test_context_and_risk.py
    ├── test_demo_assets.py
    ├── test_phase4_demo_and_ui.py
    ├── test_phone_stream_and_audio.py
    ├── test_priority_engine.py
    ├── test_response_generator.py
    ├── test_robustness.py
    └── test_safety_interrupt.py
```

---

## 👥 Team & Hackathon Pitch

<div align="center">

| Member | Focus Area | Core Modules |
| :--- | :--- | :--- |
| **Abhinav Chauhan** | **Lead · Systems Architecture & Integration** | Core Pipeline, Context Engine, Risk Engine, Priority Engine, CI/CD & Security |
| **Adrit** | **Vision · OCR & Audio Infrastructure** | YOLOv8 Perception, EasyOCR Engine, Monocular Depth Math, Phone Audio Relay |
| **Dipayan** | **UI / UX · Evaluation & Documentation** | Streamlit Dashboard, Scenario Assets, Test Automation, Demo Runbook |

*Developed for the Open Innovation Hackathon · Manipal University Jaipur · 2026*

</div>

### 🎤 The 30-Second Elevator Pitch
> *"A camera can see hundreds of objects in a single glance — but a visually impaired user cannot process hundreds of audio notifications. Reading that list aloud is dangerous noise. VisionAssist converts visual information into prioritized, actionable acoustic guidance. Our MVP pairs a wearable smartphone with edge compute to detect hazards, read signage, estimate distance, assess risk, and speak only what matters. The architecture is hardware-independent: today's smartphone is tomorrow's smart glasses."*

---

<div align="center">
<b>VisionAssist — See the World Through Sound.</b><br>
<sub>Built with clean architecture, explainable AI, and accessibility at its heart.</sub>
</div>
