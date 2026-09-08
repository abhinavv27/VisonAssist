# VisionAssist — System Architecture & Execution Master Plan

> **Context-Aware Wearable Vision Assistance for Visually Impaired Users**  
> *"See the world through sound."*  
> **Master Technical Design Document & 24-Hour Hackathon Execution Roadmap**  
> **Manipal University Jaipur · Open Innovation Hackathon · 2026**

---

## 1. Executive Summary & Problem Space

### 1.1 The Core Problem
Visually impaired people navigate an environment engineered almost exclusively for sighted individuals. Everyday mobility demands continuous, low-bandwidth, high-consequence spatial decisions:
- Identifying obstacles (stairs, curbs, low-hanging signs, vehicles) before physical contact.
- Locating doorways, seating, and navigation corridors in unfamiliar spaces.
- Reading room numbers, labels, signs, and instructions.
- Avoiding sensory overload while keeping hands free.

### 1.2 The Innovation Gap
Assistive computer-vision prototypes frequently fail because of two critical anti-patterns:
1. **The "Everything Announcement" Trap**: Traditional computer vision generates an unranked list of detections (`"Person. Bottle. Chair. Door. Stairs."`). Reading this aloud is not assistance — **it is noise** that drowns out essential acoustic navigation cues.
2. **Hardware Lock-in**: Many systems bind their logic to a single proprietary PCB, camera module, or expensive 3D-printed enclosure, rendering them brittle, expensive, and unmaintainable.

### 1.3 The VisionAssist Thesis
> **"The intelligence is the product. The hardware is an interchangeable interface."**

VisionAssist is a **modular perception-and-audio-guidance platform**. It introduces a transparent, rule-based **Attention and Risk Engine** that intercepts raw detections, computes real-world proximity and centrality, scores collision hazards, and speaks **only the single most actionable guidance sentence** required at any given moment.

---

## 2. End-to-End System Architecture

```
                                 [ PHYSICAL WORLD ]
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: INPUT & STREAMING ADAPTERS (`input/`)                                 │
│  - OpenCVWebcam (Laptop camera / USB webcam)                                   │
│  - PhoneStreamCamera (Smartphone MJPEG / RTSP via IP camera)                   │
│  - BaseCamera (Hardware-agnostic abstract interface)                           │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ BGR Frame (640x480)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: PERCEPTION ENGINE (`perception/`)                                      │
│  - Object Detection: YOLOv8n (BBoxes, Class Labels, Confidence)                │
│  - OCR Extraction: EasyOCR (Text bounding boxes, confidence, text string)      │
│  - Spatial Classification: `position.py` (Left / Centre / Right)               │
│  - Proximity / Depth: `depth.py` (Geometric pinhole projection + ground-plane)  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Detections + Spatial/Distance Metadata
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: CONTEXT & RISK ENGINE (`intelligence/context_engine.py`, `risk_engine.py`)│
│  - Assembles per-object context tuples                                          │
│  - Calculates Risk Score: proximity + centrality + base danger + uncertainty   │
│  - Categorizes priority tiers: HIGH (>=65) | MEDIUM (35-64) | LOW (<35)        │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Sorted Context Entities
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: PRIORITY & ATTENTION ENGINE (`intelligence/priority_engine.py`)       │
│  - Selects top actionable item based on Product Mode                           │
│  - Suppresses low-priority background clutter                                  │
│  - Enforces time-based debounce cooldowns & urgent hazard overrides            │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Single Prioritized Event
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: NATURAL RESPONSE GENERATOR (`intelligence/response_generator.py`)     │
│  - Converts prioritized context into natural template sentence                 │
│  - e.g., "Stairs ahead, approximately 2 metres away."                           │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Speech Text String: { "text": str }
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: AUDIO & SPEECH ENGINE (`audio/tts.py`)                                 │
│  - Asynchronous background worker queue (never blocks video loop)               │
│  - Native Windows SAPI voice synthesizer / pyttsx3 (100% offline, zero latency)│
│  - Interrupt handling for critical safety alerts                                │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Audio Waves
                                         ▼
                              [ USER'S EARPHONES / SPEAKER ]
```

---

## 3. Team Interface Contracts (The Technical Boundary)

To eliminate integration friction and allow parallel work, each module communicates via strict contracts:

| Interface | Producer | Consumer | Data Shape / Contract |
| :--- | :--- | :--- | :--- |
| **Frame Capture** | `input.BaseCamera` | `perception` | `(success: bool, frame: np.ndarray[H, W, 3])` |
| **Object Detection** | `perception.ObjectDetector` | `intelligence.ContextEngine` | `List[{ "object": str, "confidence": float, "x_center": float, "y_center": float, "bbox": [x1, y1, x2, y2] }]` |
| **OCR Text** | `perception.OCRReader` | `intelligence.PriorityEngine` | `List[{ "text": str, "confidence": float, "bbox": [x1, y1, x2, y2] }]` |
| **Depth / Proximity** | `perception.depth` | `intelligence.ContextEngine` | `(distance_meters: float, proximity_category: str)` |
| **Context Entity** | `intelligence.ContextEngine` | `intelligence.PriorityEngine` | `List[{ "object": str, "confidence": float, "position": str, "position_desc": str, "distance": float, "risk_score": float, "priority": str, "bbox": list }]` |
| **Priority Selection**| `intelligence.PriorityEngine` | `intelligence.ResponseGenerator` | `{ "type": "obstacle"\|"ocr"\|"quick_look", "candidate": dict, "is_critical": bool, "priority": str }` or `None` |
| **Response Text** | `intelligence.ResponseGenerator` | `audio.TextToSpeechEngine` | `{ "text": str }` |
| **Audio Output** | `audio.TextToSpeechEngine` | OS Audio Hardware | `speak(text: str, interrupt: bool = False)` |

---

## 4. Mathematical Risk & Spatial Formulations

### 4.1 The Risk Formula
$$\text{Risk Score} = S_{\text{danger}} + S_{\text{proximity}} + S_{\text{centrality}} + S_{\text{movement}} + \delta_{\text{uncertainty}}$$

1. **Base Object Danger ($S_{\text{danger}} \in [0, 50]$)**:
   $$S_{\text{danger}} = \left( \frac{\text{Base Danger (0-100)}}{100} \right) \times 50$$
   - *Stairs*: 85 &rarr; 42.5 pts
   - *Car / Bus / Truck*: 90–95 &rarr; 45.0–47.5 pts
   - *Chair / Obstacle*: 40 &rarr; 20.0 pts
   - *Door*: 35 &rarr; 17.5 pts
   - *Bottle / Phone / Book*: 5–10 &rarr; 2.5–5.0 pts

2. **Proximity Score ($S_{\text{proximity}} \in [0, 30]$)**:
   $$S_{\text{proximity}} = \begin{cases} 
   30.0 & d \le 1.0\text{ m} \\ 
   20.0 + (2.5 - d) \times 6.6 & 1.0 < d \le 2.5\text{ m} \\ 
   10.0 + (4.5 - d) \times 5.0 & 2.5 < d \le 4.5\text{ m} \\ 
   2.0 & d > 4.5\text{ m} 
   \end{cases}$$
   *Non-hazard scaling*: If $\text{Base Danger} \le 20$ (e.g. handheld items), $S_{\text{proximity}}$ is scaled by $(\text{Base Danger} / 50.0)$ to prevent harmless objects from triggering false alarms.

3. **Centrality Score ($S_{\text{centrality}} \in [2, 20]$)**:
   - `Centre` (user path of travel): $+20.0\text{ pts}$
   - `Slightly Left / Right`: $+12.0\text{ pts}$
   - `Left / Right`: $+8.0\text{ pts}$
   - `Far Left / Far Right`: $+2.0\text{ pts}$

4. **Priority Tier Classification**:
   - **`HIGH`** ($\ge 65.0$): Spoken immediately; interrupts ongoing audio if critical.
   - **`MEDIUM`** ($35.0 - 64.9$): Spoken if no higher-risk object exists.
   - **`LOW`** ($< 35.0$): Suppressed during continuous obstacle awareness to eliminate noise.

---

## 5. Product Modes & User Experience

| Mode | Trigger | System Behavior | Spoken Output Example |
| :--- | :--- | :--- | :--- |
| **Mode 1: Quick Look** | Key press `'1'` or UI button | Captures current frame and summarizes the two most salient items. | *"There is a doorway ahead, and a chair slightly to your right."* |
| **Mode 2: Obstacle Awareness** | Default continuous mode | Background monitoring of navigational path. Suppresses low threats and announces high/medium hazards with 3.5s debounce. | *"Stairs ahead, approximately 2 metres away."* |
| **Mode 3: Read** | Key press `'3'` or UI selector | Focuses on text recognition (room numbers, labels, signs). | *"Computer Science Lab, Room 204."* |
| **Mode 4: Ask** | Command / Query | Grounded visual question answering for specific queries (*"Where is the door?"*). | *"The door is to your right, about 3 metres away."* |
| **Mode 5: Safety Alert** | Automatic proximity trigger | High-urgency collision override if an obstacle is within 1.0m directly ahead. | *"Warning. Obstacle directly ahead, less than one metre."* |

---

## 6. Team Role Division & Ownership

```
                            ┌────────────────────────────────────────┐
                            │            ABHINAV CHAUHAN             │
                            │    Lead / Core Systems + Integration   │
                            └───────────────────┬────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
┌─────────────────────────────────┐                           ┌─────────────────────────────────┐
│              ADRIT              │                           │             DIPAYAN             │
│       Vision / OCR / Audio      │                           │    UI / Testing / Documentation │
└─────────────────────────────────┘                           └─────────────────────────────────┘
```

### 6.1 Abhinav Chauhan (Lead / Core Systems + Integration)
- **Direct Code Ownership**:
  - `intelligence/risk_engine.py` (Mathematical risk scoring formula)
  - `intelligence/priority_engine.py` (Attention filtering, debounce logic, top-item selector)
  - `intelligence/context_engine.py` (Perception-to-intelligence aggregation)
  - `input/camera_interface.py` & `input/webcam.py` (Camera pipeline)
  - `app.py` (Main integration loop, CLI runner, hotkey event handling)
  - `config.py` (Central thresholds, weights, coordinate boundaries)
- **Primary Responsibilities**:
  - Maintain repository integrity and branch merges on GitHub.
  - Ensure all modules conform to the interface contracts.
  - Coordinate end-to-end latency benchmarks (< 50ms per loop).
  - Deliver the technical pitch and architecture explanation to judges.

### 6.2 Adrit (Vision / OCR / Audio)
- **Direct Code Ownership**:
  - `perception/object_detection.py` (YOLOv8n setup, model inference, bounding box extraction)
  - `perception/ocr.py` (EasyOCR / Tesseract pipeline, text cleaning, signage parsing)
  - `perception/depth.py` (Monocular distance math, MiDaS integration if compute permits)
  - `perception/position.py` (Left/Centre/Right spatial logic)
  - `audio/tts.py` (Speech synthesis queue, Windows SAPI / pyttsx3 tuning, voice clarity)
- **Primary Responsibilities**:
  - Benchmark YOLOv8n inference on laptop CPU (target 20-30 FPS).
  - Tune OCR confidence filters to avoid reading background noise or texture artifacts.
  - Ensure TTS speech synthesis is completely non-blocking.

### 6.3 Dipayan (UI / Testing / Documentation)
- **Direct Code Ownership**:
  - `ui/dashboard.py` (Streamlit 3-panel accessible dashboard)
  - `utils/drawing.py` (Visual overlays, danger bounding boxes, crosshairs, audio banner)
  - `tests/` (`test_context_and_risk.py`, `test_priority_engine.py`, `test_response_generator.py`)
  - `README.md` & Presentation assets (Slide decks, demo test cases, judging scripts)
- **Primary Responsibilities**:
  - Curate and test physical demonstration scenarios (chair obstacle, room sign, walking towards stairs).
  - Record backup demo videos and capture high-resolution screenshots.
  - Ensure the Streamlit dashboard renders cleanly during live demonstrations.

---

## 7. The 24-Hour Phased Execution Roadmap

```
Hour:   0    1          4          8          12         16         20         22     24
        ├────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────┤
Phase:  [ P1 ] [   P2   ] [   P3   ] [   P4   ] [   P5   ] [   P6   ] [   P7   ] [ P8  ]
        Setup    Vision    Risk/TTS   Integrate  Scenarios   UI/Demo    QA/Test   Rehearse
```

### Phase 1: Environment & Repository Setup (Hour 0 – Hour 1)
- **Goal**: Scope freeze, GitHub repo setup, dependencies installed, test baseline working.
- **Status**: ✅ **COMPLETED (Boilerplate MVP v0 live on `main` branch)**.
- **Checklist**:
  - [x] Cloned empty repository and scaffolded modular structure.
  - [x] Set up `config.py`, `requirements.txt`, `.gitignore`.
  - [x] Verified unit tests passing (`12 passed in 0.18s`).
  - [x] Initial commit pushed to `git@github.com:abhinavv27/VisonAssist.git`.

---

### Phase 2: Core Vision & Capture Optimization (Hour 1 – Hour 4)
- **Goal**: Live camera input and real-time YOLOv8n inference running on laptop hardware.
- **Tasks by Role**:
  - **Abhinav**: Validate `OpenCVWebcam` on the physical laptop camera. Benchmark loop latency. Verify smooth fallback when camera is occupied.
  - **Adrit**: Verify `ultralytics` imports cleanly; test `yolov8n.pt` on physical objects (chair, person, bottle, doorway). Calibrate bounding box coordinates and verify `position.py` classifications.
  - **Dipayan**: Create mock test scenes (printed "ROOM 204" sign, chair in hallway) and write pytest test cases for edge-case coordinates.

---

### Phase 3: Context Engine & Audio Pipeline Calibration (Hour 4 – Hour 8)
- **Goal**: Connect perception outputs to the Risk Engine and establish non-blocking audio guidance.
- **Tasks by Role**:
  - **Abhinav**: Connect `ContextEngine` to `RiskEngine`. Fine-tune base danger weights for common hackathon venue items (chairs, bags, doors, people).
  - **Adrit**: Connect `audio/tts.py` to Windows SAPI. Verify speech rate (175 wpm) and ensure non-blocking queue behavior prevents any video stuttering. Integrate EasyOCR for reading test signs.
  - **Dipayan**: Build out the Streamlit dashboard components: Live View container, Objects Detected list, Priority cards, and the Audio Banner.

---

### Phase 4: Full System Integration — The Closed Loop (Hour 8 – Hour 12)
- **Goal**: Complete end-to-end loop: `Camera -> YOLO/OCR -> Context -> Priority -> Speech`.
- **Tasks by Role**:
  - **Abhinav**: Integrate all layers inside `app.py`. Test live switching between Mode 1 (Quick Look) and Mode 2 (Obstacle Awareness).
  - **Adrit**: Run combined YOLO + OCR test on a frame containing both an obstacle and a sign. Ensure priority engine suppresses OCR text when a high-risk obstacle is in path.
  - **Dipayan**: Connect Streamlit session state to live camera and intelligence engines so dashboard updates synchronously with voice output.

---

### Phase 5: Danger Alerts, Scenarios & Edge Cases (Hour 12 – Hour 16)
- **Goal**: Hardening, debouncing, and handling sudden obstacles and false positives.
- **Tasks by Role**:
  - **Abhinav**: Implement collision interrupt logic in `PriorityEngine` (if an obstacle appears within 1.0m, interrupt ongoing speech and speak immediately).
  - **Adrit**: Test distance estimator with varied lighting and angles. Add bounds checking to prevent division by zero on degenerate bounding boxes.
  - **Dipayan**: Stress-test the system with rapid movements, partial occlusions, and multi-object clusters. Document edge cases.

---

### Phase 6: UI Polish, Documentation & Demo Script (Hour 16 – Hour 20)
- **Goal**: Prepare judge-facing visual presentation, clean up dashboard layout, and finalize scripts.
- **Tasks by Role**:
  - **Abhinav**: Write architectural summary slides and review README technical sections.
  - **Adrit**: Test audio output over laptop speakers and Bluetooth earphones. Verify speech clarity in noisy hackathon environment.
  - **Dipayan**: Polish Streamlit CSS styling, capture high-res screenshots, record 30-second fallback demo video clips.

---

### Phase 7: Full Integration Testing & Freezing (Hour 20 – Hour 22)
- **Goal**: Total feature freeze. End-to-end testing of the exact demo sequence.
- **Tasks by Role**:
  - **Entire Team**: NO NEW FEATURES.
  - Run `py -m pytest tests/` to ensure zero regressions.
  - Execute the exact 6-step judging demo sequence 5 times consecutively.

---

### Phase 8: Demo Rehearsal (Hour 22 – Hour 24)
- **Goal**: Rehearse 30-second, 60-second, and 2-minute pitch narrations.
- **Tasks by Role**:
  - **Abhinav**: Lead live demo narration and judge technical defense.
  - **Adrit**: Manage physical camera positioning and test props.
  - **Dipayan**: Monitor dashboard display and backup presentation slides.

---

## 8. Abhinav's Action Guide (Lead / Core Systems)

As the project lead, your job is to guide the team, safeguard the architecture, and ensure the closed-loop runs with rock-solid stability.

### Step 1: Initialize Local Development
```bash
cd "d:\antigravity work\VisonAssist"
git pull origin main
py -m pip install -r requirements.txt
py -m pytest tests/
```

### Step 2: Supervise Module Interfaces
- Ensure Adrit adheres to `perception/object_detection.py` returning:
  `{ "object": str, "confidence": float, "x_center": float, "y_center": float, "bbox": [x1, y1, x2, y2] }`.
- Ensure Dipayan's dashboard consumes `context_items` without mutating the core intelligence state.

### Step 3: Calibrate the Attention Formula
- Open `config.py` and adjust danger ratings according to your physical demo room:
  - If demo space has many chairs: test `chair: 40`.
  - If demonstrating with a person walking in front: test `person: 45`.
- Verify that minor items (bottles, notebooks, phones) remain **suppressed** so judges see the system's filtering intelligence in action.

### Step 4: Run the CLI Live Loop
```bash
py app.py
```
- Test key `'1'` (Quick Look), `'2'` (Obstacle Awareness), `'3'` (Read OCR), and `'r'` (Reset cooldown).

---

## 9. Adrit's Action Guide (Vision / OCR / Audio)

### Step 1: YOLOv8 Setup & Verification
```bash
py -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLOv8 ready!')"
```
- Verify that `perception/object_detection.py` accurately detects common indoor objects.

### Step 2: OCR Setup & Calibration
```bash
py -c "import easyocr; reader = easyocr.Reader(['en'], gpu=False); print('EasyOCR ready!')"
```
- Point camera at printed room signs. Tune confidence threshold in `perception/ocr.py` so only clear text is extracted.

### Step 3: Speech Synthesis Tuning
- Check `audio/tts.py`. On Windows, the native SAPI voice is crisp, offline, and zero-latency.
- Test volume and rate settings (`config.TTS_RATE = 175`).

---

## 10. Dipayan's Action Guide (UI / Testing / Documentation)

### Step 1: Run the Streamlit Dashboard
```bash
py -m streamlit run ui/dashboard.py
```
- Verify that the 3 panels (Live View, Objects, Priority Queue) and the teal Current Audio banner display smoothly.

### Step 2: Prepare the Demo Test Scenarios
Curate three physical props for the live judging demo:
1. **The Obstacle**: A chair placed 1.5 to 2.0 metres in front of the camera.
2. **The Room Sign**: A printed sheet with **`ROOM 204 — COMPUTER SCIENCE LAB`**.
3. **The Low-Risk Distractor**: A water bottle placed on the desk to demonstrate intelligent suppression.

### Step 3: Record Backup Video Demos
- Use OBS or screen recorder to record a 60-second video demonstrating:
  - The live feed detecting chair & stairs.
  - The voice speaking only the prioritized obstacle.
  - The OCR reading the room sign aloud.

---

## 11. Judging Demo Script (2–3 Minutes)

### Step 1: Opening Framing (30 Seconds)
> *"Judges, a camera can see hundreds of objects at once — but a visually impaired person cannot listen to hundreds of notifications. Traditional computer vision gives you a flat list: person, bottle, chair, door, stairs. That is not assistance; it is overwhelming noise.*  
> *With VisionAssist, we built a context-aware perception and audio platform. Our core innovation is an Attention and Risk Engine that decides what the user actually needs to hear, and speaks only that."*

### Step 2: Detection vs. Prioritization Contrast (30 Seconds)
- Point camera at a scene containing a person, a chair, and a water bottle.
- Show the dashboard: all three are detected in Panel 2.
- Point to Panel 3: the bottle is categorized as **LOW (Suppressed)**; the chair is **MEDIUM**.
- System speaks: **`"Chair ahead, approximately 1.5 metres."`**
- Emphasize to judges: *"Notice how the bottle was completely suppressed. The user only heard about the obstacle in their path."*

### Step 3: Sudden Obstacle & Safety Alert (30 Seconds)
- Move the chair closer (< 1.0m directly ahead).
- System immediately interrupts: **`"Warning. Obstacle directly ahead, less than one metre."`**
- Explain: *"Proximity and centrality elevated the risk score past 65, triggering an immediate safety warning."*

### Step 4: Text & Signage Reading (30 Seconds)
- Switch to Mode 3 (Read) and hold up the **`"ROOM 204 — COMPUTER SCIENCE LAB"`** sign.
- System reads: **`"Computer Science Lab, Room 204."`**

### Step 5: The Closing Architectural Pitch (30 Seconds)
> *"Today, this runs live on a standard laptop webcam and CPU with zero cloud dependencies. But our architecture is hardware-agnostic by design. The same decision engine can move to a smartphone, a Raspberry Pi, or dedicated smart glasses without rewriting the reasoning layers. VisionAssist makes computer vision truly useful, context-aware, and accessible."*

---

## 12. Anticipated Judge Q&A Strategy

| Question | Winning Answer |
| :--- | :--- |
| **"What is actually innovative here? Isn't object detection already solved?"** | *"Object detection is a solved commodity — that's why we don't claim to invent it. Our innovation is the **decision, attention, and risk engine** built on top of detection. Sighted people don't process every pixel; their brain focuses on relevant hazards. We created the reasoning layer that converts raw detection into actionable, prioritized audio guidance without cognitive overload."* |
| **"Why not just use ChatGPT or Gemini with a camera?"** | *"Cloud LLMs require constant internet, incur 2–4 seconds of latency, and offer generic narrative descriptions on request rather than continuous, millisecond-level collision guidance. VisionAssist runs **100% offline**, operates with sub-50ms latency, and acts as a continuous real-time guardian."* |
| **"Why don't you have physical smart glasses built today?"** | *"In a 24-hour hackathon, soldering a custom headset risks building an unreliable hardware gimmick with an empty software core. We deliberately prioritized building a robust, fully-tested **hardware-agnostic platform**. The software interfaces are decoupled so swapping the webcam for an ESP32 or Raspberry Pi requires zero changes to the intelligence layer."* |
| **"What happens if the AI model makes a mistake?"** | *"VisionAssist employs conservative thresholds, explicit priority tiers, and confidence gates. Low-confidence detections are downweighted. Furthermore, we explicitly state that this is an assistive technology prototype, not an autonomous mobility system, ensuring responsible and safe deployment."* |

---

## 13. Hardware Evolution Roadmap

```
[ V1: 24-Hour Hackathon MVP ]
  Laptop Webcam + Laptop CPU + Offline SAPI TTS + Streamlit Dashboard
                           │
                           ▼
[ V2: Phone-Assisted Wearable ]
  Smartphone camera via RTSP/MJPEG + Laptop Edge Compute + Bluetooth Audio
                           │
                           ▼
[ V3: Raspberry Pi Wearable ]
  Raspberry Pi Zero 2W / 4 + CSI Camera + Ultrasonic Distance Sensor + Bone-Conduction Audio
                           │
                           ▼
[ V4: Dedicated Smart Glasses ]
  ESP32-S3 / Edge AI Processor + Dual Camera / ToF Depth + IMU + Lightweight 3D Frame
```

---

## 14. Repository Quick Reference

- **Repository**: `git@github.com:abhinavv27/VisonAssist.git`
- **Track**: Open Innovation
- **Primary Language**: Python 3.10+
- **Key Commands**:
  - Run Unit Tests: `py -m pytest tests/`
  - Run Live OpenCV Loop: `py app.py`
  - Run Mock Stream: `py app.py --mock`
  - Run Streamlit UI: `py app.py --streamlit`
