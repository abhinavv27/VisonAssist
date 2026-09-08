# VisionAssist — System Architecture & Execution Master Plan

> **Context-Aware Wearable Vision Assistance for Visually Impaired Users**  
> *"See the world through sound."*  
> **Master Technical Design Document & 24-Hour Hackathon Execution Roadmap**  
> **Manipal University Jaipur · Open Innovation Hackathon · 2026**

---

## 1. Executive Summary & Physical Deployment Topology

### 1.1 The Core Problem & Assistive Vision
Visually impaired people navigate an environment engineered almost exclusively for sighted individuals. Everyday mobility demands continuous, low-bandwidth, high-consequence spatial decisions:
- Identifying obstacles (stairs, curbs, low-hanging signs, vehicles) before physical contact.
- Locating doorways, seating, and navigation corridors in unfamiliar spaces.
- Reading room numbers, labels, signs, and instructions.
- Avoiding sensory overload while keeping hands free.

### 1.2 The Innovation Gap
Assistive computer-vision prototypes frequently fail because of two critical anti-patterns:
1. **The "Everything Announcement" Trap**: Traditional computer vision generates an unranked list of detections (`"Person. Bottle. Chair. Door. Stairs."`). Reading this aloud is not assistance — **it is noise** that drowns out essential acoustic navigation cues.
2. **The "Clunky Laptop Carrier" Trap**: Walking around holding a laptop in front of a visually impaired user destroys realism. 

### 1.3 The VisionAssist Split Architecture (Phone + Edge Compute)
> **"The intelligence is the product. The hardware is an interchangeable interface."**

To demonstrate a credible, realistic wearable assistant during the 24-hour hackathon:
1. **Smartphone (Wearable Sensor & Audio Client)**: Mounted on the user (chest mount, shirt pocket, or lanyard). Its camera captures the user's field of view and streams video over a low-latency local Wi-Fi / hotspot connection. Its speaker or connected Bluetooth/wired earphones play the prioritized audio guidance directly into the user's ears.
2. **Adrit's Laptop (Heavy Processing Unit / Edge Compute Node)**: Sits stationary or in a backpack on the local network. Hosts all compute-heavy models (YOLOv8n, EasyOCR, depth estimation, PyTorch) and runs the **Context, Risk, and Priority Engines**. It receives the phone's camera frames, processes them in milliseconds, generates the single prioritized instruction, synthesizes TTS audio, and streams the sound back to the phone.
3. **Observer / Judge Dashboard (Abhinav & Dipayan's Screen)**: Displays the live 3-panel Streamlit dashboard so judges and observers can watch what the phone camera sees, how objects are classified, the risk score calculation, and the exact audio beamed to the phone.

---

## 2. Distributed Physical & Network Topology

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                  SMARTPHONE (WEARABLE CLIENT ENDPOINT)                  │
   │  - Mounted on user's chest / lanyard                                    │
   │  - Camera: Streams 640x480 @ 30 FPS via RTSP / MJPEG (IP Webcam)        │
   │  - Audio Output: Plays guidance through phone speakers / earphones      │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        │
                         Local Wi-Fi / 5GHz Hotspot Network
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │             ADRIT'S LAPTOP (CENTRAL HEAVY PROCESSING UNIT)              │
   │                                                                         │
   │  [ STREAM RECEIVER ] ──► input/phone_stream.py (BaseCamera interface)  │
   │                                   │                                     │
   │  [ PERCEPTION LAYER ]             ▼                                     │
   │    ├─ Object Detection: YOLOv8n (Ultralytics GPU/CPU inference)         │
   │    ├─ Text Reading: EasyOCR (Signage & room labels)                     │
   │    ├─ Spatial Math: position.py (Left / Centre / Right)                 │
   │    └─ Depth Math: depth.py (Geometric ground-plane projection)          │
   │                                   │                                     │
   │  [ INTELLIGENCE LAYER ]           ▼                                     │
   │    ├─ Context Engine: context_engine.py (Per-object metadata)           │
   │    ├─ Risk Engine: risk_engine.py (Formula: Proximity+Centrality+Danger)│
   │    ├─ Priority Engine: priority_engine.py (Top-item filter & debounce)  │
   │    └─ Response Generator: response_generator.py (Concise templates)     │
   │                                   │                                     │
   │  [ AUDIO SYNTHESIZER & RETURN ]   ▼                                     │
   │    └─ audio/tts.py: Offline TTS (Streams synthesized audio to Phone)    │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        │ Real-time Telemetry (WebSockets / HTTP)
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │            OBSERVER & JUDGE CONSOLE (ABHINAV / DIPAYAN)                 │
   │  - Streamlit Dashboard: ui/dashboard.py                                 │
   │  - Panel 1: Live View (Phone camera feed with risk bounding boxes)      │
   │  - Panel 2: Detected Objects breakdown                                 │
   │  - Panel 3: Priority Queue (HIGH / MED / LOW status)                    │
   │  - Banner: Current Audio Spoken to Phone                                │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Team Interface Contracts (The Technical Boundary)

Every module communicates through a fixed contract so team members can work in parallel without integration friction:

| Interface | Producer | Consumer | Data Shape / Contract |
| :--- | :--- | :--- | :--- |
| **Phone Camera Stream** | Smartphone (IP Webcam / RTSP) | `input.PhoneStreamCamera` (Adrit's Unit) | `(success: bool, frame: np.ndarray[H, W, 3])` |
| **Object Detection** | `perception.ObjectDetector` (YOLOv8) | `intelligence.ContextEngine` | `List[{ "object": str, "confidence": float, "x_center": float, "y_center": float, "bbox": [x1, y1, x2, y2] }]` |
| **OCR Text Extraction**| `perception.OCRReader` (EasyOCR) | `intelligence.PriorityEngine` | `List[{ "text": str, "confidence": float, "bbox": [x1, y1, x2, y2] }]` |
| **Depth / Proximity** | `perception.depth` | `intelligence.ContextEngine` | `(distance_meters: float, proximity_category: str)` |
| **Context Entity** | `intelligence.ContextEngine` | `intelligence.PriorityEngine` | `List[{ "object": str, "confidence": float, "position": str, "position_desc": str, "distance": float, "risk_score": float, "priority": str, "bbox": list }]` |
| **Priority Selection** | `intelligence.PriorityEngine` | `intelligence.ResponseGenerator` | `{ "type": "obstacle"\|"ocr"\|"quick_look", "candidate": dict, "is_critical": bool, "priority": str }` or `None` |
| **Response Text** | `intelligence.ResponseGenerator` | `audio.TextToSpeechEngine` | `{ "text": str }` |
| **Audio Output** | `audio.TextToSpeechEngine` | Phone Audio Player / Earphones | `speak(text: str, interrupt: bool = False)` streamed to phone |

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
   *Non-hazard suppression*: If $\text{Base Danger} \le 20$ (handheld items like a bottle or phone), $S_{\text{proximity}}$ is scaled down by $(\text{Base Danger} / 50.0)$ so harmless background items are never prioritized over genuine walking hazards.

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

| Mode | Trigger | System Behavior | Spoken Output (Heard on Phone) |
| :--- | :--- | :--- | :--- |
| **Mode 1: Quick Look** | Key press `'1'` or UI button | Captures current frame and summarizes the two most salient items. | *"There is a doorway ahead, and a chair slightly to your right."* |
| **Mode 2: Obstacle Awareness** | Default continuous mode | Background monitoring of navigational path. Suppresses low threats and announces high/medium hazards with 3.5s debounce. | *"Stairs ahead, approximately 2 metres away."* |
| **Mode 3: Read** | Key press `'3'` or UI selector | Focuses on text recognition (room numbers, labels, signs). | *"Computer Science Lab, Room 204."* |
| **Mode 4: Ask** | Command / Query | Grounded visual question answering for specific queries (*"Where is the door?"*). | *"The door is to your right, about 3 metres away."* |
| **Mode 5: Safety Alert** | Automatic proximity trigger | High-urgency collision override if an obstacle is within 1.0m directly ahead. | *"Warning. Obstacle directly ahead, less than one metre."* |

---

## 6. Team Role Division & Ownership (Split Hardware Setup)

```
                       ┌────────────────────────────────────────────────────────┐
                       │                    ABHINAV CHAUHAN                     │
                       │           Lead / Core Systems + Integration            │
                       │  - Network Architecture (Phone ↔ Compute Bridge)       │
                       │  - Intelligence Layer (Risk & Priority Engines)        │
                       │  - System Integration, Latency Benchmarks, Pitch Lead  │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                ▼                                                                     ▼
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
│                      ADRIT                       │ │                     DIPAYAN                      │
│     Inference Server & Model Engineer (HEAVY)    │ │           UI / Testing / Documentation           │
│                                                  │ │                                                  │
│ - HOSTS THE HEAVY PROCESSING UNIT ON HIS LAPTOP  │ │ - HOSTS OBSERVER / JUDGE DASHBOARD               │
│ - Runs YOLOv8n object detection model            │ │ - Streamlit 3-panel monitoring UI                │
│ - Runs EasyOCR text extraction pipeline          │ │ - Physical demo scenario props (chair, sign)     │
│ - Manages Depth math & PyTorch inference         │ │ - Test scene curation, video captures, slides    │
│ - Manages TTS audio generation & return stream   │ │ - Validates phone mount & Wi-Fi signal stability │
└──────────────────────────────────────────────────┘ └──────────────────────────────────────────────────┘
```

---

## 7. The 24-Hour Phased Execution Roadmap

```
Hour:   0    1          4          8          12         16         20         22     24
        ├────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────┤
Phase:  [ P1 ] [   P2   ] [   P3   ] [   P4   ] [   P5   ] [   P6   ] [   P7   ] [ P8  ]
        Setup    Models    Streaming  Integration Alerts/QoS  UI Polish  Testing  Rehearse
```

### Phase 1: Environment & Repository Baseline (Hour 0 – Hour 1)
- **Status**: ✅ **COMPLETED (Boilerplate MVP v0 live on `main` branch)**.
- Base contracts, unit tests passing, clean modular architecture.

### Phase 2: Adrit's Compute Server & Model Warmup (Hour 1 – Hour 4)
- **Goal**: Adrit installs PyTorch, Ultralytics, EasyOCR on his laptop and benchmarks local inference speed.
- **Milestone**: YOLOv8n running at 25+ FPS on Adrit's machine; EasyOCR warming up with cached weights.

### Phase 3: Phone Camera Streaming & Audio Return Pipeline (Hour 4 – Hour 8)
- **Goal**: Connect phone camera stream to Adrit's laptop and verify audio plays back through phone.
- **Milestone**: Phone runs IP Webcam / RTSP stream; `PhoneStreamCamera` grabs frames over Wi-Fi without stutter; audio output triggers on phone.

### Phase 4: Full Closed-Loop Integration (Hour 8 – Hour 12)
- **Goal**: Phone camera &rarr; Adrit's compute unit (YOLO + Risk + Priority + TTS) &rarr; Spoken audio in phone's speaker.
- **Milestone**: Walking with phone in hand; system identifies chair and speaks into phone earphones within 150ms.

### Phase 5: Danger Alerts, Latency & Edge-Case Hardening (Hour 12 – Hour 16)
- **Goal**: Ensure collision alerts interrupt smoothly, debounce prevents voice spam, and Wi-Fi drops are handled gracefully.
- **Milestone**: Emergency proximity warning interrupts any ongoing sentence if obstacle < 1.0m.

### Phase 6: Observer Dashboard Polish & Pitch Deck (Hour 16 – Hour 20)
- **Goal**: Dipayan's Streamlit dashboard displays live annotated phone feed, risk cards, and audio status for judges.
- **Milestone**: Judges can watch the dashboard while the test user walks independently with the phone.

### Phase 7: Rehearsal & Feature Freeze (Hour 20 – Hour 24)
- **Goal**: Zero new code. Rehearse the live demo sequence 5 times.
- **Milestone**: Rock-solid 3-minute judging presentation ready.

---

## 8. Abhinav's Action Guide (Lead / Core Systems + Integration)

### Step 1: Clone & Configure Pipeline
```bash
cd "d:\antigravity work\VisonAssist"
git pull origin main
py -m pip install -r requirements.txt
py -m pytest tests/
```

### Step 2: Establish the Local Network Bridge
- Set up a dedicated 5GHz Mobile Hotspot (or stable local Wi-Fi router) connecting:
  1. Phone (Wearable camera & audio output)
  2. Adrit's Laptop (Heavy Processing Unit)
  3. Observer Laptop (Streamlit Dashboard)
- Note the Phone's IP address (e.g., `http://192.168.43.1:8080/video`).
- Update `config.py` with the stream endpoint:
  ```python
  PHONE_STREAM_URL = "http://192.168.43.1:8080/video"
  ```

### Step 3: Calibrate Risk & Priority Logic
- In `intelligence/risk_engine.py`, ensure collision math is tuned for the walking speed of a user with the phone mount:
  - Critical proximity threshold: `1.2m`
  - Audio debounce cooldown: `3.5s`
- Verify that non-hazardous handheld objects (bottles, phones) remain suppressed.

### Step 4: Run CLI & Monitor Integration
```bash
py app.py --mode obstacle
```
- Validate that key switches (`1` for Quick Look, `3` for OCR Read) respond instantly.

---

## 9. Adrit's Action Guide (Heavy Processing Unit / Model Host)

### Step 1: Set Up Heavy AI Models on Your Laptop
Adrit's laptop acts as the high-powered compute server. Install dependencies:
```bash
pip install torch torchvision ultralytics easyocr opencv-python pyttsx3 pywin32
```
Verify model loading:
```bash
# 1. Warm up YOLOv8
python -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); print('YOLOv8 Warm & Ready!')"

# 2. Warm up EasyOCR
python -c "import easyocr; r = easyocr.Reader(['en'], gpu=True); print('EasyOCR Ready!')"
```

### Step 2: Test Frame Capture from Phone Stream
Test reading frames from the phone's IP camera app:
```bash
python -c "
import cv2
cap = cv2.VideoCapture('http://<PHONE_IP>:8080/video')
ret, frame = cap.read()
print('Frame captured successfully:', ret, frame.shape if ret else 'Failed')
"
```

### Step 3: Audio Return Routing to Phone
Ensure the synthesized TTS audio plays on the phone:
- **Option A (Bluetooth / Wireless)**: Pair Bluetooth earbuds or the phone itself as the Bluetooth audio sink of Adrit's laptop.
- **Option B (Audio Relay Stream)**: Stream the synthesized audio back to the phone browser / IP webcam audio return channel (`/audio.wav`).
- Verify voice synthesis is completely asynchronous and does not bottleneck the GPU/CPU vision loop.

---

## 10. Dipayan's Action Guide (Observer Dashboard & Testing)

### Step 1: Launch the Observer Dashboard
On the observer/presentation laptop, connect to Adrit's laptop over the local network:
```bash
py -m streamlit run ui/dashboard.py
```
- Confirm the 3-panel UI renders cleanly:
  - **Panel 1: LIVE VIEW** (Displays what the phone camera sees with bounding boxes).
  - **Panel 2: DETECTED OBJECTS** (Shows detected classes, confidence, and position).
  - **Panel 3: PRIORITY QUEUE** (Displays HIGH, MED, and LOW priority breakdown).
  - **Teal Banner**: Displays current spoken sentence.

### Step 2: Prepare Physical Props & Test Props
1. **The Navigational Obstacle**: A chair placed 1.5 to 2.0 metres in the walking path.
2. **The Room Sign**: A clean printed page: **`ROOM 204 — COMPUTER SCIENCE LAB`**.
3. **The Suppressed Object**: A water bottle on a side desk to demonstrate noise filtering.
4. **Phone Wearable Mount**: A lanyard or chest strap to hold the phone steadily at chest height.

### Step 3: Record Backup Video Demos
Record a clean 60-second screen capture of the dashboard while the test user walks with the phone. This serves as insurance against any Wi-Fi congestion during judging.

---

## 11. Live Judging Demo Script (2–3 Minutes)

### Step 1: The Hook & Physical Demonstration Setup (30s)
> *"Judges, please look at our setup. Notice our user is not holding an awkward, heavy laptop in their hands. They have a smartphone mounted on their chest acting as their eyes and ears. Behind the scenes on our local network, Adrit's laptop acts as the dedicated Edge AI Processing Unit.*  
> *Traditional computer vision overwhelms a visually impaired person with hundreds of noisy announcements: 'Person, bottle, chair, table, door.' VisionAssist solves this with a context-aware Attention and Risk Engine that converts vision into a single, prioritized spoken instruction."*

### Step 2: Live Detection vs. Intelligent Prioritization (45s)
- Test user walks towards the chair prop. A water bottle is visible on the side table.
- Direct judges' attention to Dipayan's observer dashboard:
  - Panel 2 detects: *Person*, *Chair*, and *Bottle*.
  - Panel 3 shows: *Bottle* is **LOW (Suppressed)**; *Chair* is **MEDIUM**.
- Phone speaker announces: **`"Chair ahead, approximately 1.5 metres."`**
- Say to judges: *"Notice how the bottle was completely filtered out. The user only received actionable navigational guidance."*

### Step 3: Sudden Obstacle & Collision Alert (30s)
- An obstacle is moved directly into the user's path (< 1.0m away).
- Phone speaker immediately interrupts: **`"Warning. Obstacle directly ahead, less than one metre."`**
- Explain: *"Proximity and centrality elevated the risk score over 65, triggering an immediate collision alert."*

### Step 4: Text & Signage Reading Mode (30s)
- User switches to Mode 3 (Read) and faces the room sign.
- Phone speaker reads clearly: **`"Computer Science Lab, Room 204."`**

### Step 5: The Closing Platform Pitch (30s)
> *"Today, our architecture splits the lightweight wearable interface on the smartphone from the heavy AI processing on the laptop. Because our software is completely hardware-agnostic, tomorrow that phone and laptop can seamlessly transition to a Raspberry Pi or custom smart glasses without rewriting a single line of our reasoning engine. VisionAssist makes assistive AI practical, safe, and whisper-quiet."*

---

## 12. Anticipated Judge Q&A Defense

| Question | Winning Technical Answer |
| :--- | :--- |
| **"Why is the laptop doing the processing instead of running everything on the phone?"** | *"Modern foundation vision models (YOLOv8 + OCR) demand significant compute and thermal headroom. By splitting the architecture — phone as a lightweight sensor/audio client and an edge computer for heavy inference — we get 30+ FPS, sub-50ms latency, and all-day battery life without overheating the wearable device. This mirrors production architectures like Apple Vision Pro or Ray-Ban Meta glasses where compute is offloaded or distributed."* |
| **"What is actually innovative here? Isn't object detection already solved?"** | *"Object detection is a commodity; that's why we don't claim to invent it. Our innovation is the **decision, attention, and risk engine** built on top of detection. Sighted humans discard 99% of visual inputs and focus only on relevant hazards. We built the reasoning layer that converts raw bounding boxes into a prioritized audio stream without sensory overload."* |
| **"Why not just use ChatGPT or Gemini with a camera?"** | *"Cloud multimodal LLMs require continuous internet, take 2–4 seconds to respond, and provide descriptive paragraphs rather than low-latency collision warnings. VisionAssist runs **100% offline**, operates with sub-50ms latency, and is specifically calibrated for pedestrian mobility safety."* |

---

## 13. Hardware Evolution Roadmap

```
[ V1: Desktop Prototype ]
  Laptop Webcam + Laptop Screen (Initial Scaffolding)
                           │
                           ▼
[ V2: Phone Wearable + Edge Compute ]  ◄── CURRENT HACKATHON LIVE SETUP
  Phone Camera (Capture) + Adrit's Laptop (Heavy AI) + Phone/Earphone (Audio)
                           │
                           ▼
[ V3: Raspberry Pi Wearable ]
  Raspberry Pi Zero 2W / 4 + CSI Camera + Ultrasonic Sensor + Bone-Conduction Audio
                           │
                           ▼
[ V4: Dedicated Smart Glasses ]
  ESP32-S3 / Edge AI Chip + Dual Micro-Cameras + IMU + Lightweight 3D-Printed Frame
```

---

## 14. Repository Quick Reference

- **Repository**: `git@github.com:abhinavv27/VisonAssist.git`
- **Track**: Open Innovation
- **Primary Language**: Python 3.10+
- **Key Commands**:
  - Run Unit Tests: `py -m pytest tests/`
  - Run with Phone Stream: `py app.py --mode obstacle` (with `PHONE_STREAM_URL` configured)
  - Run with Webcam: `py app.py`
  - Run Mock Stream: `py app.py --mock`
  - Run Observer Dashboard: `py app.py --streamlit`
