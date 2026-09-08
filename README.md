# VisionAssist

> **Context-Aware Wearable Vision Assistance for Visually Impaired Users**  
> *"See the world through sound."*  
> **Open Innovation Hackathon · 24-Hour MVP v0**  
> *Manipal University Jaipur · 2026*

---

## 👥 Team & Responsibilities

| Team Member | Role | Key Modules |
| :--- | :--- | :--- |
| **Abhinav Chauhan** | **Lead / Core Systems + Integration** | Architecture, Camera Pipeline, Context Engine, Risk Engine, Priority Engine, System Integration |
| **Adrit** | **Vision / OCR / Audio** | YOLOv8 Detection, EasyOCR Reader, Spatial Math & Depth Estimator, TTS Integration |
| **Dipayan** | **UI / Testing / Documentation** | Streamlit Dashboard UI, Test Scenarios, README & Pitch Materials, Demos |

---

## 💡 The Core Innovation

A camera can detect hundreds of objects in a single frame. Reading that list aloud is not assistance — **it is noise**.

VisionAssist introduces a transparent **Attention & Risk Engine** that answers:
> *"What does the user actually need to know right now?"*

```
Traditional CV:    "Person. Bottle. Chair. Door. Stairs." (Noise overload)
VisionAssist:      "Stairs ahead, approximately 2 metres away." (One clear priority)
```

$$\text{Risk Score} = \text{proximity score} + \text{centrality score} + \text{base danger score} + \text{movement factor}$$

| Object | Base Danger | Distance | Position | Priority Result |
| :--- | :---: | :---: | :---: | :--- |
| **Stairs** | 85 | 2.0 m | Centre | **HIGH** &rarr; Spoken immediately |
| **Car / Vehicle** | 90 | 4.0 m | Centre | **HIGH** &rarr; Spoken immediately |
| **Chair** | 40 | 1.5 m | Left | **MEDIUM** &rarr; Spoken if path is clear |
| **Bottle** | 10 | 1.0 m | Right | **LOW** &rarr; Suppressed to prevent noise |

---

## 🏛️ System Architecture

VisionAssist is built with a **hardware-agnostic adapter architecture**. The intelligence layers never import from specific hardware, allowing the input to transition seamlessly from a laptop webcam to a phone camera, Raspberry Pi, or custom smart glasses without rewriting perception or reasoning code.

```
+-------------------------------------------------------------------------+
|                              INPUT LAYER                                |
|           Laptop Webcam  |  Smartphone Stream  |  Hardware Camera       |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                           PERCEPTION LAYER                              |
|   YOLOv8 Detection  │  EasyOCR Text  │  Depth / Proximity  │  Position   |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                        CONTEXT & RISK ENGINE                            |
|             What is nearby? Where? How urgent? Collision risk?          |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                           PRIORITY ENGINE                               |
|          Selects single top-priority item; filters out audio spam       |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                      NATURAL RESPONSE GENERATOR                         |
|         Concise template speech: "Stairs ahead, approximately 2m"       |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                        OFFLINE TEXT-TO-SPEECH                           |
|        Native Windows SAPI / pyttsx3  ──►  Earphones / Speakers         |
+-------------------------------------------------------------------------+
```

---

## 🔄 Team Interface Contracts

Every module communicates through a fixed contract so team members can work in parallel without integration friction:

| Interface | Data Shape / Contract |
| :--- | :--- |
| **Object Detector** | `{ "object": str, "confidence": float, "x_center": float, "y_center": float, "bbox": [x1, y1, x2, y2] }` |
| **Depth Module** | `(distance_meters: float, proximity_category: str)` |
| **Context Engine** | `{ "object": str, "position": str, "distance": float, "risk_score": float, "priority": str }` |
| **Priority Engine** | Selects top item dict or `None` if debounced / suppressed |
| **Response Generator** | `{ "text": str }` |
| **TTS Engine** | Receives `text: str`, outputs spoken audio via non-blocking background queue |

---

## 🕹️ Product Modes

1. **Mode 1: Quick Look** &mdash; Instant scene overview on user command (*"There is a doorway ahead and a person on your left."*)
2. **Mode 2: Obstacle Awareness** &mdash; Continuous background hazard detection (*"Chair ahead, approximately 1.5 metres."*)
3. **Mode 3: Read** &mdash; Reads text, signage, and room numbers (*"Computer Science Lab, Room 204."*)
4. **Mode 4: Ask** &mdash; Grounded visual question answering (*"What is in front of me?"*)
5. **Mode 5: Safety Alert** &mdash; Critical proximity interruption (*"Warning. Obstacle ahead."*)

---

## 📂 Project Structure

```
VisonAssist/
├── app.py                      # Main application runner (CLI & Streamlit)
├── config.py                   # Central settings, danger weights, thresholds
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── input/                      # Hardware-agnostic capture
│   ├── camera_interface.py     # BaseCamera abstract class
│   ├── webcam.py               # Laptop webcam capture (with mock fallback)
│   └── phone_stream.py         # Smartphone IP stream capture
│
├── perception/                 # Computer vision & OCR
│   ├── object_detection.py     # YOLOv8 detector
│   ├── ocr.py                  # Text reader
│   ├── depth.py                # Monocular distance & proximity estimator
│   └── position.py             # Left / Centre / Right spatial classification
│
├── intelligence/               # Reasoning & decision making
│   ├── context_engine.py       # Assembles contextual entities
│   ├── risk_engine.py          # Rule-based risk scoring formula
│   ├── priority_engine.py      # Noise filter & top-item selector
│   └── response_generator.py   # Natural template sentence generator
│
├── audio/                      # Speech synthesis
│   └── tts.py                  # Thread-safe offline speech engine
│
├── ui/                         # User interface
│   └── dashboard.py            # Streamlit 3-panel accessible dashboard
│
├── hardware/                   # Future wearable adapters
│   ├── camera_adapter.py       # ESP32 / Pi camera adapter
│   ├── ultrasonic_adapter.py   # Ultrasonic distance sensor driver
│   ├── imu_adapter.py          # Inertial measurement unit driver
│   └── button_adapter.py       # Tactile trigger button adapter
│
├── models/                     # Pre-trained model weights
├── utils/                      # Logging & drawing helpers
└── tests/                      # Automated unit test suite
```

---

## 🚀 Quickstart

### 1. Clone and Install
```bash
git clone git@github.com:abhinavv27/VisonAssist.git
cd VisonAssist
pip install -r requirements.txt
```

### 2. Run the Test Suite
```bash
py -m pytest tests/
```

### 3. Run Standalone Desktop Mode (OpenCV + Voice)
```bash
# Run with laptop webcam
py app.py

# Run with simulated camera feed (no webcam needed)
py app.py --mock
```

**Hotkeys in OpenCV Window:**
- `1` : Switch to Quick Look
- `2` : Switch to Obstacle Awareness
- `3` : Switch to Read (OCR)
- `r` : Reset audio cooldown
- `q` : Quit

### 4. Launch the Accessible Streamlit Dashboard
```bash
py app.py --streamlit
# or directly:
py -m streamlit run ui/dashboard.py
```

---

## 🎤 Judging Pitches

### 30-Second Pitch
> *"A camera can see hundreds of things at once — but a visually impaired user doesn't need hundreds of notifications. VisionAssist converts visual information into prioritized, actionable audio guidance. Our MVP uses a camera and laptop to detect objects, read text, estimate position and proximity, assess relevance, and speak only what matters. The architecture is hardware-independent, so today's webcam can become tomorrow's smart glasses."*

### 60-Second Pitch
> *"Visually impaired users face a constant stream of small, unaided decisions — where is the door, is that a step, what does this sign say. Existing assistive computer-vision projects exist, but most either announce everything they detect, overwhelming the user, or are locked to one specific piece of expensive hardware. VisionAssist adds a context and risk engine on top of standard detection, OCR, and depth estimation, so the system decides what the user actually needs to hear right now, and says only that. We built a full closed loop — camera, perception, context, priority, and speech — running locally, with an architecture designed from the start to move onto a phone, a Raspberry Pi, or dedicated smart glasses without rewriting the reasoning layer."*
