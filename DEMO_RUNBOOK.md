# VisionAssist — 3-Minute Live Judging Runbook & Demo Guide

> **Official Step-by-Step Operator Guide for Hackathon Demonstrations**  
> *Track: Open Innovation · Manipal University Jaipur · 2026*

---

## 🎯 Demo Overview & Objective

This runbook gives the exact timing, speech cues, physical actions, and software switches to deliver a flawless, high-impact 3-minute presentation to judges.

### Team Live Stationing
- **Test User (Walking with Smartphone)**: Wears the phone chest-mounted or holds it at chest height facing forward, wearing earphones.
- **Abhinav (Lead Presenter)**: Speaks the pitch, controls narrative, and defends technical architecture during Q&A.
- **Adrit (Edge Compute Operator)**: Monitors model throughput, GPU/CPU latency, and audio return on his laptop.
- **Dipayan (Dashboard Operator & Prop Handler)**: Projects the Streamlit Dashboard on the big monitor and manages the test props.

---

## ⏱️ The 3-Minute Minute-by-Minute Script

```
0:00                0:30                1:15                1:45                2:15                2:45        3:00
  │                   │                   │                   │                   │                   │           │
  ├─ 1. The Hook ─────┼─ 2. Detection vs ─┼─ 3. Collision ────┼─ 4. OCR Reading ──┼─ 5. Ask Mode ─────┼─ 6. Close ┤
  │  (Noise Problem)  │     Prioritization│     Alert (< 1m)  │     ("ROOM 204")  │     (Visual Q&A)  │   Pitch   │
```

---

### Step 1: The Problem Hook (0:00 – 0:30)
- **Presenter (Abhinav)**:
  > *"Judges, a camera can detect hundreds of objects in a single frame. But a visually impaired person cannot listen to a never-ending stream of object names: 'Person. Bottle. Chair. Desk. Door. Stairs.' That is not assistance — it is overwhelming noise.*  
  > *With VisionAssist, we built an Attention and Risk Engine that converts raw detection into a single, prioritized sentence: what is nearby, where it is, and how urgent it is."*
- **Visual Action (Dipayan)**:
  - Point to the live Streamlit dashboard on the projector showing the live camera feed.

---

### Step 2: Detection vs. Prioritization Contrast (0:30 – 1:15)
- **Physical Setup**:
  - A chair is placed 1.5m in front of the test user.
  - A water bottle is placed on a side table to the right.
- **Presenter**:
  > *"Watch what happens right now. The camera sees both the chair and the water bottle. Traditional systems would read both aloud. Watch how VisionAssist prioritizes."*
- **Software Behavior**:
  - Panel 2 detects: `Chair (1.5m)` and `Bottle (1.0m)`.
  - Panel 3 shows: `Bottle` is marked **[LOW - Suppressed]**; `Chair` is marked **[MEDIUM]**.
  - **Phone Audio Speaks**: **`"Chair ahead, approximately 1.5 metres."`**
- **Presenter**:
  > *"Notice how the bottle was completely filtered out. The system decided the user only needed to know about the obstacle in their path."*

---

### Step 3: Sudden Obstacle & Collision Alert (1:15 – 1:45)
- **Physical Action**:
  - Dipayan moves the chair directly into the user's path (< 1.0m away).
- **Software Behavior**:
  - Bounding box flashes **Red** on the dashboard.
  - Temporal tracker detects distance closing rapidly ($\Delta d < -0.2\text{m}$).
  - Risk score spikes past $75.0$ (`HIGH`).
  - **Phone Audio Immediately Interrupts**: **`"Warning. Chair directly ahead, less than one metre."`**
- **Presenter**:
  > *"Because proximity and centrality escalated, the system bypassed the debounce cooldown and issued an immediate collision alert."*

---

### Step 4: Text & Signage OCR Reading (1:45 – 2:15)
- **Physical Action**:
  - User switches to Mode 3 (Read) by pressing `'3'` on the dashboard (or double-tapping).
  - Dipayan holds up the `ROOM 204 — COMPUTER SCIENCE LAB` demo sign.
- **Software Behavior**:
  - EasyOCR extracts text with 95%+ confidence.
  - **Phone Audio Speaks**: **`"Computer Science Lab, Room 204."`**
- **Presenter**:
  > *"In Read Mode, VisionAssist instantly parses signs, room numbers, and notices so the user can navigate unfamiliar indoor campuses independently."*

---

### Step 5: Ask Mode — Grounded Visual Q&A (2:15 – 2:45)
- **Physical Action**:
  - Switch to Mode 4 (Ask).
  - Presenter asks: *"Where is the door?"*
- **Software Behavior**:
  - Context engine searches spatial coordinates of the door.
  - **Phone Audio Speaks**: **`"The door is on your left, approximately 2.8 metres away."`**

---

### Step 6: The Architectural Closing Pitch (2:45 – 3:00)
- **Presenter**:
  > *"Today, our system runs across a lightweight smartphone on the user and an edge processing unit on the network, delivering sub-35ms latency with zero cloud dependency. Our architecture is 100% hardware-agnostic — tomorrow, that phone camera can become Raspberry Pi glasses without changing a single line of our intelligence core. Thank you."*

---

## 🛠️ Pre-Flight Verification Checklist (Dipayan's 5-Minute Pre-Demo Check)

- [ ] **1-Command Automated Pre-Flight**: Run `py app.py --preflight` (checks all 8 systems: Python, AI stack, YOLO, OCR, Camera, Phone Bridge, Audio, and E2E smoke test).
- [ ] **1-Command Latency Profiler**: Run `py app.py --profile` to verify pipeline latency is `< 40ms` and effective FPS is `> 25 FPS`.
- [ ] **1-Command Automated Rehearsal**: Run `py app.py --demo` to verify all 6 judging stages execute with speech under 3 minutes.
- [ ] **Network Check**: Edge laptop and smartphone connected to the same Wi-Fi / Hotspot.
- [ ] **Stream Check**: Open `http://<PHONE_IP>:8080/video` in browser to verify camera frames are streaming.
- [ ] **Audio Relay Check**: Open `http://<LAPTOP_IP>:8088/phone-audio` on phone browser, tap *"Enable Phone Audio"*, and verify audio plays into earphones.
- [ ] **Backup Demo Video**: Verify `demo/backup_demo.mp4` is present (or regenerate via `py scripts/record_backup_demo.py`).
- [ ] **Dashboard Display**: Launch `py app.py --streamlit` on presentation screen.
- [ ] **Physical Props Ready**:
  - Chair obstacle
  - Water bottle distractor
  - Printed `demo_props/room_sign.html`

---

## ⚡ Quick Reference Commands

| Command | Purpose |
| :--- | :--- |
| `py app.py --preflight` | Automated 8-point hardware & software readiness check |
| `py app.py --profile` | Sub-40ms latency and FPS microsecond profiler |
| `py app.py --demo` | 6-stage 3-minute automated judge rehearsal in terminal |
| `py app.py --streamlit` | Dark modern Streamlit presentation dashboard |
| `py app.py --mock --lang hi` | Headless/mock camera loop with natural Hindi voice guidance |
| `py scripts/record_backup_demo.py` | Generates offline backup video (`demo/backup_demo.mp4`) |


