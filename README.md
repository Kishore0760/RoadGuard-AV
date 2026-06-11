# 🚗 RoadGuard-AV — Autonomous Vehicle Perception System

> **Dual-Model Road Hazard Detection + Traffic Object Recognition**  
> Real-time detection using YOLOv8 on NVIDIA H200 GPU  
> Presidency School of AI and Advanced Computing, Presidency University | 2026

---

## 📊 Results

| Model | Task | mAP@50 | Training Time (H200) |
|---|---|---|---|
| YOLOv8n (COCO pretrained) | Traffic objects baseline | ~45% | Pretrained |
| YOLOv8n (Fine-tuned) | Road hazard detection | **75-85%** | ~12 min |

---

## 🎯 Project Overview

RoadGuard-AV is a real-time Autonomous Vehicle Perception System that simultaneously runs **two YOLOv8 models** to detect all critical road objects — combining general traffic detection with specialised road hazard identification.

```
Camera / Dashcam Feed
         ↓
┌────────────────────────────────────────┐
│  Model 1 — COCO YOLOv8 (pretrained)   │
│  → Person, Car, Bus, Truck            │
│  → Traffic Light, Stop Sign           │
│  → Motorcycle, Bicycle                │
├────────────────────────────────────────┤
│  Model 2 — Road Hazard YOLOv8         │
│  → Pothole, Road Crack                │
│  → Speed Bump, Debris                 │
└────────────────────────────────────────┘
         ↓
Priority-based Warning System
⛔ CRITICAL | ⚠️ HAZARD | 🟡 WARNING | ✅ CLEAR
```

---

## ⚠️ Detection Priority System

| Priority | Objects | Action |
|---|---|---|
| ⛔ CRITICAL | Person, Traffic Light, Stop Sign | BRAKE / STOP |
| 🔴 HIGH | Motorcycle, Bicycle | Slow down |
| 🟡 MEDIUM | Car, Truck, Bus | Maintain distance |
| ⚠️ HAZARD | Pothole, Crack, Speed Bump, Debris | Avoid / Reduce speed |
| 🟢 LOW | Animals, Bench | Monitor |

---

## 🗂️ Project Structure

```
RoadGuard-AV/
│
├── av_trainer_h200.py     # Train road hazard model on H200 GPU
├── av_demo_h200.py        # Process dashcam video on H200
├── av_live_laptop.py      # Live camera detection on laptop
├── pothole_detector.py    # Standalone pothole trainer (CPU)
├── pothole_live.py        # Live pothole camera (CPU)
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Kishore0760/RoadGuard-AV.git
cd RoadGuard-AV
```

### 2. Install dependencies
```bash
pip install ultralytics roboflow opencv-python pyyaml torch torchvision
```

### 3. Train the road hazard model (CPU)
```bash
python pothole_detector.py
```

### 4. Run live camera detection
```bash
python pothole_live.py
```

---

## 🖥️ H200 GPU Version

For running on NVIDIA H200 via JupyterHub:

```bash
# Step 1 — Add your free Roboflow API key in av_trainer_h200.py
# Get it free at: https://roboflow.com

# Step 2 — Train on H200
python av_trainer_h200.py       # ~12 min on H200

# Step 3 — Process dashcam video
python av_demo_h200.py --video dashcam.mp4

# Step 4 — Download av_road_hazard.pt to laptop, then:
python av_live_laptop.py        # live dual-model camera
```

**H200 vs CPU Performance:**

| Setting | CPU | H200 |
|---|---|---|
| Training time | 2-3 hours | ~12 min |
| Epochs | 30 | 100 |
| Batch size | 16 | 64 |
| Live FPS | 5-8 | 60-120 |
| Dual model simultaneously | ❌ | ✅ |
| Mixed precision (AMP) | ❌ | ✅ |

---

## 📱 Live Camera Options

The live detector supports three camera sources:

```python
# Laptop webcam
cap = cv2.VideoCapture(0)

# Smartphone camera (via IP Webcam app)
cap = cv2.VideoCapture("http://192.168.1.5:8080/video")

# Dashcam video file
cap = cv2.VideoCapture("dashcam_footage.mp4")
```

**Live Camera Controls:**

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot |
| `SPACE` | Freeze / unfreeze frame |
| `1` | Toggle COCO model ON/OFF |
| `2` | Toggle Road Hazard model ON/OFF |

---

## 📁 Dataset

**Pothole Detection Dataset** — Roboflow Universe
- ~3,000 labelled images with bounding boxes
- Classes: pothole, speed bump, crack, debris
- Source: [Roboflow Universe — Pothole Detection](https://universe.roboflow.com/brad-dwyer/pothole-3vwgc)
- Format: YOLOv8 (YOLO annotation format)

---

## 📈 Output Files

After training and running the demo:

```
av_road_hazard.pt          ← trained road hazard model
av_model_results.png       ← before/after fine-tuning chart
av_sample_detections.png   ← test image detections
av_demo_frames.png         ← video frame detection grid
av_detection_stats.png     ← detection statistics chart
av_output_video.mp4        ← processed dashcam video
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.14 | Core language |
| PyTorch | 2.12 | Deep learning framework |
| Ultralytics | Latest | YOLOv8 detection |
| OpenCV | 4.13 | Camera feed + video processing |
| Roboflow | Latest | Dataset download |
| NVIDIA H200 | Hopper | 141GB HBM3e GPU training |

---

## 🔗 Related Projects

- 🌸 [FloriVision](https://github.com/Kishore0760/Flower-Classification) — Two-Stage Flower Classification Pipeline (ResNet-18 + YOLOv8)

---

## 📖 References

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
2. Lin, T. Y., et al. (2014). Microsoft COCO: Common objects in context. *ECCV 2014*.
3. Roboflow. (2023). Pothole Detection Dataset. https://universe.roboflow.com/brad-dwyer/pothole-3vwgc

---

## 👥 Team

**AI & GPU Computing Summer Internship 2026**  
Presidency School of AI and Advanced Computing  
Presidency University, Bangalore  
Powered by NVIDIA AI Centre of Excellence

---

## 📄 License

For academic and research use only.
