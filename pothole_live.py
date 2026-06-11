# =============================================================================
# 🚗  LIVE POTHOLE & OBSTACLE DETECTOR — Real-Time Camera
# =============================================================================
# Run AFTER pothole_detector.py has finished training.
#
# CONTROLS:
#   Q / ESC   → Quit
#   S         → Save screenshot
#   SPACE     → Freeze / unfreeze frame
#   W         → Toggle warning sounds (visual only — no audio needed)
#
# RUN:
#   cd Downloads
#   python pothole_live.py
# =============================================================================

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import os

print("=" * 60)
print("  🚗  LIVE POTHOLE & OBSTACLE DETECTOR")
print("=" * 60)

# ── Config ────────────────────────────────────────────────────
MODEL_PATH   = "pothole_best.pt"
CAMERA_INDEX = 0
CONF_THRESH  = 0.30     # Show detections above this confidence
DANGER_THRESH = 0.15    # Box covers >15% of screen → DANGER warning

# BGR colours
CLR_RED    = (50,  50,  255)
CLR_ORANGE = (0,  140,  255)
CLR_GREEN  = (50, 220,  50)
CLR_WHITE  = (255,255,  255)
CLR_BLACK  = (0,   0,   0)
CLR_DARK   = (26,  26,  46)
CLR_YELLOW = (0,  220,  220)

# Hazard colours by class name
HAZARD_COLORS = {
    'pothole':    (50,  50, 255),   # red
    'speedbump':  (0,  165, 255),   # orange
    'speed bump': (0,  165, 255),
    'debris':     (0,  215, 255),   # yellow
    'crack':      (128, 0,  255),   # purple
    'default':    (50, 200, 50),    # green for unknown
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️   Device: {device}")

# ── Load model ────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"❌  Model '{MODEL_PATH}' not found!")
    print("    Run pothole_detector.py first to train the model.")
    print("\n    Testing with COCO pretrained model instead ...")
    MODEL_PATH = 'yolov8n.pt'

print(f"📦  Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names
print(f"✅  Model loaded — {len(CLASS_NAMES)} classes: {list(CLASS_NAMES.values())}")

# ── Open camera ───────────────────────────────────────────────
print(f"\n📷  Opening camera {CAMERA_INDEX} ...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"❌  Could not open camera {CAMERA_INDEX}")
    print("    Try: CAMERA_INDEX = 1")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SCREEN_AREA = W * H

print(f"✅  Camera opened: {W}x{H}")
print("\n🚗  Live hazard detection running!")
print("    Q = Quit | S = Screenshot | SPACE = Freeze\n")


# ── Helpers ───────────────────────────────────────────────────
def overlay_rect(img, x1, y1, x2, y2, color, alpha=0.5):
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W,x2), min(H,y2)
    if x2<=x1 or y2<=y1: return
    sub  = img[y1:y2, x1:x2]
    rect = np.full(sub.shape, color[::-1] if len(color)==3 else color,
                   dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1-alpha, 0, sub)
    img[y1:y2, x1:x2] = sub

def get_hazard_color(class_name, conf):
    name  = class_name.lower()
    color = HAZARD_COLORS.get(name, HAZARD_COLORS['default'])
    if conf < 0.5:
        color = tuple(int(c * 0.7) for c in color)
    return color

def danger_level(box_area_ratio, conf):
    if box_area_ratio > 0.20 and conf > 0.6:
        return "DANGER"
    elif box_area_ratio > 0.10 and conf > 0.4:
        return "WARNING"
    else:
        return "DETECTED"

def draw_warning_overlay(frame, level, n_hazards):
    """Full screen flash for DANGER level"""
    if level == "DANGER":
        # Red border flash
        cv2.rectangle(frame, (0,0), (W,H), (0,0,200), 20)
        # Big centered warning
        overlay_rect(frame, W//4, H//3, 3*W//4, 2*H//3,
                     (0,0,180), alpha=0.8)
        cv2.putText(frame, "⚠ DANGER",
                    (W//4 + 30, H//2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, CLR_WHITE, 3, cv2.LINE_AA)
        cv2.putText(frame, f"HAZARD DETECTED",
                    (W//4 + 30, H//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, CLR_YELLOW, 2, cv2.LINE_AA)

    elif level == "WARNING":
        cv2.rectangle(frame, (0,0), (W,H), (0,140,255), 12)


# ── Stats tracking ────────────────────────────────────────────
detection_log = []    # stores (timestamp, class, conf)
fps_display   = 0.0
prev_time     = time.time()
frozen        = False
frozen_frame  = None
screenshot_n  = 0
frame_count   = 0
PRED_EVERY    = 2     # Run YOLO every 2 frames for speed

# Last detection state
last_detections = []
max_level       = "CLEAR"


# ── Main loop ─────────────────────────────────────────────────
while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        frame = frozen_frame.copy()

    display     = frame.copy()
    frame_count += 1

    # ── Run detection ─────────────────────────────────────────
    if frame_count % PRED_EVERY == 0 and not frozen:
        results         = model(frame, verbose=False, conf=CONF_THRESH)[0]
        last_detections = []
        max_level       = "CLEAR"

        if results.boxes is not None:
            for box in results.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf  = box.conf[0].item()
                cls_i = int(box.cls[0].item())
                name  = CLASS_NAMES.get(cls_i, f"class_{cls_i}")

                box_area = (x2-x1) * (y2-y1)
                ratio    = box_area / SCREEN_AREA
                level    = danger_level(ratio, conf)

                last_detections.append({
                    'box': (x1,y1,x2,y2),
                    'conf': conf,
                    'name': name,
                    'level': level,
                    'color': get_hazard_color(name, conf)
                })

                # Log detection
                detection_log.append({
                    'time': time.time(),
                    'class': name,
                    'conf': conf,
                    'level': level
                })

                if level == "DANGER":
                    max_level = "DANGER"
                elif level == "WARNING" and max_level != "DANGER":
                    max_level = "WARNING"
                elif max_level == "CLEAR":
                    max_level = "DETECTED"

    # ── Draw detections ───────────────────────────────────────
    for d in last_detections:
        x1,y1,x2,y2 = d['box']
        color = d['color']
        conf  = d['conf']
        name  = d['name']
        level = d['level']

        # Box fill (semi-transparent)
        overlay_rect(display, x1,y1,x2,y2,
                     tuple(c//4 for c in color), alpha=0.25)

        # Box border — thicker for danger
        thickness = 4 if level == "DANGER" else 2
        cv2.rectangle(display, (x1,y1), (x2,y2), color, thickness)

        # Corner markers
        clen = 20
        cv2.line(display, (x1,y1), (x1+clen,y1), color, 3)
        cv2.line(display, (x1,y1), (x1,y1+clen), color, 3)
        cv2.line(display, (x2,y1), (x2-clen,y1), color, 3)
        cv2.line(display, (x2,y1), (x2,y1+clen), color, 3)
        cv2.line(display, (x1,y2), (x1+clen,y2), color, 3)
        cv2.line(display, (x1,y2), (x1,y2-clen), color, 3)
        cv2.line(display, (x2,y2), (x2-clen,y2), color, 3)
        cv2.line(display, (x2,y2), (x2,y2-clen), color, 3)

        # Label
        label_txt = f"{name.upper()} {conf*100:.0f}%"
        lw, lh    = cv2.getTextSize(label_txt,
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.rectangle(display, (x1, y1-30), (x1+lw+12, y1), color, -1)
        cv2.putText(display, label_txt,
                    (x1+6, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_WHITE, 2, cv2.LINE_AA)

        # Level badge
        badge_color = {
            "DANGER": (0,0,200),
            "WARNING": (0,140,255),
            "DETECTED": (0,180,0)
        }.get(level, (100,100,100))
        cv2.putText(display, level,
                    (x1+6, y2+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2, cv2.LINE_AA)

    # ── Full screen warning overlay ───────────────────────────
    if max_level in ("DANGER", "WARNING"):
        draw_warning_overlay(display, max_level, len(last_detections))

    # ── Top status bar ────────────────────────────────────────
    bar_color = {
        "DANGER":   (0, 0, 120),
        "WARNING":  (0, 80, 160),
        "DETECTED": (0, 80, 0),
        "CLEAR":    (30, 30, 50),
    }.get(max_level, (30,30,50))
    overlay_rect(display, 0, 0, W, 65, bar_color, alpha=0.85)

    status_icon = {
        "DANGER":   "⛔",
        "WARNING":  "⚠️",
        "DETECTED": "🔍",
        "CLEAR":    "✅",
    }.get(max_level, "🔍")

    n = len(last_detections)
    status_txt = (
        f"  {n} HAZARD{'S' if n!=1 else ''} DETECTED"
        if n > 0 else "  ROAD CLEAR"
    )
    status_color = {
        "DANGER":   CLR_RED,
        "WARNING":  CLR_YELLOW,
        "DETECTED": CLR_GREEN,
        "CLEAR":    CLR_GREEN,
    }.get(max_level, CLR_WHITE)

    cv2.putText(display, status_txt,
                (10, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, status_color, 2, cv2.LINE_AA)

    # ── Right side stats panel ────────────────────────────────
    panel_x = W - 200
    overlay_rect(display, panel_x, 70, W, 200, CLR_DARK, alpha=0.75)
    cv2.putText(display, "DETECTION STATS",
                (panel_x+8, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,200), 1)

    total_detections = len(detection_log)
    danger_count     = sum(1 for d in detection_log if d['level']=="DANGER")

    stats_lines = [
        f"Total hazards : {total_detections}",
        f"Danger events : {danger_count}",
        f"Current       : {n} objects",
        f"FPS           : {fps_display:.1f}",
    ]
    for i, line in enumerate(stats_lines):
        cv2.putText(display, line,
                    (panel_x+8, 115 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)

    # ── Bottom controls bar ───────────────────────────────────
    overlay_rect(display, 0, H-38, W, H, CLR_DARK, alpha=0.7)
    cv2.putText(display,
                "  YOLOv8 Pothole & Obstacle Detector  |  "
                "[Q] Quit   [S] Save   [SPACE] Freeze",
                (8, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,180), 1)

    # ── FROZEN label ─────────────────────────────────────────
    if frozen:
        cv2.putText(display, "FROZEN — press SPACE to resume",
                    (W//2 - 200, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_YELLOW, 2, cv2.LINE_AA)

    # ── FPS ───────────────────────────────────────────────────
    now        = time.time()
    fps_display = 0.9*fps_display + 0.1*(1.0/max(now-prev_time,1e-6))
    prev_time  = now

    # ── Show ─────────────────────────────────────────────────
    cv2.imshow("🚗 Live Pothole & Obstacle Detector", display)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('s'):
        screenshot_n += 1
        fname = f"road_capture_{screenshot_n:02d}_{max_level}.png"
        cv2.imwrite(fname, display)
        print(f"📸  Screenshot saved: {fname}")
    elif key == ord(' '):
        frozen = not frozen
        if frozen:
            frozen_frame = frame.copy()
            print(f"⏸   Frozen — {n} hazard(s) visible")
        else:
            print("▶️   Resumed")

cap.release()
cv2.destroyAllWindows()

# ── Session summary ───────────────────────────────────────────
print(f"""
{'='*50}
  🚗  SESSION SUMMARY
{'='*50}
  Total hazards detected : {len(detection_log)}
  Danger events          : {sum(1 for d in detection_log if d['level']=='DANGER')}
  Warning events         : {sum(1 for d in detection_log if d['level']=='WARNING')}
  Screenshots saved      : {screenshot_n}
{'='*50}
""")
