# =============================================================================
# 🚗  AV PERCEPTION SYSTEM — LIVE CAMERA (YOUR LAPTOP)
# =============================================================================
# Run this on your PC/laptop using the model trained on H200
# Requires: av_road_hazard.pt  (copy from H200 to your laptop)
#
# RUN:
#   cd Downloads
#   python av_live_laptop.py
#
# CONTROLS:
#   Q / ESC  → Quit
#   S        → Save screenshot
#   SPACE    → Freeze frame
#   1        → Toggle COCO model (vehicles/people)
#   2        → Toggle Road Hazard model (potholes)
# =============================================================================

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import os

print("=" * 65)
print("  🚗  AV PERCEPTION SYSTEM — LIVE CAMERA")
print("  Dual Model: COCO + Road Hazard Detection")
print("=" * 65)

CAMERA_INDEX = 0
CONF_THRESH  = 0.30

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️   Device: {device}")

# ── Load models ───────────────────────────────────────────────
print("\n📦  Loading models ...")
coco_model = YOLO('yolov8n.pt')
print("  ✅  COCO model (80 classes)")

hazard_path = 'av_road_hazard.pt'
if os.path.exists(hazard_path):
    hazard_model = YOLO(hazard_path)
    USE_HAZARD   = True
    print(f"  ✅  Road hazard model: {hazard_path}")
else:
    USE_HAZARD   = False
    print("  ⚠️  Road hazard model not found — COCO only")
    print("     Copy av_road_hazard.pt from H200 to this folder")

# AV-relevant COCO classes
AV_CLASSES = {
    0:  ('Person',        (50, 50, 255),   'CRITICAL'),
    1:  ('Bicycle',       (0, 165, 255),   'HIGH'),
    2:  ('Car',           (50, 200, 50),   'MEDIUM'),
    3:  ('Motorcycle',    (0, 165, 255),   'HIGH'),
    5:  ('Bus',           (50, 180, 50),   'MEDIUM'),
    7:  ('Truck',         (50, 180, 50),   'MEDIUM'),
    9:  ('Traffic Light', (0, 220, 220),   'CRITICAL'),
    11: ('Stop Sign',     (50, 50, 255),   'CRITICAL'),
    15: ('Cat',           (200,100,200),   'LOW'),
    16: ('Dog',           (200,100,200),   'MEDIUM'),
}

# ── Open camera ───────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌  Camera {CAMERA_INDEX} not found. Try CAMERA_INDEX = 1")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n✅  Camera: {W}x{H}")
print("  Q=Quit  S=Screenshot  SPACE=Freeze  1=Toggle COCO  2=Toggle Hazard\n")

# State
prev_time     = time.time()
fps_d         = 0.0
frozen        = False
frozen_frame  = None
screenshot_n  = 0
frame_count   = 0
PRED_EVERY    = 2
last_dets     = []
show_coco     = True
show_hazard   = True
session_log   = []


def overlay_rect(img, x1, y1, x2, y2, color, alpha=0.5):
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W,x2), min(H,y2)
    if x2<=x1 or y2<=y1: return
    sub  = img[y1:y2,x1:x2]
    fill = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(fill, alpha, sub, 1-alpha, 0, sub)
    img[y1:y2,x1:x2] = sub


while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret: break
    else:
        frame = frozen_frame.copy()

    display     = frame.copy()
    frame_count += 1

    # ── Run detection ─────────────────────────────────────────
    if frame_count % PRED_EVERY == 0 and not frozen:
        last_dets = []

        # COCO model
        if show_coco:
            res = coco_model(frame, verbose=False, conf=CONF_THRESH)[0]
            if res.boxes is not None:
                for box in res.boxes:
                    cls_i = int(box.cls[0].item())
                    if cls_i not in AV_CLASSES: continue
                    conf  = box.conf[0].item()
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    name, color, priority = AV_CLASSES[cls_i]
                    last_dets.append({
                        'box':(x1,y1,x2,y2), 'conf':conf,
                        'name':name, 'color':color,
                        'priority':priority, 'source':'COCO'
                    })

        # Road hazard model
        if show_hazard and USE_HAZARD:
            res = hazard_model(frame, verbose=False, conf=CONF_THRESH)[0]
            if res.boxes is not None:
                hnames = hazard_model.names
                for box in res.boxes:
                    cls_i = int(box.cls[0].item())
                    conf  = box.conf[0].item()
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    name  = hnames.get(cls_i, 'hazard')
                    last_dets.append({
                        'box':(x1,y1,x2,y2), 'conf':conf,
                        'name':name.upper(), 'color':(50,50,200),
                        'priority':'HAZARD', 'source':'ROAD'
                    })

        session_log.append(len(last_dets))

    # ── Draw detections ───────────────────────────────────────
    max_priority = 'CLEAR'
    for d in last_dets:
        x1,y1,x2,y2 = d['box']
        color    = d['color']
        priority = d['priority']

        # Semi-transparent fill
        overlay_rect(display,x1,y1,x2,y2,
                     tuple(c//5 for c in color), alpha=0.2)
        # Box
        cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
        # Corners
        for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),
                               (x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(display,(cx,cy),(cx+dx*18,cy),color,3)
            cv2.line(display,(cx,cy),(cx,cy+dy*18),color,3)
        # Label
        lbl    = f"{d['name']} {d['conf']*100:.0f}%"
        lw,_   = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)[0]
        cv2.rectangle(display,(x1,y1-24),(x1+lw+10,y1),color,-1)
        cv2.putText(display, lbl, (x1+5,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        # Source tag
        src_color = (50,200,50) if d['source']=='COCO' else (50,50,255)
        cv2.putText(display, d['source'], (x1+5,y2+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, src_color, 1)

        if priority == 'CRITICAL':   max_priority = 'CRITICAL'
        elif priority == 'HAZARD' and max_priority != 'CRITICAL':
            max_priority = 'HAZARD'
        elif priority == 'HIGH' and max_priority not in ('CRITICAL','HAZARD'):
            max_priority = 'HIGH'

    # ── CRITICAL flash ────────────────────────────────────────
    if max_priority == 'CRITICAL':
        cv2.rectangle(display,(0,0),(W,H),(0,0,180),18)
        overlay_rect(display,W//4,H//3,3*W//4,2*H//3,(0,0,150),alpha=0.8)
        cv2.putText(display,"⛔  CRITICAL ALERT",
                    (W//4+20,H//2-10),
                    cv2.FONT_HERSHEY_DUPLEX,1.6,(255,255,255),3)
        cv2.putText(display,"STOP / SLOW DOWN",
                    (W//4+40,H//2+48),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,220,220),2)
    elif max_priority == 'HAZARD':
        cv2.rectangle(display,(0,0),(W,H),(0,80,200),12)

    # ── Top bar ───────────────────────────────────────────────
    bar_c = {'CRITICAL':(0,0,100),'HAZARD':(0,60,120),
             'HIGH':(0,100,0),'CLEAR':(30,30,50)}.get(max_priority,(30,30,50))
    overlay_rect(display,0,0,W,58,bar_c,alpha=0.85)

    n = len(last_dets)
    status_txt = (f"  ⛔ CRITICAL — {n} object(s)" if max_priority=='CRITICAL'
                  else f"  ⚠  HAZARD — {n} detected" if max_priority=='HAZARD'
                  else f"  ✅  {n} object(s) detected" if n > 0
                  else "  ✅  CLEAR")
    status_clr = ((50,50,255) if max_priority=='CRITICAL'
                  else (0,200,255) if max_priority=='HAZARD'
                  else (50,220,50))
    cv2.putText(display, status_txt, (10,40),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, status_clr, 2, cv2.LINE_AA)

    # ── Right info panel ──────────────────────────────────────
    px = W-210
    overlay_rect(display,px,65,W,195,(26,26,46),alpha=0.8)
    for i, line in enumerate([
        "MODEL STATUS",
        f"COCO  : {'ON ' if show_coco else 'OFF'}  [1]",
        f"HAZARD: {'ON ' if show_hazard else 'OFF'} [2]",
        f"FPS   : {fps_d:.1f}",
        f"Device: {device.upper()}",
    ]):
        clr = (150,150,220) if i==0 else (255,255,255)
        cv2.putText(display, line, (px+8,90+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1)

    # ── Bottom bar ────────────────────────────────────────────
    overlay_rect(display,0,H-38,W,H,(26,26,46),alpha=0.7)
    cv2.putText(display,
                "  AV Perception  |  COCO+RoadHazard  |  "
                "[Q]Quit [S]Save [SPC]Freeze [1]COCO [2]Hazard",
                (8,H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (160,160,180), 1)

    if frozen:
        cv2.putText(display,"FROZEN — press SPACE",
                    (W//2-150,H//2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,220,220),2)

    # ── FPS ───────────────────────────────────────────────────
    now    = time.time()
    fps_d  = 0.9*fps_d + 0.1*(1.0/max(now-prev_time,1e-6))
    prev_time = now

    cv2.imshow("🚗 AV Perception System", display)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'),27):       break
    elif key == ord('s'):
        screenshot_n += 1
        fname = f"av_capture_{screenshot_n:02d}_{max_priority}.png"
        cv2.imwrite(fname, display)
        print(f"📸  Saved: {fname}")
    elif key == ord(' '):
        frozen = not frozen
        if frozen: frozen_frame = frame.copy()
        print("⏸ Frozen" if frozen else "▶ Resumed")
    elif key == ord('1'):
        show_coco = not show_coco
        print(f"COCO model: {'ON' if show_coco else 'OFF'}")
    elif key == ord('2'):
        show_hazard = not show_hazard
        print(f"Hazard model: {'ON' if show_hazard else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
avg_dets = sum(session_log)/max(len(session_log),1)
print(f"\n✅  Session complete")
print(f"   Avg detections/frame : {avg_dets:.1f}")
print(f"   Screenshots saved    : {screenshot_n}")
