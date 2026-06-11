# =============================================================================
# 🚗  AV PERCEPTION SYSTEM — VIDEO DEMO (H200)
# =============================================================================
# Processes dashcam video OR folder of images on H200
# Runs TWO models simultaneously:
#   Model 1 — COCO YOLOv8   : cars, pedestrians, traffic lights, signs
#   Model 2 — Road Hazard   : potholes, cracks, speed bumps
#
# HOW TO USE ON JUPYTERHUB:
#   1. Upload a dashcam video to JupyterHub (mp4/avi/mov)
#   2. Run: python av_demo_h200.py --video your_video.mp4
#   OR
#   2. Run: python av_demo_h200.py --images folder_of_images/
#   OR
#   2. Run: python av_demo_h200.py  (uses test images from dataset)
#
# OUTPUT:
#   av_output_video.mp4    ← processed video with detections
#   av_demo_frames.png     ← sample frames grid
#   av_detection_stats.png ← detection statistics chart
# =============================================================================

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
from collections import defaultdict

print("=" * 65)
print("  🚗  AV PERCEPTION SYSTEM — DUAL MODEL VIDEO DEMO")
print("  COCO Detection + Road Hazard Detection")
print("=" * 65)

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--video',  type=str, default=None,
                    help='Path to dashcam video file')
parser.add_argument('--images', type=str, default=None,
                    help='Path to folder of images')
parser.add_argument('--conf',   type=float, default=0.30,
                    help='Confidence threshold (default 0.30)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️   Device: {device}")
if device == 'cuda':
    print(f"  GPU : {torch.cuda.get_device_name(0)}")

CONF = args.conf

# ─────────────────────────────────────────────────────────────
# Load Both Models
# ─────────────────────────────────────────────────────────────

print("\n📦  Loading models ...")

# Model 1: COCO — vehicles, pedestrians, traffic lights, signs
coco_model  = YOLO('yolov8n.pt')
print("  ✅  COCO model loaded (80 classes)")

# Model 2: Road hazard — potholes, cracks, speed bumps
hazard_path = 'av_road_hazard.pt'
if os.path.exists(hazard_path):
    hazard_model = YOLO(hazard_path)
    print(f"  ✅  Road hazard model loaded: {hazard_path}")
    USE_HAZARD = True
else:
    print("  ⚠️  av_road_hazard.pt not found — using COCO only")
    print("     Run av_trainer_h200.py first for road hazard detection")
    USE_HAZARD = False

# COCO classes we care about for AV
AV_COCO_CLASSES = {
    0:  ('person',        (50, 50, 255),  'CRITICAL'),
    1:  ('bicycle',       (0, 165, 255),  'HIGH'),
    2:  ('car',           (50, 200, 50),  'MEDIUM'),
    3:  ('motorcycle',    (0, 165, 255),  'HIGH'),
    5:  ('bus',           (50, 200, 50),  'MEDIUM'),
    7:  ('truck',         (50, 200, 50),  'MEDIUM'),
    9:  ('traffic light', (0, 220, 220),  'CRITICAL'),
    11: ('stop sign',     (50, 50, 255),  'CRITICAL'),
    13: ('bench',         (150,150,150),  'LOW'),
    15: ('cat',           (200,100,200),  'MEDIUM'),
    16: ('dog',           (200,100,200),  'MEDIUM'),
}

PRIORITY_COLORS = {
    'CRITICAL': (50,  50, 255),
    'HIGH':     (0,  140, 255),
    'MEDIUM':   (50, 200,  50),
    'LOW':      (150,150, 150),
    'HAZARD':   (0,   80, 200),
}

# ─────────────────────────────────────────────────────────────
# Find input images/video
# ─────────────────────────────────────────────────────────────

def get_images_from_folder(folder, max_imgs=50):
    exts = ('.jpg','.jpeg','.png','.bmp')
    imgs = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(exts)]
    return imgs[:max_imgs]

input_images = []
video_path   = None

if args.video and os.path.exists(args.video):
    video_path = args.video
    print(f"\n📹  Video input: {args.video}")
elif args.images and os.path.exists(args.images):
    input_images = get_images_from_folder(args.images)
    print(f"\n🖼️   Image folder: {args.images} ({len(input_images)} images)")
else:
    # Auto-find test images from dataset
    for candidate in [
        'av_project/dataset/test/images',
        'av_project/dataset/valid/images',
        'pothole_project/dataset/test/images',
        'pothole_project/dataset/valid/images',
    ]:
        if os.path.exists(candidate):
            input_images = get_images_from_folder(candidate, max_imgs=30)
            print(f"\n🖼️   Using dataset images: {candidate}")
            print(f"     ({len(input_images)} images found)")
            break

    if not input_images:
        print("\n⚠️  No input found. Usage:")
        print("     python av_demo_h200.py --video dashcam.mp4")
        print("     python av_demo_h200.py --images road_photos/")
        print("\n   Upload a video or image folder to JupyterHub first.")
        exit(0)

# ─────────────────────────────────────────────────────────────
# Detection function — runs both models on one frame
# ─────────────────────────────────────────────────────────────

detection_log = defaultdict(int)   # class → count
priority_log  = defaultdict(int)   # priority → count

def detect_frame(img_bgr):
    """Run both models on a single BGR frame, return annotated frame + stats"""
    global detection_log, priority_log
    h, w = img_bgr.shape[:2]
    annotated = img_bgr.copy()
    detections = []

    # ── Model 1: COCO ──────────────────────────────────────────
    coco_res = coco_model(img_bgr, verbose=False, conf=CONF)[0]
    if coco_res.boxes is not None:
        for box in coco_res.boxes:
            cls_i = int(box.cls[0].item())
            if cls_i not in AV_COCO_CLASSES:
                continue
            conf  = box.conf[0].item()
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            name, color, priority = AV_COCO_CLASSES[cls_i]
            detections.append({
                'box': (x1,y1,x2,y2), 'conf': conf,
                'name': name, 'color': color,
                'priority': priority, 'source': 'COCO'
            })
            detection_log[name] += 1
            priority_log[priority] += 1

    # ── Model 2: Road Hazard ───────────────────────────────────
    if USE_HAZARD:
        haz_res = hazard_model(img_bgr, verbose=False, conf=CONF)[0]
        if haz_res.boxes is not None:
            haz_classes = hazard_model.names
            for box in haz_res.boxes:
                cls_i = int(box.cls[0].item())
                conf  = box.conf[0].item()
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                name = haz_classes.get(cls_i, f'hazard_{cls_i}')
                detections.append({
                    'box': (x1,y1,x2,y2), 'conf': conf,
                    'name': name, 'color': (50, 50, 200),
                    'priority': 'HAZARD', 'source': 'ROAD'
                })
                detection_log[name] += 1
                priority_log['HAZARD'] += 1

    # ── Draw boxes ────────────────────────────────────────────
    for d in detections:
        x1,y1,x2,y2 = d['box']
        color    = d['color']
        priority = d['priority']

        # Box
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)

        # Corner markers
        clen = 15
        for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),
                               (x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(annotated,(cx,cy),(cx+dx*clen,cy),color,3)
            cv2.line(annotated,(cx,cy),(cx,cy+dy*clen),color,3)

        # Label
        label = f"{d['name']} {d['conf']*100:.0f}%"
        lw,lh = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)[0]
        cv2.rectangle(annotated,(x1,y1-24),(x1+lw+10,y1),color,-1)
        cv2.putText(annotated, label, (x1+5,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255,255,255), 1, cv2.LINE_AA)

        # Priority badge
        badge = PRIORITY_COLORS.get(priority, (150,150,150))
        cv2.putText(annotated, priority, (x1+5, y2+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    badge, 2, cv2.LINE_AA)

    # ── Status bar ────────────────────────────────────────────
    n_crit = sum(1 for d in detections if d['priority'] == 'CRITICAL')
    n_haz  = sum(1 for d in detections if d['priority'] == 'HAZARD')
    n_tot  = len(detections)

    bar_color = (0,0,120) if n_crit > 0 else (0,80,0)
    overlay   = annotated.copy()
    cv2.rectangle(overlay, (0,0), (w,50), bar_color, -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    status = (f"  ⛔ CRITICAL: {n_crit} object(s)"
              if n_crit > 0 else
              f"  ⚠ HAZARD: {n_haz}" if n_haz > 0
              else f"  ✅ {n_tot} object(s) detected")
    status_clr = (50,50,255) if n_crit > 0 else (50,220,50)
    cv2.putText(annotated, status, (10,34),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, status_clr, 2, cv2.LINE_AA)

    # ── Model labels ──────────────────────────────────────────
    cv2.putText(annotated, "COCO: vehicles/people/signs",
                (w-300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (50,200,50), 1, cv2.LINE_AA)
    if USE_HAZARD:
        cv2.putText(annotated, "Road: potholes/hazards",
                    (w-300, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (50,50,255), 1, cv2.LINE_AA)

    return annotated, detections


# ─────────────────────────────────────────────────────────────
# Process video
# ─────────────────────────────────────────────────────────────

if video_path:
    print(f"\n🎬  Processing video: {video_path}")
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    w_out  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_out  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = 'av_output_video.mp4'
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w_out, h_out)
    )

    sample_frames = []
    frame_n = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        annotated, dets = detect_frame(frame)
        writer.write(annotated)
        frame_n += 1

        # Save sample frames
        if frame_n % max(1, total//6) == 0:
            sample_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        if frame_n % 50 == 0:
            elapsed = time.time() - t0
            fps_proc = frame_n / elapsed
            print(f"  Frame {frame_n}/{total} | "
                  f"{fps_proc:.1f} FPS | "
                  f"{frame_n/total*100:.0f}% done")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"\n✅  Video processed: {frame_n} frames in {elapsed:.1f}s "
          f"({frame_n/elapsed:.1f} FPS)")
    print(f"💾  Saved: {out_path}")

    # Sample frames grid
    if sample_frames:
        n  = min(len(sample_frames), 6)
        r  = (n+2)//3
        fig, axes = plt.subplots(r, 3, figsize=(18, r*4),
                                 facecolor='#1a1a2e')
        fig.suptitle('🚗  AV Perception — Sample Frames',
                     fontsize=16, fontweight='bold', color='white')
        for i, ax in enumerate(axes.flat):
            if i < n:
                ax.imshow(sample_frames[i])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('av_demo_frames.png', dpi=150,
                    bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print("💾  Saved: av_demo_frames.png")

# ─────────────────────────────────────────────────────────────
# Process images
# ─────────────────────────────────────────────────────────────

elif input_images:
    print(f"\n🖼️   Processing {len(input_images)} images ...")
    t0     = time.time()
    rows   = (min(len(input_images),6)+2)//3
    fig, axes = plt.subplots(rows, 3, figsize=(18, rows*4),
                              facecolor='#1a1a2e')
    fig.suptitle('🚗  AV Perception System — Dual Model Detection',
                 fontsize=16, fontweight='bold', color='white', y=1.01)

    for i, (ax, img_path) in enumerate(
            zip(axes.flat, input_images[:rows*3])):
        frame    = cv2.imread(img_path)
        if frame is None:
            ax.axis('off'); continue
        annotated, dets = detect_frame(frame)
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        n_crit = sum(1 for d in dets if d['priority']=='CRITICAL')
        n_haz  = sum(1 for d in dets if d['priority']=='HAZARD')
        title  = (f"⛔ CRITICAL ({n_crit})" if n_crit else
                  f"⚠ Hazard ({n_haz})" if n_haz else
                  f"✅ {len(dets)} detected")
        color  = 'red' if n_crit else 'orange' if n_haz else 'lime'
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')

    for ax in axes.flat[min(len(input_images),rows*3):]:
        ax.axis('off')

    elapsed = time.time() - t0
    print(f"✅  Processed {len(input_images)} images in {elapsed:.1f}s "
          f"({len(input_images)/elapsed:.1f} FPS on {device})")

    plt.tight_layout()
    plt.savefig('av_demo_frames.png', dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("💾  Saved: av_demo_frames.png")

# ─────────────────────────────────────────────────────────────
# Detection Statistics Chart
# ─────────────────────────────────────────────────────────────

if detection_log:
    BG = '#1a1a2e'
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    # Objects detected
    ax1.set_facecolor(BG)
    items  = sorted(detection_log.items(), key=lambda x: x[1], reverse=True)[:10]
    names  = [x[0] for x in items]
    counts = [x[1] for x in items]
    bar_c  = ['#DD4444' if n in ['person','stop sign','traffic light']
               else '#4C72B0' for n in names]
    ax1.barh(names, counts, color=bar_c, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Detection Count', color='white')
    ax1.set_title('Objects Detected', color='white',
                  fontweight='bold', fontsize=12)
    ax1.tick_params(colors='white')
    ax1.spines[['top','right']].set_visible(False)
    ax1.spines[['left','bottom']].set_color('#444466')
    ax1.grid(axis='x', alpha=0.2, color='white')

    # Priority distribution
    ax2.set_facecolor(BG)
    pri_items = [(k,v) for k,v in priority_log.items() if v > 0]
    pri_names = [x[0] for x in pri_items]
    pri_vals  = [x[1] for x in pri_items]
    pri_cols  = [{'CRITICAL':(255,50,50),'HIGH':(255,140,0),
                  'MEDIUM':(50,200,50),'LOW':(150,150,150),
                  'HAZARD':(200,50,50)}.get(p,(100,100,100))
                 for p in pri_names]
    pri_cols  = [tuple(c/255 for c in col) for col in pri_cols]
    ax2.pie(pri_vals, labels=pri_names, colors=pri_cols,
            autopct='%1.0f%%', textprops={'color':'white'},
            wedgeprops={'edgecolor':'#1a1a2e','linewidth':2})
    ax2.set_title('Detection Priority Distribution',
                  color='white', fontweight='bold', fontsize=12)

    plt.suptitle('🚗  AV Perception System — Detection Statistics',
                 fontsize=14, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    plt.savefig('av_detection_stats.png', dpi=150,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    print("💾  Saved: av_detection_stats.png")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
total_dets = sum(detection_log.values())
print(f"""
{'='*65}
  🚗  DEMO COMPLETE
{'='*65}
  Total detections   : {total_dets}
  Critical events    : {priority_log.get('CRITICAL',0)}
  Road hazards       : {priority_log.get('HAZARD',0)}
  Device used        : {device}

  Most detected:
""")
for name, cnt in sorted(detection_log.items(),
                         key=lambda x: x[1], reverse=True)[:5]:
    print(f"    {name:<20} : {cnt}")

print(f"""
  📁  Output files:
  ✅  av_demo_frames.png       ← detection grid
  ✅  av_detection_stats.png   ← statistics charts
""" + ("  ✅  av_output_video.mp4     ← processed video\n"
       if video_path else "") + f"{'='*65}")
