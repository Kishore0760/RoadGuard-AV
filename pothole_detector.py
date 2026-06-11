# =============================================================================
# 🚗  POTHOLE & OBSTACLE DETECTOR — YOLOv8
# Automobile Safety Computer Vision Project
# =============================================================================
# Detects road hazards in real-time:
#   🕳️  Pothole    🚧  Speed Bump    🪨  Debris
#
# SETUP (run once):
#   python -m pip install ultralytics roboflow
#
# DATASET:
#   Free from Roboflow Universe — script downloads it automatically
#   Requires a free Roboflow account → https://roboflow.com (30 seconds)
#
# RUN:
#   python pothole_detector.py
# =============================================================================

import os
import time
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

print("=" * 60)
print("  🚗  POTHOLE & OBSTACLE DETECTOR")
print("  Automobile Safety — YOLOv8 Object Detection")
print("=" * 60)

# ── Config ────────────────────────────────────────────────────
PROJECT_DIR  = "pothole_project"
DATA_DIR     = os.path.join(PROJECT_DIR, "dataset")
EPOCHS       = 30       # H200 UPGRADE → 100
IMG_SIZE     = 640      # YOLOv8 standard detection size
BATCH_SIZE   = 16       # H200 UPGRADE → 64
CONF_THRESH  = 0.35     # Minimum confidence to show detection

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 │ Download Dataset from Roboflow
# ─────────────────────────────────────────────────────────────
# Using the free public "Pothole Detection" dataset from Roboflow Universe
# ~3,000 labelled images with bounding boxes
#
# TO GET YOUR FREE API KEY:
#   1. Go to https://roboflow.com
#   2. Sign up (free, 30 seconds)
#   3. Go to Settings → Roboflow API → copy your API key
#   4. Paste it below replacing YOUR_API_KEY_HERE
# ─────────────────────────────────────────────────────────────

API_KEY = "YOUR_API_KEY_HERE"   # ← Paste your free Roboflow API key here

data_yaml = os.path.join(DATA_DIR, "data.yaml")

if not os.path.exists(data_yaml):
    if API_KEY == "YOUR_API_KEY_HERE":
        print("""
╔══════════════════════════════════════════════════════════╗
║  DATASET SETUP — ONE TIME ONLY                           ║
║                                                          ║
║  1. Go to: https://roboflow.com                          ║
║  2. Sign up FREE (30 seconds)                            ║
║  3. Settings → API → copy your API key                   ║
║  4. Open this file and replace:                          ║
║     API_KEY = "YOUR_API_KEY_HERE"                        ║
║     with your actual key                                 ║
║  5. Run again                                            ║
║                                                          ║
║  OR use the manual download option below                 ║
╚══════════════════════════════════════════════════════════╝

MANUAL DOWNLOAD OPTION (no account needed):
  1. Go to: https://universe.roboflow.com/brad-dwyer/pothole-3vwgc
  2. Click Download → YOLOv8 format → Show download code
  3. Copy the code and run it in terminal
  4. Move the downloaded folder to: pothole_project/dataset/
        """)
        exit(0)

    try:
        from roboflow import Roboflow
        print("📥  Downloading pothole dataset from Roboflow ...")
        rf      = Roboflow(api_key=API_KEY)
        project = rf.workspace("brad-dwyer").project("pothole-3vwgc")
        version = project.version(1)
        dataset = version.download("yolov8", location=DATA_DIR)
        print("✅  Dataset downloaded!")
    except Exception as e:
        print(f"❌  Download failed: {e}")
        print("    Try the manual download option above")
        exit(1)
else:
    print(f"✅  Dataset already exists in '{DATA_DIR}/'")

# Read dataset info
with open(data_yaml, 'r') as f:
    data_info = yaml.safe_load(f)

CLASS_NAMES = data_info.get('names', ['pothole'])
NUM_CLASSES = len(CLASS_NAMES)
print(f"\n📂  Classes ({NUM_CLASSES}): {CLASS_NAMES}")

# ─────────────────────────────────────────────────────────────
# STEP 2 │ Train YOLOv8 for Pothole Detection
# ─────────────────────────────────────────────────────────────
# Note: This uses yolov8n.pt (detection) NOT yolov8n-cls.pt (classification)
# Detection gives bounding boxes + confidence scores
# ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  🚀 Training YOLOv8 for {EPOCHS} epochs")
print(f"{'='*60}\n")

model = YOLO('yolov8n.pt')   # Pretrained on COCO — knows cars, people, etc.

t0 = time.time()
results = model.train(
    data=data_yaml,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    workers=0,           # H200 UPGRADE → 4
    project=PROJECT_DIR,
    name='pothole_model',
    exist_ok=True,
    verbose=True,
    # Augmentation — helps a lot for road detection
    flipud=0.1,          # Vertical flip (unusual angles)
    fliplr=0.5,          # Horizontal flip
    mosaic=1.0,          # Mix 4 images — great for detecting small potholes
    mixup=0.1,
    degrees=5.0,         # Slight rotation (camera angle variation)
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,         # Colour variation (day/night/weather)
    hsv_s=0.7,
    hsv_v=0.4,
)

elapsed = (time.time() - t0) / 60
print(f"\n✅  Training complete in {elapsed:.1f} min")

# ─────────────────────────────────────────────────────────────
# STEP 3 │ Evaluate Model
# ─────────────────────────────────────────────────────────────

print("\n📊  Evaluating model on validation set ...")
best_model_path = os.path.join(PROJECT_DIR, 'pothole_model', 'weights', 'best.pt')
trained_model   = YOLO(best_model_path)
metrics         = trained_model.val(data=data_yaml, verbose=False)

map50   = metrics.box.map50   * 100
map5095 = metrics.box.map     * 100
precision = metrics.box.mp    * 100
recall    = metrics.box.mr    * 100

print(f"\n{'='*50}")
print(f"  📊  DETECTION RESULTS")
print(f"{'='*50}")
print(f"  mAP@50       : {map50:.1f}%")
print(f"  mAP@50-95    : {map5095:.1f}%")
print(f"  Precision    : {precision:.1f}%")
print(f"  Recall       : {recall:.1f}%")
print(f"{'='*50}")
print("""
  mAP@50 explained:
  → Measures detection accuracy at 50% overlap threshold
  → 70%+ is good for road hazard detection
  → 80%+ is excellent
""")

# ─────────────────────────────────────────────────────────────
# STEP 4 │ Test on Sample Images
# ─────────────────────────────────────────────────────────────

print("🔍  Running detections on sample images ...")

val_img_dir = os.path.join(DATA_DIR, 'valid', 'images')
if os.path.exists(val_img_dir):
    val_imgs = [
        os.path.join(val_img_dir, f)
        for f in os.listdir(val_img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ][:9]

    if val_imgs:
        rows = 3
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.suptitle('🚗  Pothole Detection — Sample Results',
                     fontsize=16, fontweight='bold')

        for ax, img_path in zip(axes.flat, val_imgs):
            det     = trained_model(img_path, verbose=False, conf=CONF_THRESH)[0]
            orig    = Image.open(img_path).convert('RGB')
            img_arr = np.array(orig)

            ax.imshow(img_arr)
            ax.axis('off')

            n_det = 0
            if det.boxes is not None:
                for box in det.boxes:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf  = box.conf[0].item()
                    cls_i = int(box.cls[0].item())
                    label = CLASS_NAMES[cls_i] if cls_i < len(CLASS_NAMES) else str(cls_i)

                    color = 'red' if conf > 0.7 else 'orange'
                    rect  = patches.Rectangle(
                        (x1,y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f"{label} {conf*100:.0f}%",
                            color='white', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor=color, alpha=0.8))
                    n_det += 1

            title_color = 'red' if n_det > 0 else 'green'
            ax.set_title(
                f"⚠️  {n_det} hazard(s) detected" if n_det > 0
                else "✅  Road clear",
                color=title_color, fontsize=9, fontweight='bold'
            )

        plt.tight_layout()
        plt.savefig('pothole_detections.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("💾  Saved: pothole_detections.png")

# ─────────────────────────────────────────────────────────────
# STEP 5 │ Compare with COCO Pretrained
# ─────────────────────────────────────────────────────────────

print("\n📊  Comparison: Before vs After fine-tuning ...")

# COCO pretrained (no fine-tuning)
coco_model = YOLO('yolov8n.pt')
coco_metrics = coco_model.val(data=data_yaml, verbose=False)
coco_map50   = coco_metrics.box.map50 * 100

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

models_l = ['YOLOv8 (COCO only)\nNo fine-tuning',
            'YOLOv8 (Fine-tuned)\nOn pothole dataset']
maps     = [coco_map50, map50]
colors   = ['#4C72B0', '#DD4444']
bars     = ax.bar(models_l, maps, color=colors, width=0.45,
                  edgecolor='white', linewidth=1.5)

ax.set_ylabel('mAP@50 (%)', fontsize=12, color='white')
ax.set_title('Fine-tuning Impact — Pothole Detection\nCOCO Pretrained vs Fine-tuned on Road Data',
             fontsize=13, fontweight='bold', color='white', pad=15)
ax.set_ylim([0, 105])
ax.tick_params(colors='white', labelsize=10)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#444466')
ax.grid(axis='y', alpha=0.2, color='white')
for b, m in zip(bars, maps):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
            f'{m:.1f}%', ha='center', fontweight='bold',
            fontsize=13, color='white')
plt.tight_layout()
plt.savefig('pothole_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("💾  Saved: pothole_comparison.png")

# ─────────────────────────────────────────────────────────────
# STEP 6 │ Save best model path for live detector
# ─────────────────────────────────────────────────────────────

import shutil
shutil.copy(best_model_path, 'pothole_best.pt')
print("\n💾  Best model saved as: pothole_best.pt")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print(f"""
{'='*60}
  🏆  POTHOLE DETECTOR SUMMARY
{'='*60}
  Model         : YOLOv8n (fine-tuned on road hazard data)
  mAP@50        : {map50:.1f}%
  Precision     : {precision:.1f}%
  Recall        : {recall:.1f}%
  Training time : {elapsed:.1f} minutes

  📁  Output files:
  ✅  pothole_best.pt          ← saved model
  ✅  pothole_detections.png   ← sample detections
  ✅  pothole_comparison.png   ← before/after chart

  🚀  Next step:
      python pothole_live.py
      → Opens webcam for live road hazard detection
{'='*60}
""")
