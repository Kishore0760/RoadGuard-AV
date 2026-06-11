# =============================================================================
# 🚗  AUTONOMOUS VEHICLE PERCEPTION SYSTEM — TRAINER (H200)
# =============================================================================
# Trains a YOLOv8 model on road hazard data (potholes, cracks, speed bumps)
# Combined with COCO pretrained model for vehicles, pedestrians, traffic signs
#
# RUN IN JUPYTERHUB TERMINAL:
#   pip install ultralytics roboflow pyyaml
#   python av_trainer_h200.py
#
# EXPECTED TIME ON H200: ~10-15 minutes
# =============================================================================

import os
import time
import yaml
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

print("=" * 65)
print("  🚗  AV PERCEPTION SYSTEM — H200 TRAINER")
print("  Road Hazard Detection + Traffic Object Detection")
print("=" * 65)

# ── GPU Check ─────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️   Device: {device}")
if device == 'cuda':
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("  ✅  H200 detected!")
else:
    print("  ⚠️  No GPU — training will be slow on CPU")

# ── H200 Config ───────────────────────────────────────────────
PROJECT_DIR  = "av_project"
DATA_DIR     = os.path.join(PROJECT_DIR, "dataset")
EPOCHS       = 100      # H200: 100 epochs in ~10 min
BATCH_SIZE   = 64       # H200: 64 vs 16 on CPU
IMG_SIZE     = 640      # YOLOv8 standard
WORKERS      = 4        # H200: 4 parallel data workers
PATIENCE     = 20       # Early stopping

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"""
⚙️   H200 Training Config:
  Epochs      : {EPOCHS}
  Batch size  : {BATCH_SIZE}
  Image size  : {IMG_SIZE}x{IMG_SIZE}
  Workers     : {WORKERS}
  Precision   : Mixed (AMP)
  Device      : {device}
""")

# ─────────────────────────────────────────────────────────────
# STEP 1 │ Dataset Download
# ─────────────────────────────────────────────────────────────

API_KEY  = "YOUR_API_KEY_HERE"   # ← paste free Roboflow key here
data_yaml = os.path.join(DATA_DIR, "data.yaml")

if not os.path.exists(data_yaml):
    if API_KEY == "YOUR_API_KEY_HERE":
        print("""
╔══════════════════════════════════════════════════════════════╗
║  DATASET SETUP REQUIRED                                      ║
║                                                              ║
║  1. Go to: https://roboflow.com  →  Sign up free             ║
║  2. Settings → API  →  Copy your API key                     ║
║  3. Open this file, replace:                                 ║
║       API_KEY = "YOUR_API_KEY_HERE"                          ║
║     with your key                                            ║
║  4. Run again                                                ║
║                                                              ║
║  OR manual download:                                         ║
║  https://universe.roboflow.com/brad-dwyer/pothole-3vwgc      ║
║  Download as YOLOv8 format → extract to av_project/dataset/ ║
╚══════════════════════════════════════════════════════════════╝
        """)
        exit(0)

    try:
        from roboflow import Roboflow
        print("📥  Downloading road hazard dataset from Roboflow ...")
        rf      = Roboflow(api_key=API_KEY)
        project = rf.workspace("brad-dwyer").project("pothole-3vwgc")
        dataset = project.version(1).download("yolov8", location=DATA_DIR)
        print("✅  Dataset downloaded!")
    except Exception as e:
        print(f"❌  Download failed: {e}")
        exit(1)
else:
    print("✅  Dataset already exists — skipping download")

# Read class info
with open(data_yaml) as f:
    data_info  = yaml.safe_load(f)
CLASS_NAMES = data_info.get('names', ['pothole'])
NUM_CLASSES = len(CLASS_NAMES)
print(f"📂  Road hazard classes ({NUM_CLASSES}): {CLASS_NAMES}")

# Count dataset images
for split in ['train', 'valid', 'test']:
    img_dir = os.path.join(DATA_DIR, split, 'images')
    if os.path.exists(img_dir):
        n = len([f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f"    {split:<8}: {n} images")

# ─────────────────────────────────────────────────────────────
# STEP 2 │ Train YOLOv8 Road Hazard Detector
# ─────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  🚀 Training YOLOv8 Road Hazard Detector")
print(f"     {EPOCHS} epochs  |  batch {BATCH_SIZE}  |  {device}")
print(f"{'='*65}\n")

hazard_model = YOLO('yolov8n.pt')
t0 = time.time()

hazard_model.train(
    data      = data_yaml,
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = BATCH_SIZE,
    workers   = WORKERS,
    device    = 0 if device == 'cuda' else 'cpu',
    project   = PROJECT_DIR,
    name      = 'road_hazard',
    exist_ok  = True,
    amp       = True,       # Mixed precision — H200 Tensor Cores
    patience  = PATIENCE,   # Stop if no improvement for 20 epochs
    # Augmentation for road conditions
    flipud    = 0.1,
    fliplr    = 0.5,
    mosaic    = 1.0,
    mixup     = 0.15,
    degrees   = 5.0,
    translate = 0.1,
    scale     = 0.5,
    hsv_h     = 0.015,
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    copy_paste = 0.1,
)

elapsed = (time.time() - t0) / 60
print(f"\n✅  Training complete in {elapsed:.1f} min on {device}")

# ─────────────────────────────────────────────────────────────
# STEP 3 │ Evaluate Both Models
# ─────────────────────────────────────────────────────────────

print("\n📊  Evaluating road hazard model ...")
best_path    = os.path.join(PROJECT_DIR, 'road_hazard', 'weights', 'best.pt')
trained      = YOLO(best_path)
metrics      = trained.val(data=data_yaml, verbose=False)

map50     = metrics.box.map50  * 100
map5095   = metrics.box.map    * 100
precision = metrics.box.mp     * 100
recall    = metrics.box.mr     * 100

print(f"\n{'='*50}")
print(f"  📊  ROAD HAZARD MODEL RESULTS")
print(f"{'='*50}")
print(f"  mAP@50       : {map50:.1f}%")
print(f"  mAP@50-95    : {map5095:.1f}%")
print(f"  Precision    : {precision:.1f}%")
print(f"  Recall       : {recall:.1f}%")
print(f"  Training time: {elapsed:.1f} min on {device}")
print(f"{'='*50}")

# COCO model for comparison
print("\n📊  Evaluating COCO pretrained (baseline) ...")
coco_model   = YOLO('yolov8n.pt')
coco_metrics = coco_model.val(data=data_yaml, verbose=False)
coco_map50   = coco_metrics.box.map50 * 100

# ─────────────────────────────────────────────────────────────
# STEP 4 │ Comparison Chart
# ─────────────────────────────────────────────────────────────

BG = '#1a1a2e'
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

# Bar chart — before vs after
ax1 = axes[0]
ax1.set_facecolor(BG)
models   = ['COCO Pretrained\n(No fine-tuning)',
            'Fine-tuned\n(Road Hazard Data)']
maps     = [coco_map50, map50]
colors   = ['#4C72B0', '#DD4444']
bars     = ax1.bar(models, maps, color=colors, width=0.4,
                   edgecolor='white', linewidth=1.5)
ax1.set_ylabel('mAP@50 (%)', color='white', fontsize=11)
ax1.set_title('Fine-tuning Impact\nCOCO vs Road Hazard Training',
              color='white', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 105])
ax1.tick_params(colors='white')
ax1.spines[['top','right']].set_visible(False)
ax1.spines[['left','bottom']].set_color('#444466')
ax1.grid(axis='y', alpha=0.2, color='white')
for b, m in zip(bars, maps):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
             f'{m:.1f}%', ha='center', fontweight='bold',
             fontsize=13, color='white')

# Metrics bar
ax2 = axes[1]
ax2.set_facecolor(BG)
metric_names = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']
metric_vals  = [map50, map5095, precision, recall]
m_colors     = ['#DD4444','#E87B2B','#2B9E5E','#2B7FBF']
bars2 = ax2.bar(metric_names, metric_vals, color=m_colors,
                width=0.5, edgecolor='white', linewidth=1.5)
ax2.set_ylabel('Score (%)', color='white', fontsize=11)
ax2.set_title('Model Metrics Breakdown\nRoad Hazard Detector',
              color='white', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 105])
ax2.tick_params(colors='white')
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['left','bottom']].set_color('#444466')
ax2.grid(axis='y', alpha=0.2, color='white')
for b, m in zip(bars2, metric_vals):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
             f'{m:.1f}%', ha='center', fontweight='bold',
             fontsize=12, color='white')

plt.tight_layout()
plt.savefig('av_model_results.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("💾  Saved: av_model_results.png")

# ─────────────────────────────────────────────────────────────
# STEP 5 │ Sample Detections on Test Images
# ─────────────────────────────────────────────────────────────

test_dir = os.path.join(DATA_DIR, 'test', 'images')
if not os.path.exists(test_dir):
    test_dir = os.path.join(DATA_DIR, 'valid', 'images')

if os.path.exists(test_dir):
    test_imgs = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg','.jpeg','.png'))
    ][:6]

    if test_imgs:
        from PIL import Image
        import matplotlib.patches as patches

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
        fig.suptitle('🚗  Road Hazard Detections — H200 Trained Model',
                     fontsize=16, fontweight='bold', color='white')

        for ax, img_path in zip(axes.flat, test_imgs):
            det     = trained(img_path, verbose=False, conf=0.25)[0]
            img_arr = np.array(Image.open(img_path).convert('RGB'))
            ax.imshow(img_arr); ax.axis('off')
            ax.set_facecolor('#16213e')

            n = 0
            if det.boxes is not None:
                for box in det.boxes:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf  = box.conf[0].item()
                    cls_i = int(box.cls[0].item())
                    lbl   = (CLASS_NAMES[cls_i]
                             if cls_i < len(CLASS_NAMES) else str(cls_i))
                    color = 'red' if conf > 0.7 else 'orange'
                    rect  = patches.Rectangle(
                        (x1,y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1-6, f"{lbl} {conf*100:.0f}%",
                            fontsize=8, fontweight='bold', color='white',
                            bbox=dict(facecolor=color, alpha=0.8,
                                      boxstyle='round,pad=0.2'))
                    n += 1

            ax.set_title(
                f"⚠️  {n} hazard(s)" if n > 0 else "✅  Road clear",
                color='red' if n > 0 else 'green',
                fontsize=10, fontweight='bold', pad=6
            )

        plt.tight_layout()
        plt.savefig('av_sample_detections.png', dpi=150,
                    bbox_inches='tight', facecolor=BG)
        plt.close()
        print("💾  Saved: av_sample_detections.png")

# ─────────────────────────────────────────────────────────────
# STEP 6 │ Save model for inference
# ─────────────────────────────────────────────────────────────

shutil.copy(best_path, 'av_road_hazard.pt')
print("\n💾  Model saved as: av_road_hazard.pt")

# GPU memory report
if device == 'cuda':
    print(f"\n💾  GPU VRAM used: "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

print(f"""
{'='*65}
  🏆  TRAINING COMPLETE
{'='*65}
  Road Hazard mAP@50 : {map50:.1f}%
  COCO baseline      : {coco_map50:.1f}%
  Improvement        : +{map50-coco_map50:.1f}%
  Training time      : {elapsed:.1f} min on {device}

  📁  Output files:
  ✅  av_road_hazard.pt         ← trained model
  ✅  av_model_results.png      ← comparison chart
  ✅  av_sample_detections.png  ← test detections

  🚀  Next step:
      python av_demo_h200.py    ← process video on H200
      python av_live_laptop.py  ← live camera on your PC
{'='*65}
""")
