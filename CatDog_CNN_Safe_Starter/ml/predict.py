"""
Cat vs Dog CNN — Prediction CLI
---------------------------------
Usage:
    python ml/predict.py path/to/image.jpg

Returns the predicted class (cat / dog) and a confidence score.
"""

from pathlib import Path
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import tensorflow as tf

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Predict cat or dog from an image.")
parser.add_argument("image", help="Path to a JPG / PNG image file")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ml" / "cat_dog_model.keras"
IMAGE_PATH = Path(args.image)

if not MODEL_PATH.exists():
    sys.exit("❌  Trained model not found. Run: python ml/train.py")
if not IMAGE_PATH.is_file():
    sys.exit(f"❌  Image not found: {IMAGE_PATH}")

# ── Load model & predict ──────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)

image = tf.keras.utils.load_img(IMAGE_PATH, target_size=(128, 128))
array = tf.keras.utils.img_to_array(image)          # shape (128,128,3), 0-255
array = np.expand_dims(array, axis=0)               # shape (1,128,128,3)

raw_score = float(model.predict(array, verbose=0)[0][0])   # 0=cat, 1=dog

if raw_score >= 0.5:
    label      = "dog"
    confidence = raw_score
else:
    label      = "cat"
    confidence = 1.0 - raw_score

# ── Output ────────────────────────────────────────────────────────────────────
bar_filled = int(confidence * 20)
bar = "█" * bar_filled + "░" * (20 - bar_filled)

print(f"\n{'='*40}")
print(f"  Image    : {IMAGE_PATH.name}")
print(f"  Prediction: {label.upper()}")
print(f"  Confidence: {confidence:.1%}  [{bar}]")
print(f"  Raw score : {raw_score:.4f}  (0 = cat, 1 = dog)")
print(f"{'='*40}\n")
print("ℹ️   Note: confidence reflects the model score, not a factual guarantee.")
