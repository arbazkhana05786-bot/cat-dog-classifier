"""
Cat vs Dog — Flask Web API
---------------------------
Serves a browser-based UI and an /api/predict endpoint.

Run:
    python app.py

Then open: http://localhost:5000
"""

from pathlib import Path
import io
import sys
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ml" / "cat_dog_model.keras"
STATIC_DIR = ROOT / "public"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# ── Load model once at startup ────────────────────────────────────────────────
model = None

def load_model():
    global model
    if MODEL_PATH.exists():
        print(f"[OK] Loading model from {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"[WARN] Failed to load model: {e}")
            model = None
    else:
        print("[INFO] No trained model found yet. Run: python ml/train.py to train one.")
        print("       The web UI is running at http://localhost:5000")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "model_ready": model is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Run `python ml/train.py` first, then restart the server."
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        img = tf.keras.utils.load_img(
            io.BytesIO(img_bytes), target_size=(128, 128)
        )
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)

        raw_score = float(model.predict(arr, verbose=0)[0][0])

        if raw_score >= 0.5:
            label      = "dog"
            confidence = raw_score
        else:
            label      = "cat"
            confidence = 1.0 - raw_score

        return jsonify({
            "label":      label,
            "confidence": round(confidence * 100, 1),
            "raw_score":  round(raw_score, 4),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    print("\n* PawsAI Cat vs Dog Web App is live!")
    print("* URL: http://localhost:5000\n")
    app.run(debug=False, port=5000, host="127.0.0.1")
