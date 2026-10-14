# 🐾 PawsAI — Cat vs Dog Image Classifier

A modern, full-stack Deep Learning project featuring a **Transfer Learning CNN (MobileNetV2)** that achieves **92.5% validation accuracy**, paired with an interactive dark-mode web application and a CLI prediction tool.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Architecture** | Pre-trained **MobileNetV2** backbone + Custom Dense Head |
| **Accuracy** | **92.5%** validation accuracy (99.4% training accuracy) |
| **Interactive Web UI** | Glassmorphism dark mode, drag-and-drop upload, confidence bar |
| **REST API** | Flask `/api/predict` (multipart/form-data) & `/api/status` |
| **CLI Tool** | Terminal predictor with visual confidence meter |
| **Auto Dataset Setup** | Script to automatically fetch & organize balanced datasets |

---

## 📂 Project Structure

```
CatDog_CNN_Safe_Starter/
├── app.py                  # Flask web server & REST API
├── download_data.py        # Automated dataset downloader
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
│
├── ml/
│   ├── train.py            # Model training script (MobileNetV2)
│   ├── predict.py          # Terminal prediction CLI
│   ├── convert_to_tfjs.py  # Optional TF.js converter guide
│   ├── cat_dog_model.keras # Saved trained model (92.5% acc)
│   └── training_history.png# Accuracy & loss curves plot
│
├── data/
│   ├── train/
│   │   ├── cat/            # 80 training cat photos
│   │   └── dog/            # 80 training dog photos
│   └── validation/
│       ├── cat/            # 20 validation cat photos
│       └── dog/            # 20 validation dog photos
│
└── public/                 # Web application frontend
    ├── index.html          # Single Page Interface
    ├── style.css           # Premium dark-mode styling
    ├── app.js              # Drag-and-drop & API integration
    └── hero.jpg            # Generated hero image
```

---

## 🚀 Quick Start Guide

### 1. Requirements

- Python 3.10 – 3.12 (Python 3.11 recommended)
- Dependencies installed:
  ```powershell
  pip install -r requirements.txt
  ```

---

### 2. 🌐 Launch the Web Application

Run the server from this folder:

```powershell
python app.py
```

Then open your browser at:
👉 **[http://localhost:5000](http://localhost:5000)**

**Features of the Web App:**
- **Live Status:** The top-right badge displays `"Model Ready"` when the model is loaded.
- **Drag & Drop:** Drop any JPG, PNG, or WebP photo into the upload box.
- **Instant Inference:** Displays Cat / Dog verdict, percentage confidence, and a gradient visual bar.

---

### 3. 🖥️ CLI Single Image Prediction

Predict any image directly from your terminal:

**Test on a Cat:**
```powershell
python ml/predict.py data/validation/cat/cat_001.jpg
```
*Output:*
```
========================================
  Image    : cat_001.jpg
  Prediction: CAT
  Confidence: 96.6%  [███████████████████░]
  Raw score : 0.0343  (0 = cat, 1 = dog)
========================================
```

**Test on a Dog:**
```powershell
python ml/predict.py data/validation/dog/dog_001.jpg
```
*Output:*
```
========================================
  Image    : dog_001.jpg
  Prediction: DOG
  Confidence: 91.3%  [██████████████████░░]
  Raw score : 0.9129  (0 = cat, 1 = dog)
========================================
```

---

### 4. 🧠 Retraining the Model

To retrain or train on new images:
```powershell
python ml/train.py
```

- Automatically validates image presence across all 4 data folders.
- Uses data augmentation (horizontal flip, random rotation).
- Employs `EarlyStopping` and `ReduceLROnPlateau` callbacks.
- Saves the best checkpoint to `ml/cat_dog_model.keras`.
- Generates loss/accuracy curves in `ml/training_history.png`.

---

### 5. 📥 Adding More Data

Run the automated data populator to fetch fresh images:
```powershell
python download_data.py
```
Or manually place your images into:
- `data/train/cat/` & `data/train/dog/`
- `data/validation/cat/` & `data/validation/dog/`

*(Source: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data))*

---

## 🏗️ Neural Network Architecture

```
Input Image (128 × 128 × 3, RGB, 0–255)
  │
  ▼
Rescaling Layer (maps [0, 255] → [-1, 1])
  │
  ▼
Data Augmentation (RandomFlip, RandomRotation)
  │
  ▼
MobileNetV2 Backbone (Frozen ImageNet feature extractor, ~2.2M params)
  │
  ▼
GlobalAveragePooling2D (reduces spatial dimensions to 1280 features)
  │
  ▼
Dense Layer (128 units, ReLU activation)
  │
  ▼
Dropout (0.3 rate for regularization)
  │
  ▼
Output Layer (1 unit, Sigmoid activation)
  │
  └─► Score < 0.5 = CAT  |  Score ≥ 0.5 = DOG
```

---

## 📡 REST API Reference

### `GET /api/status`
Checks if the model is loaded and ready.
- **Response:**
  ```json
  {
    "model_exists": true,
    "model_path": "CatDog_CNN_Safe_Starter/ml/cat_dog_model.keras",
    "model_ready": true
  }
  ```

### `POST /api/predict`
Accepts an image file via `multipart/form-data` with key `file`.
- **Response (200 OK):**
  ```json
  {
    "label": "dog",
    "confidence": 91.3,
    "raw_score": 0.9129
  }
  ```

---

## 📄 License
This project is open-source and educational under the MIT License.
