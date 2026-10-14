# 🐾 PawsAI — Cat vs Dog CNN Classifier

A full-stack Deep Learning project that trains a **Convolutional Neural Network** to classify images as a cat or dog, and serves the results through a beautiful interactive web app.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **CNN Architecture** | 4 Conv2D + BatchNorm + MaxPool blocks → GlobalAvgPool → Dense |
| **Data Augmentation** | Flip, Rotation, Zoom, Brightness (training only) |
| **Training Callbacks** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| **Web UI** | Drag-and-drop upload, animated result card, confidence bar |
| **Flask API** | `/api/predict` endpoint (multipart/form-data) |
| **CLI Predict** | `python ml/predict.py <image>` for terminal use |

---

## 📂 Project Structure

```
CatDog_CNN_Safe_Starter/
├── app.py                  # Flask web server
├── requirements.txt        # Python dependencies
│
├── ml/
│   ├── train.py            # Model training script
│   ├── predict.py          # CLI prediction tool
│   ├── convert_to_tfjs.py  # Optional TF.js conversion guide
│   └── cat_dog_model.keras # (created after training)
│
├── data/
│   ├── train/
│   │   ├── cat/            # ← put training cat images here
│   │   └── dog/            # ← put training dog images here
│   └── validation/
│       ├── cat/            # ← put validation cat images here
│       └── dog/            # ← put validation dog images here
│
└── public/
    ├── index.html          # Web app (SPA)
    ├── style.css
    ├── app.js
    └── hero.jpg
```

---

## 🚀 Quick Start

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If PowerShell blocks script execution:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 2. Add your dataset

Download from **[Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)** (requires free account).

Place images in:
- `data/train/cat/` and `data/train/dog/`  *(≥ 1 000 images each recommended)*
- `data/validation/cat/` and `data/validation/dog/`  *(≥ 200 each)*

Do **not** put the same image in both train and validation.

### 3. Train the model

```bash
python ml/train.py
```

Training will:
- Validate that images exist in all four folders
- Train for up to 20 epochs with early stopping
- Save the best model → `ml/cat_dog_model.keras`
- Save training curves → `ml/training_history.png`

### 4. Launch the web app

```bash
python app.py
```

Open **http://localhost:5000** — drag, drop, classify!

---

## 🖥️ CLI Prediction

```bash
python ml/predict.py path/to/photo.jpg
```

Example output:
```
========================================
  Image    : photo.jpg
  Prediction: DOG
  Confidence: 94.3%  [██████████████████░░]
  Raw score : 0.9430  (0 = cat, 1 = dog)
========================================
```

---

## 🧠 Model Architecture

```
Input (128×128×3)
  → Rescaling [0,1] + RandomFlip + RandomRotation + RandomZoom + RandomBrightness
  → Conv2D(32)  + BatchNorm + MaxPool  → 64×64×32
  → Conv2D(64)  + BatchNorm + MaxPool  → 32×32×64
  → Conv2D(128) + BatchNorm + MaxPool  → 16×16×128
  → Conv2D(256) + BatchNorm + GlobalAvgPool → 256
  → Dense(256) + Dropout(0.5)
  → Dense(1, sigmoid)  →  0 = cat, 1 = dog
```

---

## ⚠️ Important Notes

- The model is trained from scratch on **your dataset** — it is not pre-trained.
- Confidence scores are raw sigmoid outputs and are not a guarantee of correctness.
- At least **1 000 images per class** in training is recommended for reliable results.
- This project has no database, no user accounts, and no cloud backend.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow >= 2.13` | CNN training and inference |
| `numpy` | Array manipulation |
| `pillow` | Image loading |
| `matplotlib` | Training curve plots |
| `flask` | Web server & REST API |
