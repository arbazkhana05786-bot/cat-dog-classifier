"""
Cat vs Dog CNN — Training Script
---------------------------------
Trains a binary image classifier using TensorFlow/Keras.
Output:
  ml/cat_dog_model.keras   – saved model
  ml/training_history.png  – accuracy & loss curves
"""

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
TRAIN_DIR  = ROOT / "data" / "train"
VAL_DIR    = ROOT / "data" / "validation"
MODEL_PATH = ROOT / "ml" / "cat_dog_model.keras"
PLOT_PATH  = ROOT / "ml" / "training_history.png"

# ── Hyper-parameters ─────────────────────────────────────────────────────────
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 20
SEED       = 42

# ── Pre-flight checks ─────────────────────────────────────────────────────────
for folder in (TRAIN_DIR / "cat", TRAIN_DIR / "dog",
               VAL_DIR / "cat",   VAL_DIR / "dog"):
    if not folder.exists() or not any(folder.iterdir()):
        raise SystemExit(
            f"\n❌  No images found in: {folder}\n"
            "    Add cat/dog images to the data/ folders — see README.md for details.\n"
            "    Dataset source: https://www.kaggle.com/c/dogs-vs-cats/data\n"
        )

print("✅  Data folders validated.")

# ── Datasets ──────────────────────────────────────────────────────────────────
def make_dataset(directory, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        class_names=["cat", "dog"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        label_mode="binary",
        shuffle=shuffle,
    )

train_ds = make_dataset(TRAIN_DIR, shuffle=True)
val_ds   = make_dataset(VAL_DIR,   shuffle=False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    # Use pre-trained MobileNetV2 backbone (trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pre-trained feature extractor

    inputs = layers.Input(shape=(128, 128, 3))
    # MobileNetV2 expects pixels scaled to [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.10)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="CatDog_MobileNetV2")

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

# ── Training ──────────────────────────────────────────────────────────────────
print(f"\n🚀  Training for up to {EPOCHS} epochs …\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb_list,
)

# ── Plot training curves ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Cat vs Dog CNN — Training History", fontsize=14)

axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"📊  Training curves saved to: {PLOT_PATH}")

# ── Done ──────────────────────────────────────────────────────────────────────
final_val_acc = max(history.history["val_accuracy"])
print(f"\n✅  Best validation accuracy: {final_val_acc:.2%}")
print(f"💾  Model saved to: {MODEL_PATH}")
print("    Class mapping: 0 = cat, 1 = dog")
print("\n    Next steps:")
print("      • Predict a single image:  python ml/predict.py path/to/image.jpg")
print("      • Launch the web app:      python app.py")
