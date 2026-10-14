"""
Optional conversion guide.

This is deliberately not an automatic converter because TensorFlow.js conversion
support depends on your installed Python/TensorFlow versions. Train the model first.
Then, in a compatible environment, install tensorflowjs and use its converter.

Typical command (check current tensorflowjs documentation for compatibility):
    tensorflowjs_converter --input_format=keras ml/cat_dog_model.keras public/model/

The browser app must preprocess images exactly as the trained model expects.
This starter does not include a browser UI or a converted model.
"""
from pathlib import Path

model_path = Path(__file__).with_name("cat_dog_model.keras")
if not model_path.exists():
    raise SystemExit("Train first: python ml/train.py")
print("Model exists:", model_path)
print("Read this file's instructions before converting it for TensorFlow.js.")
