from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import logging
import os
import os.path as osp

# Resolve model path relative to this file so the script works whether started
# from the repo root or from the Model directory.
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = osp.join(BASE_DIR, "Alzimer.keras")
model = load_model(MODEL_PATH, compile=False)
class_labels = ["Alzheimer's Disease: The scan shows significant brain tissue loss in memory and reasoning areas, consistent with advanced Alzheimer’s.", 
                "Cognitively Normal: The brain structure appears healthy with no visible signs of shrinkage or abnormal patterns.", 
                "Early Mild Cognitive Impairment (EMCI): Mild changes are visible in memory-related regions, suggesting early signs of cognitive decline.", 
                "Late Mild Cognitive Impairment (LMCI): Noticeable shrinkage is present in key brain regions, indicating a later stage of cognitive impairment that may progress toward Alzheimer’s."]

app = Flask(__name__)
CORS(app) 


def preprocess_image_from_bytes(image_bytes, target_size=(128, 128)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))
    arr = np.array(img).astype("float32") / 255.0
    return arr


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" in request.files:
            file = request.files["image"]
            image_bytes = file.read()
        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get("image") or data.get("image_base64")
            if not b64:
                return jsonify({"error": "No image provided. Send multipart form 'image' or JSON with 'image' (base64)."}), 400
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)

        input_shape = getattr(model, 'input_shape', None)
        logging.info(f"Model input_shape: {input_shape}")

        target_h, target_w, channels = 128, 128, 3
        channels_first = False

        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
            if len(input_shape) == 4:
                _, a, b, c = input_shape
                if a in (1, 3):
                    channels_first = True
                    channels = int(a)
                    target_h = int(b) if b is not None else target_h
                    target_w = int(c) if c is not None else target_w
                else:
                    target_h = int(a) if a is not None else target_h
                    target_w = int(b) if b is not None else target_w
                    channels = int(c) if c is not None else channels
            elif len(input_shape) == 3:
                _, a, b = input_shape
                target_h = int(a) if a is not None else target_h
                target_w = int(b) if b is not None else target_w
                channels = 1

        mode = "RGB" if channels == 3 else "L"

        img_array = preprocess_image_from_bytes(image_bytes, target_size=(target_h, target_w))

        if mode == "L":
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                img_array = np.mean(img_array, axis=2, keepdims=True)
        else:
            if img_array.ndim == 2:
                img_array = np.stack([img_array]*3, axis=-1)

        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, -1)

        input_tensor = np.expand_dims(img_array, axis=0)  

        if channels_first:
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

        preds = model.predict(input_tensor)

        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = tf.nn.softmax(preds[0]).numpy()
        else:
            probs = tf.nn.softmax(preds.reshape(-1)).numpy()

        top_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label = class_labels[top_idx] if top_idx < len(class_labels) else str(top_idx)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 6),
            "probabilities": probs.tolist(),
        })

    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Prefer MODEL_PORT for the model service so it doesn't collide with the
    # container-level $PORT that hosting providers (e.g. Railway) supply to the
    # primary web process. Fallback to PORT then to 5000.
    port = int(os.environ.get("MODEL_PORT", os.environ.get("PORT", 5000)))
    debug_env = os.environ.get("DEBUG", "false").lower()
    debug = debug_env in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
