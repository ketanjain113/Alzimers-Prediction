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

logging.basicConfig(level=logging.INFO)
logging.info(f"Python executable: {os.sys.executable}")
logging.info(f"tensorflow version: {tf.__version__}")
try:
    import keras
    logging.info(f"keras version: {keras.__version__}")
except Exception:
    logging.info("keras (standalone) not available; using tf.keras")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = osp.join(BASE_DIR, "Alzimer.keras")

# Support fetching a model at startup when deployed to platforms like Railway.
# If MODEL_URL is set, download it to MODEL_PATH (if not already present).
MODEL_URL = os.environ.get('MODEL_URL')
def _maybe_download_model(url, dest_path):
    try:
        if osp.exists(dest_path):
            logging.info("Model already present at %s, skipping download", dest_path)
            return
        logging.info("Downloading model from %s to %s", url, dest_path)
        # use urllib to avoid adding requests as a dependency
        try:
            from urllib.request import urlopen
            with urlopen(url) as resp:
                data = resp.read()
                with open(dest_path, 'wb') as f:
                    f.write(data)
            logging.info("Model downloaded successfully")
        except Exception:
            # fallback to urllib.request.urlretrieve which may handle redirects
            try:
                from urllib.request import urlretrieve
                urlretrieve(url, dest_path)
                logging.info("Model downloaded successfully (urlretrieve)")
            except Exception:
                logging.exception("Failed to download model from URL")
                raise
    except Exception:
        logging.exception("Error while attempting to download model")

if MODEL_URL:
    _maybe_download_model(MODEL_URL, MODEL_PATH)


def try_load_model(path):
    """Simple model loader with safe diagnostics.

    - Tries a straight load_model(path, compile=False).
    - On failure, attempts to read the model_config from an HDF5 file
      (if h5py is available) and logs compact diagnostics.
    - Does NOT perform runtime monkeypatches.
    """
    try:
        logging.info(f"Attempting to load model from: {path}")
        model = load_model(path, compile=False)
        logging.info("Model loaded successfully")
        return model
    except Exception:
        logging.exception("Model load failed — attempting safe diagnostics")

        # Try HDF5 diagnostics only if h5py is installed
        try:
            import h5py, json
        except Exception:
            logging.info("h5py not installed; skipping HDF5 diagnostics. Install h5py for more details.")
            raise

        if not osp.exists(path):
            logging.error("Model file does not exist: %s", path)
            raise

        try:
            with h5py.File(path, 'r') as f:
                model_config = None
                if 'model_config' in f.attrs:
                    model_config = f.attrs['model_config']
                elif 'model_config' in f:
                    model_config = f['model_config'][()]

                if model_config is None:
                    logging.error("No 'model_config' found inside HDF5 model file — file may be SavedModel or use newer format.")
                else:
                    if isinstance(model_config, bytes):
                        model_config = model_config.decode('utf-8')
                    try:
                        cfg_json = json.loads(model_config)
                        logging.error("Model serialized config root keys: %s", list(cfg_json.keys()))

                        def _scan(node, path=()):
                            if isinstance(node, dict):
                                cls = node.get('class_name') or node.get('class')
                                if isinstance(cls, str) and 'Conv' in cls:
                                    logging.error("Found Conv layer candidate at %s: %s", '/'.join(path), node)
                                cfg = node.get('config')
                                if isinstance(cfg, dict) and 'dtype' in cfg:
                                    logging.error("Layer with dtype at %s: %s", '/'.join(path), cfg)
                                for k, v in node.items():
                                    _scan(v, path + (str(k),))
                            elif isinstance(node, list):
                                for i, it in enumerate(node):
                                    _scan(it, path + (str(i),))

                        _scan(cfg_json)
                    except Exception:
                        logging.exception("Failed to parse model_config JSON for diagnostics")
        except Exception:
            logging.exception("Error while attempting to read HDF5 model file for diagnostics")

        # Re-raise the original exception so the caller (start script) sees the failure
        raise


model = try_load_model(MODEL_PATH)

class_labels = [
    "Alzheimer's Disease: The scan shows significant brain tissue loss in memory and reasoning areas, consistent with advanced Alzheimer’s.",
    "Cognitively Normal: The brain structure appears healthy with no visible signs of shrinkage or abnormal patterns.",
    "Early Mild Cognitive Impairment (EMCI): Mild changes are visible in memory-related regions, suggesting early signs of cognitive decline.",
    "Late Mild Cognitive Impairment (LMCI): Noticeable shrinkage is present in key brain regions, indicating a later stage of cognitive impairment that may progress toward Alzheimer’s.",
]

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_path": MODEL_PATH}), 200
def preprocess_image_from_bytes(image_bytes, target_size=(128, 128)):
    """Load image bytes into a numpy array resized to target_size.

    Returns an HxWxC uint8 numpy array (C=3 for RGB) scaled to [0,1].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((int(target_size[0]), int(target_size[1])), Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0
    return arr


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept multipart form file 'image' or base64 JSON {image: 'data:...'}
        if 'image' in request.files:
            f = request.files['image']
            image_bytes = f.read()
        else:
            body = request.get_json(silent=True) or {}
            b64 = body.get('image')
            if not b64:
                return jsonify({"error": "No image provided. Send multipart form-data with field 'image' or JSON with base64 'image'."}), 400
            if isinstance(b64, str) and b64.startswith('data:'):
                b64 = b64.split(',', 1)[1]
            image_bytes = base64.b64decode(b64)

        # Determine target size from model input when possible
        target_h, target_w = 128, 128
        channels = 3
        channels_first = False
        try:
            inp_shape = None
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                inp_shape = model.input_shape
            elif hasattr(model, 'inputs') and model.inputs:
                inp_shape = tuple(model.inputs[0].shape.as_list())

            if inp_shape is not None:
                # inp_shape examples: (None, H, W, C) or (None, C, H, W)
                if len(inp_shape) == 4:
                    _, a, b, c = inp_shape
                    if a in (1, 3):
                        channels_first = True
                        channels = int(a)
                        target_h = int(b) if b else target_h
                        target_w = int(c) if c else target_w
                    else:
                        target_h = int(a) if a else target_h
                        target_w = int(b) if b else target_w
                        channels = int(c) if c else channels
                elif len(inp_shape) == 3:
                    _, a, b = inp_shape
                    target_h = int(a) if a else target_h
                    target_w = int(b) if b else target_w

        except Exception:
            logging.exception("Unable to infer model input shape; using defaults")

        img_arr = preprocess_image_from_bytes(image_bytes, target_size=(target_h, target_w))

        # Ensure correct channel handling
        if channels == 1:
            # convert to grayscale
            if img_arr.ndim == 3 and img_arr.shape[2] == 3:
                img_arr = np.mean(img_arr, axis=2, keepdims=True)
        else:
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr]*3, axis=-1)

        # Create batch
        input_tensor = np.expand_dims(img_arr, axis=0)
        if channels_first:
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

        preds = model.predict(input_tensor)
        preds = np.asarray(preds)

        # Normalize to probabilities
        if preds.ndim == 2 and preds.shape[0] == 1:
            logits = preds[0]
        elif preds.ndim == 1:
            logits = preds
        else:
            logits = preds.reshape(-1)

        try:
            probs = tf.nn.softmax(logits).numpy()
        except Exception:
            # If model already returns probabilities, try to use them directly
            probs = logits.astype('float32')

        top_idx = int(np.argmax(probs))
        confidence = float(np.max(probs)) if probs.size > 0 else 0.0
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
    print(f"Starting Model API on 0.0.0.0:{port}, model_path={MODEL_PATH}")
    app.run(host="0.0.0.0", port=port, debug=debug)
