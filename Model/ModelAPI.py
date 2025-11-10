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
                img_array = np.stack([img_array] * 3, axis=-1)

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
    print(f"Starting Model API on 0.0.0.0:{port}, model_path={MODEL_PATH}")
    app.run(host="0.0.0.0", port=port, debug=debug)
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

def try_load_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        logging.exception("Initial model load failed; attempting compatibility fallback")

    try:
        import sys
        import types
        from tensorflow.keras import layers as _layers
        _orig_from_config = _layers.InputLayer.from_config

        def _patched_from_config(cls, config, custom_objects=None):
            if isinstance(config, dict) and 'batch_shape' in config:
                config = dict(config)
                config['batch_input_shape'] = tuple(config.pop('batch_shape'))

            call_attempts = []
            call_attempts.append(lambda: _orig_from_config(cls, config, custom_objects))
            call_attempts.append(lambda: _orig_from_config(config, custom_objects))
            call_attempts.append(lambda: _orig_from_config(cls, config))
            call_attempts.append(lambda: _orig_from_config(config))
            # If the original has __func__, try that form too
            if hasattr(_orig_from_config, '__func__'):
                call_attempts.insert(0, lambda: _orig_from_config.__func__(cls, config, custom_objects))

            last_err = None
            for fn in call_attempts:
                try:
                    return fn()
                except TypeError as te:
                    last_err = te
                    continue
                except Exception:
                    # unexpected error — re-raise immediately to preserve traceback
                        # Keep loader simple: try a straight load. If it fails, produce safe
                        # diagnostics (HDF5 model_config) but do not attempt complex runtime
                        # monkeypatching — that caused more confusion than benefit.
                        try:
                            logging.info(f"Attempting to load model from: {path}")
                            model = load_model(path, compile=False)
                            logging.info("Model loaded successfully")
                            return model
                        except Exception as e:
                            logging.exception("Model load failed — will attempt safe diagnostics")

                            # Diagnostics: attempt to read model_config from HDF5 (.keras/.h5)
                            try:
                                import h5py, json
                            except Exception:
                                logging.info("h5py not available; cannot dump HDF5 model_config. Install h5py for diagnostics.")
                                # Re-raise original exception to signal failure to caller
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
                                        logging.error("No 'model_config' found inside HDF5 model file — file may be SavedModel or incompatible format")
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

                            # Re-raise original load exception so caller sees failure
                            raise
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
    print(f"Starting Model API on 0.0.0.0:{port}, model_path={MODEL_PATH}")
    app.run(host="0.0.0.0", port=port, debug=debug)
