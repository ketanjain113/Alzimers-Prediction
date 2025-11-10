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
                    raise

            # If we reach here, all attempts raised TypeError — raise the last one
            if last_err is not None:
                raise last_err

        _layers.InputLayer.from_config = classmethod(_patched_from_config)
        logging.info("Patched InputLayer.from_config to accept 'batch_shape' and map to 'batch_input_shape'.")

        # Also inject a lightweight 'keras' shim into sys.modules so configs
        # that reference 'keras.*' (from standalone keras serialization) can
        # resolve to the tf.keras equivalents. We do this only temporarily.
        shim_keys = ['keras', 'keras.initializers', 'keras.regularizers', 'keras.layers', 'keras.utils', 'keras.models']
        saved_modules = {k: sys.modules.get(k) for k in shim_keys}
        # Prepare to possibly patch mixed_precision.get_policy to accept
        # string dtype names during deserialization. We'll restore it later.
        saved_get_policy = None
        try:
            sys.modules['keras'] = types.ModuleType('keras')
            # Map submodules to tf.keras modules where possible
            sys.modules['keras.initializers'] = tf.keras.initializers
            sys.modules['keras.regularizers'] = tf.keras.regularizers
            sys.modules['keras.layers'] = tf.keras.layers
            sys.modules['keras.utils'] = tf.keras.utils
            sys.modules['keras.models'] = tf.keras.models

            # Provide a shim for keras.DTypePolicy (and keras.mixed_precision)
            # so deserialization of dtype policies returns a tf.keras Policy.
            try:
                import types as _types
                km = sys.modules.get('keras') or _types.ModuleType('keras')
                def _make_dtype_policy_class():
                    class DTypePolicyShim:
                        def __init__(self, name):
                            self.name = name
                        @classmethod
                        def from_config(cls, config):
                            # config may be dict {'name': 'float32'}
                            name = config.get('name') if isinstance(config, dict) else config
                            try:
                                return tf.keras.mixed_precision.Policy(name)
                            except Exception:
                                return DTypePolicyShim(name)
                    return DTypePolicyShim

                km.DTypePolicy = _make_dtype_policy_class()
                # Also expose mixed_precision submodule mapping
                sys.modules['keras.mixed_precision'] = tf.keras.mixed_precision
                sys.modules['keras'] = km
            except Exception:
                logging.exception("Failed to insert DTypePolicy shim into keras module")

            logging.info("Inserted keras -> tf.keras shim modules into sys.modules for deserialization compatibility.")

            try:
                mp = tf.keras.mixed_precision
                if hasattr(mp, 'policy') and hasattr(mp.policy, 'get_policy'):
                    saved_get_policy = mp.policy.get_policy
                    def _patched_get_policy(dtype):
                        try:
                            if isinstance(dtype, str):
                                return tf.keras.mixed_precision.Policy(dtype)
                            return saved_get_policy(dtype)
                        except Exception:
                            if isinstance(dtype, str):
                                return tf.keras.mixed_precision.Policy(dtype)
                            raise
                    mp.policy.get_policy = _patched_get_policy
                    logging.info("Patched tf.keras.mixed_precision.policy.get_policy to accept string dtype names.")
            except Exception:
                logging.exception("Failed to patch mixed_precision.get_policy")

            model = load_model(path, compile=False)
            return model
        finally:
            for k, v in saved_modules.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
                try:
                    if saved_get_policy is not None:
                        try:
                            tf.keras.mixed_precision.policy.get_policy = saved_get_policy
                        except Exception:
                            try:
                                tf.keras.mixed_precision.policy.get_policy = saved_get_policy
                            except Exception:
                                pass
                except Exception:
                    logging.exception("Failed to restore mixed_precision.get_policy during cleanup")
    except Exception:
        logging.exception("Compatibility fallback failed")
        try:
            import h5py, json
            if osp.exists(path):
                try:
                    with h5py.File(path, 'r') as f:
                        model_config = None
                        if 'model_config' in f.attrs:
                            model_config = f.attrs['model_config']
                        elif 'model_config' in f:
                            # dataset
                            model_config = f['model_config'][()]

                        if model_config is not None:
                            try:
                                if isinstance(model_config, bytes):
                                    model_config = model_config.decode('utf-8')
                                cfg_json = json.loads(model_config)
                                # write a compact dump to logs
                                logging.error("Model serialized config (root keys): %s", list(cfg_json.keys()))
                                def _scan_layers(node, path=()):
                                    if isinstance(node, dict):
                                        for k, v in node.items():
                                            if k == 'class_name' and v and 'Conv' in v:
                                                logging.error("Found layer candidate at %s: %s", '/'.join(path), node)
                                            elif k == 'config' and isinstance(v, dict):
                                                if 'dtype' in v:
                                                    logging.error("Layer with dtype at %s: %s", '/'.join(path), v)
                                        for k, v in node.items():
                                            _scan_layers(v, path + (str(k),))
                                    elif isinstance(node, list):
                                        for idx, item in enumerate(node):
                                            _scan_layers(item, path + (str(idx),))

                                _scan_layers(cfg_json)
                            except Exception:
                                logging.exception("Failed to parse model_config JSON for diagnostics")
                        else:
                            logging.error("No 'model_config' found in HDF5 file for extra diagnostics")
                except Exception:
                    logging.exception("Error while attempting to read HDF5 model file for diagnostics")
        except Exception:
            logging.info("h5py not available or diagnostics unavailable; skipping model_config dump")

        raise


model = try_load_model(MODEL_PATH)
class_labels = ["Alzheimer's Disease: The scan shows significant brain tissue loss in memory and reasoning areas, consistent with advanced Alzheimer’s.", 
                "Cognitively Normal: The brain structure appears healthy with no visible signs of shrinkage or abnormal patterns.", 
                "Early Mild Cognitive Impairment (EMCI): Mild changes are visible in memory-related regions, suggesting early signs of cognitive decline.", 
                "Late Mild Cognitive Impairment (LMCI): Noticeable shrinkage is present in key brain regions, indicating a later stage of cognitive impairment that may progress toward Alzheimer’s."]

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
