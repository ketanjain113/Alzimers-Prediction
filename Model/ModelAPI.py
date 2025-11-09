from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

model = load_model("Alzimer.keras")
class_labels = ["Alzheimer’s Disease", "Cognitively Normal", "EMCI", "LMCI"]

app = Flask(__name__)
CORS(app)  # allow frontend request

@app.post("/predict")
def predict():
    file = request.files["image"]
    img = tf.keras.preprocessing.image.load_img(file, target_size=(128,128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    pred = model.predict(np.expand_dims(img_array, 0))
    result = class_labels[np.argmax(pred)]
    conf = float(np.max(pred))
    return jsonify({"prediction": result, "confidence": conf})

if __name__ == "__main__":
    app.run(debug=True)
