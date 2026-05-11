from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler and class names once on startup
interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "patient_risk_model.tflite"))
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load(os.path.join(BASE_DIR, "patient_scaler.save"))

with open(os.path.join(BASE_DIR, "class_names.json")) as f:
    class_names = json.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json()
        features = data["features"]  # list of 10 numbers

        x        = np.array([features], dtype=np.float32)
        x_scaled = scaler.transform(x).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], x_scaled)
        interpreter.invoke()

        probs    = interpreter.get_tensor(output_details[0]["index"])[0]
        top_idx  = int(np.argmax(probs))

        prob_dict = {
            class_names[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(class_names))
        }

        return jsonify({
            "top_label":     class_names[top_idx],
            "confidence":    round(float(probs[top_idx]) * 100, 2),
            "probabilities": prob_dict,
            "error":         None
        })

    except Exception as e:
        return jsonify({
            "top_label":     None,
            "confidence":    0,
            "probabilities": {},
            "error":         str(e)
        }), 500

@app.route("/")
def index():
    return "LifeMatrix TFLite API is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
