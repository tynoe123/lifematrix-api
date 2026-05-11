#!/usr/bin/env python3
"""
predict.py — TFLite inference bridge for prediction.php
-------------------------------------------------------
Usage (called by PHP via shell_exec):
    python3 predict.py <avg_hr> <max_hr> <avg_spo2> <min_spo2> \
                       <avg_temp> <max_temp> <fall_count>       \
                       <high_hr_count> <low_spo2_count> <high_temp_count>

Outputs a single JSON object to stdout:
{
    "top_label":    "Atrial Fibrillation Pattern",
    "confidence":   87.4,
    "probabilities": {
        "Acute Coronary Syndrome": 0.2,
        "Atrial Fibrillation Pattern": 87.4,
        ...
    },
    "error": null
}
"""

import sys
import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_resources():
    import joblib, tensorflow as tf

    model_path  = os.path.join(BASE_DIR, "patient_risk_model.tflite")
    scaler_path = os.path.join(BASE_DIR, "patient_scaler.save")
    labels_path = os.path.join(BASE_DIR, "class_names.json")

    for p in (model_path, scaler_path, labels_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    scaler = joblib.load(scaler_path)

    with open(labels_path) as f:
        class_names = json.load(f)

    return interpreter, scaler, class_names


def predict(features: list) -> dict:
    interpreter, scaler, class_names = load_resources()

    x = np.array([features], dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], x_scaled)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]['index'])[0]

    # Build probability dict (percentage, 2 decimal places)
    prob_dict = {
        class_names[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(class_names))
    }

    top_idx   = int(np.argmax(probs))
    top_label = class_names[top_idx]
    confidence = round(float(probs[top_idx]) * 100, 2)

    return {
        "top_label":     top_label,
        "confidence":    confidence,
        "probabilities": prob_dict,
        "error":         None
    }


def main():
    if len(sys.argv) != 11:
        print(json.dumps({
            "top_label": None,
            "confidence": 0,
            "probabilities": {},
            "error": f"Expected 10 feature arguments, got {len(sys.argv) - 1}"
        }))
        sys.exit(1)

    try:
        features = [float(v) for v in sys.argv[1:]]
        result   = predict(features)
    except Exception as e:
        result = {
            "top_label":     None,
            "confidence":    0,
            "probabilities": {},
            "error":         str(e)
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()