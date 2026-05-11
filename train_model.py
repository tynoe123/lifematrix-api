import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json

# =========================
# LOAD TRAINING DATA
# =========================
# CSV columns expected:
# avg_hr,max_hr,avg_spo2,min_spo2,avg_temp,max_temp,fall_count,
# high_hr_count,low_spo2_count,high_temp_count,label

df = pd.read_csv("patient_training_data.csv")

feature_cols = [
    "avg_hr",
    "max_hr",
    "avg_spo2",
    "min_spo2",
    "avg_temp",
    "max_temp",
    "fall_count",
    "high_hr_count",
    "low_spo2_count",
    "high_temp_count"
]

X = df[feature_cols].astype(float).values
y_labels = df["label"].astype(str)

class_names = sorted(y_labels.unique().tolist())
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for name, i in class_to_idx.items()}

y = np.array([class_to_idx[label] for label in y_labels])

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# HANDLE CLASS IMBALANCE
# =========================
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

# =========================
# BUILD MODEL
# =========================
model = keras.Sequential([
    layers.Input(shape=(len(feature_cols),)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# =========================
# EVALUATE
# =========================
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")

# =========================
# SAVE KERAS MODEL + SCALER + LABEL MAP
# =========================
model.save("patient_risk_model.h5")
joblib.dump(scaler, "patient_scaler.save")

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("Saved: patient_risk_model.h5, patient_scaler.save, class_names.json")

# =========================
# CONVERT TO TENSORFLOW LITE
# =========================
print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimise for size and speed (float16 quantisation keeps good accuracy)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("patient_risk_model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"TFLite model size: {len(tflite_model) / 1024:.1f} KB")

# =========================
# VERIFY TFLITE MODEL
# =========================
print("\nVerifying TFLite model with a sample input...")

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Use first validation sample as sanity check
sample = X_val[0:1].astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], sample)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])[0]
predicted_idx = int(np.argmax(output))
confidence    = float(output[predicted_idx]) * 100

print(f"  Predicted class : {class_names[predicted_idx]}")
print(f"  Confidence      : {confidence:.1f}%")
print(f"  True label      : {class_names[y_val[0]]}")

print("\nTraining and export complete.")
print("Files saved:")
print("  - patient_risk_model.h5      (Keras / full model)")
print("  - patient_risk_model.tflite  (TFLite — used by predict.py)")
print("  - patient_scaler.save        (StandardScaler)")
print("  - class_names.json           (label index map)")