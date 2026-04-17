import tensorflow as tf
import sys
import os

MODEL_PATH = 'model_ours_20251201_1255.h5'

# Derive output name automatically
OUTPUT_TFLITE = os.path.splitext(MODEL_PATH)[0] + ".tflite"

# ---------------------------
# LOAD MODEL WITH WARNING
# ---------------------------
try:
    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

except Exception as e:
    print(f"\033[93m[WARNING] Failed to load model '{MODEL_PATH}'.\033[0m")
    print(f"\033[93mReason: {e}\033[0m")
    sys.exit(1)

# ---------------------------
# CONVERT TO TFLITE
# ---------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ---------------------------
# SAVE FILE
# ---------------------------
with open(OUTPUT_TFLITE, 'wb') as f:
    f.write(tflite_model)

print(f"\033[92mModel successfully converted and saved as: {OUTPUT_TFLITE}\033[0m")
