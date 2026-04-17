import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from tqdm import tqdm
from datetime import datetime

# ---------------------- CONSTANTS ----------------------
MODEL_NAME = "model_ours_20251201_1246.h5"
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2
N_MFCC = 40   # MUST MATCH TRAINING

# ---------------------- FEATURE EXTRACTION ----------------------
def extract_mfcc_features(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        fmin=0,
        fmax=sr//2
    )
    mfcc = mfcc.T
    return (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

# ---------------------- LOAD AUDIO DATA ----------------------
def load_audio_data(folder_path, label):
    data, labels, filenames = [], [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    print(f"\nTotal samples in {folder_path}: {len(files)}")

    for file in tqdm(files, desc=f"Processing {os.path.basename(folder_path)}"):
        file_path = os.path.join(folder_path, file)
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Force exact 2 seconds length
        target_length = SAMPLE_RATE * AUDIO_LENGTH
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        mfcc_features = extract_mfcc_features(audio, sr)
        data.append(mfcc_features)
        labels.append(label)
        filenames.append(file)

    return np.array(data), np.array(labels), filenames

# ---------------------- LOAD TEST DATA ----------------------
print("Loading test dataset...\n")

X_drone, y_drone, files_drone = load_audio_data(os.path.join(DATASET_PATH, "test", "drone"), label=1)
X_bg, y_bg, files_bg = load_audio_data(os.path.join(DATASET_PATH, "test", "background"), label=0)

X_test = np.concatenate([X_drone, X_bg])
y_test = np.concatenate([y_drone, y_bg])
filenames = np.concatenate([files_drone, files_bg])

print(f"\nFinal test set shape: {X_test.shape}")

# ---------------------- LOAD MODEL ----------------------
model = load_model(MODEL_NAME)
print("Model loaded.")

# ---------------------- PREDICT ----------------------
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# ---------------------- REPORTS ----------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Background", "Drone"]))

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Background", "Drone"],
            yticklabels=["Background", "Drone"])
plt.title("Confusion Matrix – Test Data")
plt.xlabel("Predicted")
plt.ylabel("True")

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
plt.savefig(f"testing_conf_matrix_{timestamp}.png", dpi=300)
plt.show()

