import pyaudio
import wave
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import soundfile as sf

# ------------------------- CONFIG -------------------------
RAW_CHANNELS = 6       # ReSpeaker channels
TARGET_CHANNELS = 1
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2
CHUNK = 1024
TARGET_SAMPLES = SAMPLE_RATE * AUDIO_LENGTH  # 32000


MODEL_NAME = "model_ours_cnn2D_logmel_20251219_1629.h5"
prediction_threshold = 0.5


# ---------------------- FEATURE EXTRACTION ----------------------
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

def extract_mel_spectrogram_tf(audio):
    # audio: 1D float32 tensor, shape (TARGET_SAMPLES,)
    # 1) STFT
    stft = tf.signal.stft(
        audio,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )
    # 2) Magnitude
    magnitude = tf.abs(stft)
    # 3) Mel filterbank
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=SAMPLE_RATE / 2
    )
    mel_spec = tf.matmul(magnitude, mel_filterbank)
    # 4) Log compression (numerical stability only)
    mel_spec = tf.math.log(mel_spec + 1e-6)
    # 5) Channel dimension for CNN
    mel_spec = tf.expand_dims(mel_spec, axis=-1)

    return mel_spec



# ---------------------- LOAD MODEL ----------------------
try:
    model = load_model(MODEL_NAME)
    print(f"\033[92mModel loaded successfully: {MODEL_NAME}\033[0m")
except Exception as e:
    print(f"\033[91mERROR: Failed to load model: {MODEL_NAME}\033[0m")

    print(f"Reason: {e}")
    exit(1)



# ------------------------- DEVICE HELPERS -------------------------
def list_input_devices():
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxInputChannels") > 0:
            devices.append((i, info.get("name")))
    p.terminate()
    return devices

def find_input_device_index(keyword):
    p = pyaudio.PyAudio()
    device_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxInputChannels") > 0 and keyword.lower() in info.get("name").lower():
            device_idx = i
            break
    p.terminate()
    return device_idx

# ------------------------- PRINT DEVICES -------------------------
print("Available input devices:")
for i, name in list_input_devices():
    print(f"  ID {i}: {name}")

device_index = find_input_device_index("respeaker")
if device_index is None:
    print("Error: ReSpeaker device not found.")
    sys.exit(1)

# ------------------------- OPEN STREAM -------------------------
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=RAW_CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

print("Recording RAW merged audio... Press Ctrl+C to stop.")

# ------------------------- RECORD LOOP -------------------------
try:
    while True:
        raw_frames = []

        # Over-record slightly to guarantee enough samples
        for _ in range(int(SAMPLE_RATE / CHUNK * AUDIO_LENGTH) + 3):
            data = stream.read(CHUNK, exception_on_overflow=False)
            raw_frames.append(data)

        # Convert buffer → numpy
        audio = np.frombuffer(b"".join(raw_frames), dtype=np.int16).astype(np.int32)

        # Extract ONLY raw microphones (1–4)
        mic1 = audio[1::RAW_CHANNELS]
        mic2 = audio[2::RAW_CHANNELS]
        mic3 = audio[3::RAW_CHANNELS]
        mic4 = audio[4::RAW_CHANNELS]

        merged_audio = np.mean(np.stack([mic1, mic2, mic3, mic4], axis=0),axis=0)

        if len(merged_audio) > TARGET_SAMPLES:
            merged_audio = merged_audio[:TARGET_SAMPLES]
        elif len(merged_audio) < TARGET_SAMPLES:
            merged_audio = np.pad(merged_audio, (0, TARGET_SAMPLES - len(merged_audio)), mode='constant')

        GAIN = 1
        merged_audio = merged_audio * GAIN
        merged_audio = merged_audio.astype(np.int16)

        merged_audio = merged_audio.astype(np.float32) / 32768.0

        target_len = SAMPLE_RATE * AUDIO_LENGTH
        if len(merged_audio) > target_len:
            merged_audio = merged_audio[:target_len]
        elif len(merged_audio) < target_len:
            merged_audio = np.pad(merged_audio, (0, target_len - len(merged_audio)))

        merged_audio_tf = tf.convert_to_tensor(merged_audio, dtype=tf.float32)
        mel_spec = extract_mel_spectrogram_tf(merged_audio_tf).numpy()

        mel_spec = np.expand_dims(mel_spec, axis=0)


        # ---------------------- PREDICT ----------------------
        y_pred_probs = model.predict(mel_spec)
        prob = float(y_pred_probs[0][0])

        if prob >= prediction_threshold:
            label = "DRONE"
        else:
            label = "BACKGROUND"
        print(
            f"\033[92mPrediction:\033[0m {label} | "
            f"\033[94mConfidence:\033[0m {prob:.3f}"
        )







except KeyboardInterrupt:
    print("\nRecording stopped.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
