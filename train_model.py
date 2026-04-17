import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization, GlobalAveragePooling1D, Input
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import random
from tqdm import tqdm
from keras.utils import plot_model
from datetime import datetime
import random
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import soundfile as sf


SEED = 404931
print(f"\nGenerated seed for this run: {SEED}")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # disable nondeterminism
os.environ["TF_DETERMINISTIC_OPS"] = "1"      # force deterministic TF ops


# ---------------------- GPU CONFIG ----------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ---------------------- CONSTANTS ----------------------
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000    
AUDIO_LENGTH = 2        
NUM_CLASSES = 2
DATA_AUGMENTATION = False


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


# ---------------------- AUGMENTATIONS ----------------------
def add_gaussian_noise(audio, noise_level=0.001):
    return audio + noise_level * np.random.randn(len(audio))

def time_shift(audio, shift_max=0.15):
    shift = int(shift_max * SAMPLE_RATE * (np.random.rand() - 0.5))
    return np.roll(audio, shift)

def random_volume(audio, min_gain=0.95, max_gain=1.05):
    return audio * np.random.uniform(min_gain, max_gain)


# ---------------------- LOAD AUDIO DATA ----------------------
def load_audio_data(folder_path, label, augment=DATA_AUGMENTATION):
    data, labels = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    for file in tqdm(files, desc=f"Loading {folder_path}", unit="file"):
        path = os.path.join(folder_path, file)
        # 1) Load raw PCM int16
        audio, sr = sf.read(path, dtype='int16')
        # 2) Convert to float32 and normalize to [-1, 1]
        audio = audio.astype(np.float32) / 32768.0
        # 3) Enforce exact 32000 samples
        target_len = SAMPLE_RATE * AUDIO_LENGTH
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        # 4) Convert to TF tensor
        audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
        # 5) Extract Mel spectrogram using TF ops
        mel_spec = extract_mel_spectrogram_tf(audio_tf).numpy()
        data.append(mel_spec)
        labels.append(label)
        # ---------------------- AUGMENTATION ----------------------
        if augment and np.random.rand() < 0.5:
            aug_audio = audio.copy()
            if np.random.rand() < 0.3:
                aug_audio = time_shift(aug_audio)
            if np.random.rand() < 0.3:
                aug_audio = random_volume(aug_audio)
            if np.random.rand() < 0.15:
                aug_audio = add_gaussian_noise(aug_audio)
            aug_audio_tf = tf.convert_to_tensor(aug_audio, dtype=tf.float32)
            aug_mel_spec = extract_mel_spectrogram_tf(aug_audio_tf).numpy()
            data.append(aug_mel_spec)
            labels.append(label)
    return np.array(data), np.array(labels)


# ---------------------- LOAD DATASET ----------------------
print("Loading dataset...")
X_drone, y_drone = load_audio_data(os.path.join(DATASET_PATH, "train", "drone"), label=1)
X_back,  y_back  = load_audio_data(os.path.join(DATASET_PATH, "train", "background"), label=0)
X = np.concatenate([X_drone, X_back], axis=0)
y = np.concatenate([y_drone, y_back], axis=0)

# Shuffle
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print("Augmentation for this run:", DATA_AUGMENTATION)
print("Final dataset shape:", X.shape)
print("\n")


# ---------------------- MODEL ----------------------
input_shape = (X.shape[1], X.shape[2], 1)

model = Sequential([
    Input(shape=input_shape),

    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(0.0005), loss="binary_crossentropy", metrics=["accuracy", "AUC"])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ---------------------- TRAIN ----------------------
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping]
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

model_filename = f"model_ours_cnn2D_logmel_{timestamp}.h5"
model.save(model_filename)
print(f"Saved model as {model_filename}")

architecture_filename = f"model_architecture_{timestamp}.png"
plot_model(model, to_file=architecture_filename, show_shapes=True, show_layer_names=True)
print(f"Saved model architecture as {architecture_filename}")


# ---------------------- EVALUATION ----------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Background', 'Drone']))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Background', 'Drone'], yticklabels=['Background', 'Drone'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

conf_matrix_filename = f"training_confusionmatrix_{timestamp}.png"
plt.savefig(conf_matrix_filename, dpi=300, bbox_inches='tight')
print(f"Saved confusion matrix as {conf_matrix_filename}")

plt.show()

# ---------------------- PLOTS ----------------------
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.ylim(0, 1)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.ylim(0.5, 1.0) 

training_plot_filename = f"training_loss_accuracy_{timestamp}.png"
plt.savefig(training_plot_filename, dpi=300, bbox_inches='tight')
print(f"Saved training plot as {training_plot_filename}")

plt.show()


