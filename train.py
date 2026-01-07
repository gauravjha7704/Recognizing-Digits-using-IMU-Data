# ------------------------------------------------------------
# train.py
# ------------------------------------------------------------
# IMU Gesture / Digit Classification Training Script
#
# This script:
# 1. Loads IMU time-series CSV data (accelerometer + gyroscope)
# 2. Resamples all sequences to a fixed length
# 3. Performs a clean train/test split per class
# 4. Normalizes data using statistics from the training set only
# 5. Trains a 1D CNN for multi-class classification
# 6. Evaluates on a held-out test set
# 7. Saves the trained FP32 model and confusion matrix visualizations
# ------------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================

# Root directory containing class-wise CSV folders
DATA_DIR = "single_digit_merged/"

# Number of output classes
NUM_CLASSES = 11

# Per-class dataset split
TRAIN_SAMPLES_PER_CLASS = 80
TEST_SAMPLES_PER_CLASS = 20

# IMU channels used as input features
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
NUM_CHANNELS = len(CHANNELS)

# Fixed temporal length for all samples
FIXED_TIMESTEPS = 140

# Reproducibility
RANDOM_SEED = 42

# Output model path
FP32_MODEL_PATH = "imu_14class_model.keras"

# Set random seeds
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================
# CSV LOADING & RESAMPLING
# =========================

def load_csv(path):
    """
    Load a CSV file containing IMU data and resample it
    to a fixed number of timesteps.

    Args:
        path (str): Path to CSV file

    Returns:
        np.ndarray: Shape (FIXED_TIMESTEPS, NUM_CHANNELS)
    """
    df = pd.read_csv(path)[CHANNELS]
    data = df.values.astype(np.float32)

    # Resample if sequence length does not match target length
    if len(data) != FIXED_TIMESTEPS:
        data = resample(data, FIXED_TIMESTEPS)

    return data

# =========================
# DATASET LOADING (CLEAN SPLIT)
# =========================

def load_dataset():
    """
    Load dataset with a strict per-class train/test split.
    Ensures no sample leakage between sets.

    Returns:
        X_train, y_train, X_test, y_test
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(class_id))
        files = sorted(f for f in os.listdir(class_dir) if f.endswith(".csv"))
        random.shuffle(files)

        # Split files per class
        train_files = files[:TRAIN_SAMPLES_PER_CLASS]
        test_files  = files[
            TRAIN_SAMPLES_PER_CLASS:
            TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS
        ]

        # Load training samples
        for f in train_files:
            X_train.append(load_csv(os.path.join(class_dir, f)))
            y_train.append(class_id)

        # Load testing samples
        for f in test_files:
            X_test.append(load_csv(os.path.join(class_dir, f)))
            y_test.append(class_id)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test),  np.array(y_test)
    )

print("Loading dataset...")
X_train, y_train, X_test, y_test = load_dataset()
print("Train:", X_train.shape, "Test:", X_test.shape)

# =========================
# NORMALIZATION (TRAIN SET ONLY)
# =========================

# Standardize features using training statistics only
scaler = StandardScaler()

# Flatten time dimension for scaling
X_train_2d = X_train.reshape(-1, NUM_CHANNELS)
X_test_2d  = X_test.reshape(-1, NUM_CHANNELS)

# Fit scaler on training data
scaler.fit(X_train_2d)

# Apply normalization
X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
X_test  = scaler.transform(X_test_2d).reshape(X_test.shape)

# Save scaler parameters for deployment
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

# =========================
# LABEL ENCODING
# =========================

# Convert class labels to one-hot vectors
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat  = keras.utils.to_categorical(y_test, NUM_CLASSES)

# =========================
# MODEL DEFINITION (1D CNN)
# =========================

model = keras.Sequential([
    layers.Input(shape=(FIXED_TIMESTEPS, NUM_CHANNELS)),

    layers.Conv1D(32, 5, activation="relu"),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, 3, activation="relu"),
    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAINING
# =========================

model.fit(
    X_train, y_train_cat,
    epochs=40,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# =========================
# EVALUATION (HELD-OUT TEST SET)
# =========================

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")

# Predict class labels
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# =========================
# CONFUSION MATRIX VISUALIZATION
# =========================

def save_cm(cm, normalize, path, title):
    """
    Save confusion matrix as an image.

    Args:
        cm (np.ndarray): Confusion matrix
        normalize (bool): Normalize rows if True
        path (str): Output image path
        title (str): Plot title
    """
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else int(val)
            plt.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if val > cm.max() / 2 else "black"
            )

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# Save confusion matrix images
save_cm(cm, False, "confusion_matrix.png", "Confusion Matrix (Counts)")
save_cm(cm, True,  "confusion_matrix_normalized.png", "Confusion Matrix (Normalized)")

# =========================
# SAVE TRAINED MODEL
# =========================

model.save(FP32_MODEL_PATH)
print(f"\nTrained model saved to {FP32_MODEL_PATH}")