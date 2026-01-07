# ------------------------------------------------------------
# realtime_inference.py
# ------------------------------------------------------------
# Real-Time IMU Digit Recognition (UDP)
#
# This script:
# 1. Receives streamed IMU data over UDP
# 2. Records a gesture sequence on user command
# 3. Resamples and normalizes the sequence
# 4. Runs inference using a trained CNN model
# 5. Displays predicted digits and builds a multi-digit number
#
# Intended to be used with an embedded IMU device
# (e.g., Nicla Vision) streaming data via UDP.
# ------------------------------------------------------------

import socket
import numpy as np
import tensorflow as tf
from scipy.signal import resample

# =========================
# CONFIGURATION
# =========================

# UDP server configuration (listen on all interfaces)
UDP_IP = "0.0.0.0"
UDP_PORT = 5010

# Model and scaler paths
MODEL_PATH = "imu_14class_model.keras"
SCALER_MEAN_PATH = "scaler_mean.npy"
SCALER_SCALE_PATH = "scaler_scale.npy"

# Model input parameters
FIXED_TIMESTEPS = 140
NUM_CHANNELS = 6

# Class labels (digits)
CLASS_NAMES = [str(i) for i in range(11)]

# =========================
# LOAD MODEL & NORMALIZER
# =========================

# Load trained Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Load normalization parameters from training
scaler_mean = np.load(SCALER_MEAN_PATH)
scaler_scale = np.load(SCALER_SCALE_PATH)

# =========================
# UDP SOCKET SETUP
# =========================

# Create UDP socket and bind to port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Non-blocking receive with short timeout
sock.settimeout(0.01)

# User instructions
print("\nENTER → record digit")
print("ENTER → stop & predict")
print("q + ENTER → finish\n")

# Store predicted digit sequence
digit_sequence = []

try:
    while True:
        cmd = input("Ready: ").strip().lower()
        if cmd == "q":
            break

        # ------------------------------------------------
        # START RECORDING IMU DATA
        # ------------------------------------------------
        print("Recording...")
        recorded_samples = []

        while True:
            try:
                # Receive UDP packet
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                # Stop recording on ENTER key
                if input("") == "":
                    break
                continue

            # Parse CSV data: timestamp + 6 IMU values
            parts = data.decode().strip().split(",")
            if len(parts) != 7:
                continue

            try:
                sample = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue

            recorded_samples.append(sample)

        # ------------------------------------------------
        # STOP RECORDING & RUN INFERENCE
        # ------------------------------------------------
        if len(recorded_samples) < 10:
            print("Too short, skipped.\n")
            continue

        # Convert to numpy array
        data = np.array(recorded_samples, dtype=np.float32)

        # Resample to fixed length
        data = resample(data, FIXED_TIMESTEPS)

        # Normalize using training statistics
        data = (data - scaler_mean) / scaler_scale

        # Add batch dimension
        window = np.expand_dims(data, axis=0)

        # Run model inference
        probs = model.predict(window, verbose=0)[0]

        # Get predicted digit
        digit = CLASS_NAMES[np.argmax(probs)]
        digit_sequence.append(digit)

        print(f"Digit: {digit}")
        print(f"Current number: {''.join(digit_sequence)}\n")

except KeyboardInterrupt:
    # Graceful exit on Ctrl+C
    pass

# Close socket
sock.close()

# =========================
# FINAL OUTPUT
# =========================

print("\nFINAL NUMBER:")
print("".join(digit_sequence) if digit_sequence else "(none)")
