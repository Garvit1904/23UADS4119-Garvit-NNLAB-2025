# bonus_rash_detection.py

import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the pre-trained LSTM road condition classifier model
model = load_model("models/lstm_classifier.h5")

# Load the preprocessed data
X = np.load("Data/X.npy")
y = np.load("Data/y.npy")

# Define thresholds for rash driving detection
ACCEL_THRESHOLD = 2.5  # Threshold for accelerometer (in m/s^2)
GYRO_THRESHOLD = 2.0  # Threshold for gyroscope (in rad/s)

# Function to detect rash driving based on accelerometer and gyroscope readings
def detect_rash_driving(X):
    rash_driving_alerts = []

    for i in range(X.shape[0]):
        acc_x, acc_y, acc_z = X[i, :, 0], X[i, :, 1], X[i, :, 2]  # Accelerometer data
        gyro_x, gyro_y, gyro_z = X[i, :, 3], X[i, :, 4], X[i, :, 5]  # Gyroscope data

        # Check if any of the accelerometer values exceed the threshold
        if np.any(np.abs(acc_x) > ACCEL_THRESHOLD) or np.any(np.abs(acc_y) > ACCEL_THRESHOLD) or np.any(np.abs(acc_z) > ACCEL_THRESHOLD):
            rash_driving_alerts.append(True)
        # Check if any of the gyroscope values exceed the threshold
        elif np.any(np.abs(gyro_x) > GYRO_THRESHOLD) or np.any(np.abs(gyro_y) > GYRO_THRESHOLD) or np.any(np.abs(gyro_z) > GYRO_THRESHOLD):
            rash_driving_alerts.append(True)
        else:
            rash_driving_alerts.append(False)

    return rash_driving_alerts

# Function to classify road conditions and detect rash driving
def classify_and_alert_rash_driving(X):
    rash_driving_alerts = detect_rash_driving(X)
    road_conditions = model.predict(X)  # Predict the road conditions

    for i in range(len(rash_driving_alerts)):
        road_condition = np.argmax(road_conditions[i])  # Get predicted road condition
        if rash_driving_alerts[i]:
            print(f"Rash driving detected on sample {i} (Road Condition: {road_condition})!")
        else:
            print(f"Safe driving detected on sample {i} (Road Condition: {road_condition})")

# Detect rash driving on the dataset
classify_and_alert_rash_driving(X)

