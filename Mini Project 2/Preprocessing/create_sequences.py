import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Load preprocessed data
df = pd.read_csv("Data/processed_dataset.csv")

# Constants
SAMPLE_RATE = 50  # 50 Hz
WINDOW_SIZE = 3 * SAMPLE_RATE  # 3 seconds

# Encode labels
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
np.save("Data/label_classes.npy", le.classes_)

# Convert to sequences
sequences = []
labels = []

for i in range(0, len(df) - WINDOW_SIZE, WINDOW_SIZE):
    window = df.iloc[i:i+WINDOW_SIZE]
    label = window['Label'].mode()[0]
    features = window.drop(columns=['Label']).values
    sequences.append(features)
    labels.append(label)

X = np.array(sequences)
y = np.array(labels)

np.save("Data/X.npy", X)
np.save("Data/y.npy", y)

print(f"Saved sequences: X.shape={X.shape}, y.shape={y.shape}")
