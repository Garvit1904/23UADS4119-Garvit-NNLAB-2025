import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Paths to the dataset folders
DATA_DIR = 'Data'
CATEGORIES = ['Bitumin', 'Block', 'Concrete', 'Kanker']

# Columns
SENSOR_COLUMNS = ['X_Acc', 'Y_Acc', 'Z_Acc', 'X_Gyro', 'Y_Gyro', 'Z_Gyro']

def load_and_label_data():
    dataframes = []

    for category in CATEGORIES:
        folder_path = os.path.join(DATA_DIR, category)
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)

                # Keep only required columns
                df = df[SENSOR_COLUMNS].copy()

                # Add the label column
                df['Label'] = category
                dataframes.append(df)

    # Combine all into one DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def preprocess_data(df):
    features = df[SENSOR_COLUMNS]
    labels = df['Label']

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Combine back into a DataFrame
    processed_df = pd.DataFrame(scaled_features, columns=SENSOR_COLUMNS)
    processed_df['Label'] = labels

    return processed_df

if __name__ == "__main__":
    print("Loading and labeling data...")
    raw_df = load_and_label_data()
    
    print("Raw data shape:", raw_df.shape)

    print("Preprocessing and scaling data...")
    processed_df = preprocess_data(raw_df)

    output_path = os.path.join(DATA_DIR, 'processed_dataset.csv')
    processed_df.to_csv(output_path, index=False)

    print(f"Processed data saved to: {output_path}")
    print("Done!")