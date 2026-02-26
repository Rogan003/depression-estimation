import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from common import get_mfcc_windows

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y = []
    
    for index, row in df.iterrows():
        participant_id = int(row['Participant_ID'])
        score = row['PHQ8_Score']
        file_path = f"dataset/{participant_id}_AUDIO.wav"
        
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            windows = get_mfcc_windows(file_path, 13, 42, 18)
            for window in windows:
                X.append(window.flatten())
                y.append(score)
        else:
            print(f"Warning: {file_path} not found.")
            
    return np.array(X), np.array(y)

def main():
    print("Loading training data...")
    X_train, y_train = load_data("dataset/train.csv")
    
    print("Loading validation data...")
    X_val, y_val = load_data("dataset/val.csv")
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Training SVR model with {len(X_train)} samples...")
    model = SVR(C=1, epsilon=0.01, gamma='scale', kernel='rbf')
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_val_scaled)
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    pearson_corr, _ = pearsonr(y_val, y_pred)
    
    print(f"SVR Evaluation Results (Continuous):")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")

if __name__ == "__main__":
    main()
