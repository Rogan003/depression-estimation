import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from common import get_mfcc_windows
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

class AudioDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.data = []
        
        print(f"Loading data from {csv_path}...")
        for index, row in self.df.iterrows():
            participant_id = int(row['Participant_ID'])
            score = row['PHQ8_Score']
            file_path = f"dataset/{participant_id}_AUDIO.wav"
            
            if os.path.exists(file_path):
                print(f"Processing {file_path}...")
                windows = get_mfcc_windows(file_path)
                if len(windows) > 0:
                    for window in windows:
                        self.data.append((window, score))
            else:
                print(f"Warning: {file_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, score = self.data[idx]
        return torch.FloatTensor(window).unsqueeze(0), torch.FloatTensor([score])

class CNNLSTM(nn.Module):
    def __init__(self, n_mfcc=13, window_frames=156):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # n_mfcc=13, window_frames=156
        # After MaxPool 1: 13//2=6, 156//2=78
        # After MaxPool 2: 6//2=3, 78//2=39
        self.feature_dim = 32 * 3 * 39 
        
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(b, 1, -1)
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])

def main():
    # Load data
    train_dataset = AudioDataset("dataset/train.csv")
    val_dataset = AudioDataset("dataset/val.csv")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = CNNLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    print("Evaluating...")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            all_preds.extend(outputs.numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    # Pearson might fail if all predictions are the same
    if len(np.unique(all_preds)) > 1:
        pearson_corr, _ = pearsonr(all_targets, all_preds)
    else:
        pearson_corr = 0.0
        
    print(f"CNN+LSTM Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")

if __name__ == "__main__":
    main()
