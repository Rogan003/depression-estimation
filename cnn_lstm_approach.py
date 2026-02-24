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
            score = float(row['PHQ8_Score'])
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
        window, y = self.data[idx]
        return torch.FloatTensor(window).unsqueeze(0), torch.FloatTensor([y])

class CNNLSTM(nn.Module):
    def __init__(self, n_mfcc=13, window_frames=156):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        
        # After MaxPool 1: (n_mfcc//2, window_frames//2) = (6, 78)
        # After MaxPool 2: (6//2, 78//2) = (3, 39)
        # Current cnn_out shape: (batch, 32, 3, 39)
        # We want to treat one dimension as time. Let's use the window_frames dimension (width).
        # We'll pool over the height (n_mfcc) and keep the width as time sequence.
        
        self.lstm_input_size = 32 * 3 # 32 channels * 3 frequency bins
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=64, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        cnn_out = self.cnn(x) # (b, 32, 3, 39)
        
        # Reshape to (b, width, channels * height) -> (b, 39, 32 * 3)
        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous()
        cnn_out = cnn_out.view(b, cnn_out.size(1), -1)
        
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_dataset = AudioDataset("dataset/train.csv")
    val_dataset = AudioDataset("dataset/val.csv")

    # TODO: batch size improve? use everything since I have a small dataset?
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # TODO: Preprocess val before prediction? Or is it already done?
    
    model = CNNLSTM().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print("Starting training...")
    epochs = 20
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
    print("Evaluating...")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = outputs.cpu().numpy().flatten()
            targets = batch_y.numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)
            
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
