# from concurrent.futures import ProcessPoolExecutor, as_completed

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
    def __init__(self, csv_path, window_size=10, hop_length=5):
        self.df = pd.read_csv(csv_path)
        self.data = []
        
        print(f"Loading data from {csv_path}...")
        for index, row in self.df.iterrows():
            participant_id = int(row['Participant_ID'])
            score = float(row['PHQ8_Score'])
            file_path = f"dataset/{participant_id}_AUDIO.wav"
            
            if os.path.exists(file_path):
                print(f"Processing {file_path}...")
                windows = get_mfcc_windows(file_path, window_size_s=window_size, hop_length_s=hop_length)
                if len(windows) > 0:
                    for window in windows:
                        self.data.append((window, score)) # TODO: Think about these windows? Do they even make sense?
            else:
                print(f"Warning: {file_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, y = self.data[idx]
        return torch.FloatTensor(window).unsqueeze(0), torch.FloatTensor([y])

class CNNLSTM(nn.Module):
    def __init__(self):
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
        
        self.lstm_input_size = 32 * 3
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=64, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        cnn_out = self.cnn(x)

        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous()
        cnn_out = cnn_out.view(b, cnn_out.size(1), -1)
        
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])

def combined_loss(y_pred, y_true, alpha=0.5):
    # MAE part
    mae = torch.mean(torch.abs(y_pred - y_true))

    # Pearson part
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)

    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    numerator = torch.sum(y_true_centered * y_pred_centered)
    denominator = torch.sqrt(torch.sum(y_true_centered ** 2)) * torch.sqrt(torch.sum(y_pred_centered ** 2))

    pearson = numerator / (denominator + 1e-8)

    loss = alpha * mae + (1 - alpha) * (1 - pearson)
    return loss

def main(window_size=10, hop_length=5):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu") # using Mac GPU got me worse results

    train_dataset = AudioDataset("dataset/train.csv", window_size, hop_length)
    val_dataset = AudioDataset("dataset/val.csv")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = CNNLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print("Starting training...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = combined_loss(outputs, batch_y, alpha=0.7)
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

    return mae, rmse, pearson_corr

if __name__ == "__main__":
    mae, rmse, pearson_corr = main(10, 5)

    print(f"CNN+LSTM Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")
    # window_sizes = range(3, 61, 3)
    # results = []
    #
    # combinations = []
    # for w in window_sizes:
    #     for h in range(2, w + 1, 2):
    #         combinations.append((w, h))
    #
    # # Use all available CPU cores
    # num_workers = None  # None = use os.cpu_count()
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     # Submit all jobs
    #     futures = {executor.submit(main, w, h): (w, h) for w, h in combinations}
    #
    #     for future in as_completed(futures):
    #         w, h = futures[future]
    #         try:
    #             mae, rmse, pearson_corr = future.result()
    #             results.append((mae, rmse, pearson_corr, w, h))
    #         except Exception as e:
    #             print(f"Error for window={w}, hop={h}: {e}")
    #
    # # Sort by MAE, then RMSE, then descending Pearson
    # results.sort(key=lambda x: (x[0], x[1], -x[2]))
    #
    # # Print top 10 results
    # print("Top 10 results:")
    # for i, (mae, rmse, pearson, w, h) in enumerate(results[:10], 1):
    #     print(f"#{i} window_size={w}s, hop_length={h}s")
    #     print(f"   CNN+LSTM Evaluation Results:")
    #     print(f"   MAE: {mae:.4f}")
    #     print(f"   RMSE: {rmse:.4f}")
    #     print(f"   Pearson correlation: {pearson:.4f}")
