import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def merge_dataset_csv():
    train = pd.read_csv("dataset/train_split_Depression_AVEC2017.csv")
    test = pd.read_csv("dataset/dev_split_Depression_AVEC2017.csv")

    merged = pd.concat([train, test], ignore_index=True)
    merged = merged.sort_values(by="Participant_ID").reset_index(drop=True)

    merged.to_csv("dataset/merged_data.csv", index=False)

    print(merged["Participant_ID"].is_unique)

def split_dataset_80_10_10(df, seed=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        shuffle=True
    )

    train_df.to_csv("dataset/train.csv", index=False)
    val_df.to_csv("dataset/val.csv", index=False)
    test_df.to_csv("dataset/test.csv", index=False)

def get_summary_features(file_path):
    sr = 16000
    audio = preprocess(file_path, sr)
    hop_length = 16384
    
    mfcc = extract_mfcc(audio, sr, hop_length=hop_length)
    energy = extract_energy(audio)
    pitch = extract_pitch(audio, sr, hop_length=hop_length)
    
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.percentile(mfcc, 25, axis=1),
        np.percentile(mfcc, 50, axis=1),
        np.percentile(mfcc, 75, axis=1),
        # np.mean(energy), even better results
        # np.std(energy),
        # np.mean(pitch),
        # np.std(pitch)
    ])
    return features

def get_mfcc_windows(file_path, n_mfcc=13, window_size_s=10, hop_length_s=5):
    sr = 16000
    audio = preprocess(file_path, sr)
    hop_length = 16384
    
    # Calculate MFCC
    mfcc = extract_mfcc(audio, sr, n_mfcc=n_mfcc, hop_length=hop_length)
    
    # MFCC matrix shape: (n_mfcc, n_frames)
    # Convert window sizes from seconds to frames
    frames_per_sec = sr / hop_length
    window_frames = int(window_size_s * frames_per_sec)
    hop_frames = int(hop_length_s * frames_per_sec)
    
    windows = []
    for start in range(0, mfcc.shape[1] - window_frames + 1, hop_frames):
        window = mfcc[:, start : start + window_frames]
        windows.append(window)
        
    return np.array(windows) # Shape: (n_windows, n_mfcc, window_frames)

def preprocess(file_path, sr):
    # 1. Load + resample + mono
    audio, sr = librosa.load(file_path, sr=sr, mono=True)

    print("Original length:", len(audio)/sr, "seconds")

    # 2. Remove interviewer
    audio_without_interviewer = audio#remove_interviewer_from_audio(audio, file_path[8:11], sr)

    print("After trimming:", len(audio_without_interviewer)/sr, "seconds")

    # 3. Normalize
    audio_normalized = audio_without_interviewer / np.max(np.abs(audio_without_interviewer))

    return audio_normalized


def remove_interviewer_from_audio(audio, file_id, sr):
    transcript = pd.read_csv(f"dataset/{file_id}_TRANSCRIPT.csv", sep="\t")

    participant_segments = transcript[transcript["speaker"] == "Participant"]

    audio_segments = []
    for _, row in participant_segments.iterrows():
        start_sample = int(row["start_time"] * sr)
        stop_sample = int(row["stop_time"] * sr)
        audio_segments.append(audio[start_sample:stop_sample])

    if len(audio_segments) > 0:
        audio = np.concatenate(audio_segments)
    else:
        audio = np.array([])

    return audio

def extract_mfcc(audio, sr, n_mfcc=13, hop_length=512):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

def extract_pitch(audio, sr, hop_length=512):
    # F0 estimation using YIN
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length
    )
    # Fill NaNs in f0 with 0 (unvoiced)
    f0[np.isnan(f0)] = 0
    return f0

def extract_energy(audio):
    return librosa.feature.rms(y=audio)[0]