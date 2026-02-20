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

# TODO: Remove interviewer (she is less loud or just use the transcript files)?
# TODO: Play with these preprocessing params once the pipeline is established. Add some more preprocessing stuff?
def preprocess(file_path):
    # 1. Load + resample + mono
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    print("Original length:", len(audio)/sr, "seconds")

    # 2. Remove silence
    audio_trimmed, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    print("After trimming:", len(audio_trimmed)/sr, "seconds")

    # 3. Normalize
    audio_normalized = audio_trimmed / np.max(np.abs(audio_trimmed))

    return audio_normalized

def extract_mfcc(audio, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

def extract_pitch(audio, sr):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch.append(pitches[index, i])
    return np.array(pitch)

def extract_energy(audio):
    return librosa.feature.rms(y=audio)[0]

def get_summary_features(file_path):
    audio = preprocess(file_path)
    sr = 16000 # as used in preprocess
    
    mfcc = extract_mfcc(audio, sr)
    energy = extract_energy(audio)
    
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(energy),
        np.std(energy)
    ])
    return features

def get_mfcc_windows(file_path, n_mfcc=13, window_size_s=5, hop_length_s=2.5):
    audio = preprocess(file_path)
    sr = 16000
    
    # Calculate MFCC
    mfcc = extract_mfcc(audio, sr, n_mfcc=n_mfcc)
    
    # MFCC matrix shape: (n_mfcc, n_frames)
    # Convert window sizes from seconds to frames
    hop_length = 512 # default for librosa.feature.mfcc
    frames_per_sec = sr / hop_length
    window_frames = int(window_size_s * frames_per_sec)
    hop_frames = int(hop_length_s * frames_per_sec)
    
    windows = []
    for start in range(0, mfcc.shape[1] - window_frames + 1, hop_frames):
        window = mfcc[:, start : start + window_frames]
        windows.append(window)
        
    return np.array(windows) # Shape: (n_windows, n_mfcc, window_frames)