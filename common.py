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
# TODO: Split into windows of 5-10s?
# TODO: Play with these preprocessing params once the pipeline is established
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

# TODO: Add extracting of MFCC

if __name__ == "__main__":
    preprocess("dataset/300_AUDIO.wav")