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