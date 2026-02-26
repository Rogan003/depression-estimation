# Depression Estimation Using Audio with Machine Learning Models

## Project Overview
This project explores the automatic estimation of depression levels based on **acoustic characteristics of speech**, without analyzing the content of the spoken words. It uses machine learning and deep learning techniques to predict **PHQ-8 scores** (0-24) from audio recordings of clinical interviews.

The project was developed by **Veselin Roganović (SV 36/2022)** as part of the Soft Computing course at the **Faculty of Technical Sciences, Novi Sad**.

## Dataset
- **Name:** [DAICWOZ](https://www.kaggle.com/datasets/saifzaman123445/daicwoz)
- **Content:** Audio recordings of clinical interviews.
- **Targets:** PHQ-8 depression scores.
- **Sample Size:** 140 participants.
- **Data Split:** 80% Training, 10% Validation, 10% Test.

## Preprocessing
- **Resampling:** All audio is resampled to 16kHz.
- **Normalization:** Amplitude normalization is applied.
- **Feature Extraction:** MFCC (Mel-frequency cepstral coefficients) are extracted using a sliding window approach to segment long interviews and augment the data.
  - Window size: ~10-42s (depending on model configuration)
  - Hop length: ~5-18s

## Methodology
Two primary approaches were implemented and compared:

### 1. SVR (Support Vector Regression)
- **Input:** Flattened MFCC matrices from time windows.
- **Model:** SVR with an **RBF kernel**.
- **Hyperparameters:** C=1, epsilon=0.01, gamma='scale'.
- **Scaler:** `StandardScaler` from Scikit-Learn.

### 2. CNN + LSTM
- **Architecture:** 
  - **CNN:** 2D convolutional layers to extract spatial features from MFCC representations, including `BatchNormalization` and `Dropout`.
  - **LSTM:** 2 layers with 64 hidden units to capture temporal dependencies across sequences.
  - **Output:** A fully connected layer for final regression.
- **Loss Function:** A combined loss of **MAE** and **Pearson Correlation Loss** to optimize both absolute error and trend prediction.

## Results
Evaluation metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Pearson Correlation coefficient.

### Validation Results
| Model | MAE | RMSE | Pearson |
| :--- | :---: | :---: | :---: |
| SVR | 4.29 | **4.98** | **0.52** |
| CNN+LSTM | **3.99** | 5.02 | 0.45 |

### Test Results
| Model | MAE | RMSE | Pearson |
| :--- | :---: | :---: | :---: |
| SVR | **4.37** | **6.17** | **0.27** |
| CNN+LSTM | 5.51 | 7.20 | 0.05 |

**Conclusion:** SVR outperformed the deep learning approach on the test set, suggesting it is more robust for small-scale clinical datasets where deep models are prone to overfitting.

## Technologies Used
- **Python**
- **PyTorch** (Deep Learning)
- **Scikit-Learn** (SVR & Preprocessing)
- **Librosa** (Audio Processing)
- **Pandas/NumPy** (Data Management)

## How to Run
1. **Prepare Environment:** Install the required dependencies from `requirements.txt`.
2. **Fetch Dataset:** Run `python dataset_fetch.py` to obtain and prepare the data. Data will take about 6GB of storage. Also, you may need to authenticate with your Kaggle account first.
3. **Run Models:**
   - For the CNN+LSTM approach: `python cnn_lstm_approach.py`
   - For the SVR approach: `python svr_approach.py`
