"""
LSTM 모델 학습 스크립트
최근 12개월 에너지 비용 → 향후 6개월 예측
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dummy_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
LOOKBACK = 12
FORECAST = 6


def load_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    history_cols = [f"month_{i}" for i in range(1, LOOKBACK + 1)]
    future_cols = [f"month_{i}" for i in range(LOOKBACK + 1, LOOKBACK + FORECAST + 1)]

    X = df[history_cols].values.astype(np.float32)
    y = df[future_cols].values.astype(np.float32)
    return X, y


def build_lstm() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(LOOKBACK, 1)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(FORECAST),
        ],
        name="energy_cost_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model


def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"데이터 파일이 없습니다. 먼저 generate_data.py를 실행하세요: {DATA_PATH}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = load_sequences(df)

    scaler = MinMaxScaler()
    all_values = np.concatenate([X.flatten(), y.flatten()]).reshape(-1, 1)
    scaler.fit(all_values)

    X_scaled = scaler.transform(X.reshape(-1, 1)).reshape(X.shape[0], LOOKBACK, 1)
    y_scaled = scaler.transform(y.reshape(-1, 1)).reshape(y.shape[0], FORECAST)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=RANDOM_SEED
    )

    model = build_lstm()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        verbose=1,
    )

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"LSTM 모델 저장 완료: {MODEL_PATH}")
    print(f"스케일러 저장 완료: {SCALER_PATH}")


if __name__ == "__main__":
    main()
