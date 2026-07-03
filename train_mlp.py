"""
MLP 모델 학습 스크립트
에너지 빈곤 위험도(0~1) 예측
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dummy_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "mlp_scaler.pkl")


def load_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = df[["income", "members", "energy_cost", "energy_ratio"]].values.astype(np.float32)
    labels = df["is_poverty"].values.astype(np.float32)
    return features, labels


def build_mlp(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="energy_poverty_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"데이터 파일이 없습니다. 먼저 generate_data.py를 실행하세요: {DATA_PATH}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = load_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = build_mlp(X.shape[1])
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
    )

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"MLP 모델 저장 완료: {MODEL_PATH}")
    print(f"스케일러 저장 완료: {SCALER_PATH}")


if __name__ == "__main__":
    main()
