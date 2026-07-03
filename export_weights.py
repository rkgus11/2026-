"""
학습된 .h5 모델과 스케일러의 가중치를 numpy(.npz)로 추출하는 스크립트.
웹 앱(app.py)이 TensorFlow 없이 추론할 수 있도록 가중치만 저장한다.
학습 파이프라인(train_mlp.py / train_lstm.py)은 그대로 TensorFlow/Keras를 사용한다.
"""

import os

import joblib
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MLP_PATH = os.path.join(MODEL_DIR, "mlp_model.h5")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
MLP_SCALER_PATH = os.path.join(MODEL_DIR, "mlp_scaler.pkl")
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
OUTPUT_PATH = os.path.join(MODEL_DIR, "weights.npz")


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras

    mlp = keras.models.load_model(MLP_PATH, compile=False)
    lstm = keras.models.load_model(LSTM_PATH, compile=False)
    mlp_scaler = joblib.load(MLP_SCALER_PATH)
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)

    arrays = {}

    mlp_dense = [layer for layer in mlp.layers if len(layer.get_weights()) == 2]
    for idx, layer in enumerate(mlp_dense):
        w, b = layer.get_weights()
        arrays[f"mlp_w{idx}"] = w.astype(np.float32)
        arrays[f"mlp_b{idx}"] = b.astype(np.float32)
    arrays["mlp_mean"] = mlp_scaler.mean_.astype(np.float32)
    arrays["mlp_std"] = mlp_scaler.scale_.astype(np.float32)

    lstm_layers = [layer for layer in lstm.layers if len(layer.get_weights()) == 3]
    for idx, layer in enumerate(lstm_layers):
        kernel, recurrent, bias = layer.get_weights()
        arrays[f"lstm{idx}_kernel"] = kernel.astype(np.float32)
        arrays[f"lstm{idx}_recurrent"] = recurrent.astype(np.float32)
        arrays[f"lstm{idx}_bias"] = bias.astype(np.float32)

    dense_out = [layer for layer in lstm.layers if len(layer.get_weights()) == 2][-1]
    dw, db = dense_out.get_weights()
    arrays["lstm_dense_w"] = dw.astype(np.float32)
    arrays["lstm_dense_b"] = db.astype(np.float32)

    arrays["lstm_scale"] = np.asarray(lstm_scaler.scale_, dtype=np.float32)
    arrays["lstm_min"] = np.asarray(lstm_scaler.min_, dtype=np.float32)

    np.savez(OUTPUT_PATH, **arrays)
    print(f"가중치 저장 완료: {OUTPUT_PATH}")
    print(f"저장된 배열: {list(arrays.keys())}")

    _verify(mlp, lstm, mlp_scaler, lstm_scaler)


def _verify(mlp, lstm, mlp_scaler, lstm_scaler):
    """Keras 결과와 numpy 결과가 일치하는지 검증."""
    import infer

    weights = infer.load_weights(OUTPUT_PATH)

    income, members, energy_cost = 2_000_000, 3, 150_000
    ratio = energy_cost / income
    features = np.array([[income, members, energy_cost, ratio]], dtype=np.float32)
    keras_risk = float(np.asarray(mlp(mlp_scaler.transform(features), training=False))[0][0])
    numpy_risk = infer.predict_risk(weights, income, members, energy_cost)
    print(f"[MLP] keras={keras_risk:.6f}  numpy={numpy_risk:.6f}  diff={abs(keras_risk - numpy_risk):.2e}")

    history = np.linspace(120_000, 180_000, 12).astype(np.float32)
    scaled = lstm_scaler.transform(history.reshape(-1, 1)).reshape(1, 12, 1).astype(np.float32)
    keras_pred = lstm_scaler.inverse_transform(
        np.asarray(lstm(scaled, training=False))[0].reshape(-1, 1)
    ).flatten()
    numpy_pred = infer.predict_costs(weights, history)
    max_diff = float(np.max(np.abs(keras_pred - numpy_pred)))
    print(f"[LSTM] keras={keras_pred.round(0)}")
    print(f"[LSTM] numpy={numpy_pred.round(0)}")
    print(f"[LSTM] max diff={max_diff:.4f}")

    assert abs(keras_risk - numpy_risk) < 1e-4, "MLP 결과 불일치"
    assert max_diff < 1.0, "LSTM 결과 불일치"
    print("검증 통과: numpy 추론이 keras 결과와 일치합니다.")


if __name__ == "__main__":
    main()
