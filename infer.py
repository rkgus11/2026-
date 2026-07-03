"""
TensorFlow 없이 numpy만으로 MLP / LSTM 추론을 수행하는 모듈.
학습된 가중치(models/weights.npz)를 불러와 순전파를 직접 계산한다.
"""

import numpy as np

LOOKBACK = 12
FORECAST = 6


def load_weights(path: str) -> dict:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def predict_risk(weights: dict, income: float, members: int, energy_cost: float) -> float:
    ratio = energy_cost / income if income > 0 else 1.0
    features = np.array([income, members, energy_cost, ratio], dtype=np.float32)

    x = (features - weights["mlp_mean"]) / weights["mlp_std"]
    x = _relu(x @ weights["mlp_w0"] + weights["mlp_b0"])
    x = _relu(x @ weights["mlp_w1"] + weights["mlp_b1"])
    out = _sigmoid(x @ weights["mlp_w2"] + weights["mlp_b2"])
    return float(out[0])


def _lstm_layer(sequence: np.ndarray, kernel: np.ndarray, recurrent: np.ndarray, bias: np.ndarray):
    units = kernel.shape[1] // 4
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    outputs = []

    for t in range(sequence.shape[0]):
        z = sequence[t] @ kernel + h @ recurrent + bias
        i = _sigmoid(z[:units])
        f = _sigmoid(z[units : 2 * units])
        c_bar = np.tanh(z[2 * units : 3 * units])
        o = _sigmoid(z[3 * units : 4 * units])
        c = f * c + i * c_bar
        h = o * np.tanh(c)
        outputs.append(h.copy())

    return np.array(outputs, dtype=np.float32)


def predict_costs(weights: dict, history: np.ndarray) -> np.ndarray:
    scale = weights["lstm_scale"]
    min_ = weights["lstm_min"]

    hist = history.reshape(-1, 1).astype(np.float32)
    scaled = hist * scale + min_

    seq = _lstm_layer(scaled, weights["lstm0_kernel"], weights["lstm0_recurrent"], weights["lstm0_bias"])
    seq = _lstm_layer(seq, weights["lstm1_kernel"], weights["lstm1_recurrent"], weights["lstm1_bias"])

    last = seq[-1]
    pred_scaled = last @ weights["lstm_dense_w"] + weights["lstm_dense_b"]

    pred = (pred_scaled - min_) / scale
    return pred.flatten()
