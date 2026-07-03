"""
더미 데이터 생성 스크립트
에너지 빈곤 가구 1000개, 12개월 시계열 에너지 비용 포함
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_HOUSEHOLDS = 1000
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "dummy_data.csv")


def generate_household_data(n: int = NUM_HOUSEHOLDS) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)
    records = []

    for hid in range(1, n + 1):
        income = np.random.uniform(0, 5_000_000)
        members = np.random.randint(1, 7)
        energy_cost = np.random.uniform(50_000, 300_000)

        ratio = energy_cost / income if income > 0 else 1.0
        is_poverty = 1 if ratio > 0.10 else 0

        base = energy_cost
        seasonal = np.array([1.15, 1.20, 1.05, 0.90, 0.85, 0.90, 1.00, 0.95, 0.90, 0.95, 1.05, 1.25])
        noise = np.random.normal(0, 0.05, 12)
        history = base * seasonal * (1 + noise)
        history = np.clip(history, 50_000, 350_000)

        future_noise = np.random.normal(0, 0.04, 6)
        future = history[-1] * (1 + np.linspace(0.01, 0.06, 6)) * (1 + future_noise)
        future = np.clip(future, 50_000, 350_000)

        row = {
            "household_id": hid,
            "income": round(income, 0),
            "members": members,
            "energy_cost": round(energy_cost, 0),
            "energy_ratio": round(ratio, 4),
            "is_poverty": is_poverty,
            "region": np.random.choice(
                ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
            ),
        }
        for i, val in enumerate(history, 1):
            row[f"month_{i}"] = round(val, 0)
        for i, val in enumerate(future, 13):
            row[f"month_{i}"] = round(val, 0)

        records.append(row)

    return pd.DataFrame(records)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_household_data()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    poverty_rate = df["is_poverty"].mean() * 100
    print(f"데이터 생성 완료: {OUTPUT_PATH}")
    print(f"가구 수: {len(df)}, 에너지 빈곤 가구 비율: {poverty_rate:.1f}%")


if __name__ == "__main__":
    main()
