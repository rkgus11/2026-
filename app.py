"""
데이터 기반 에너지 빈곤 가구 맞춤형 복지 정책 추천 및 효과 예측 웹 시스템
Streamlit 메인 앱
"""

import os
import socket

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import infer
from policies import (
    LOW_INCOME_INFO,
    NEWS_ITEMS,
    POVERTY_STATS,
    WELFARE_PROGRAMS,
    get_max_discount,
    recommend_policies,
)

matplotlib.use("Agg")

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.npz")

REGIONS = [
    "서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

SEASONAL = np.array([1.15, 1.20, 1.05, 0.90, 0.85, 0.90, 1.00, 0.95, 0.90, 0.95, 1.05, 1.25])
MONTH_LABELS = [f"{i}월" for i in range(1, 13)]
FUTURE_LABELS = [f"예측 {i}개월" for i in range(1, 7)]


@st.cache_data(ttl=60)
def check_internet() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


@st.cache_resource
def load_models():
    if not os.path.exists(WEIGHTS_PATH):
        return None
    return infer.load_weights(WEIGHTS_PATH)


def synthesize_history(monthly_cost: float) -> np.ndarray:
    normalized = SEASONAL / SEASONAL.mean()
    return monthly_cost * normalized


def predict_risk(weights, income: float, members: int, energy_cost: float) -> float:
    return infer.predict_risk(weights, income, members, energy_cost)


def predict_costs(weights, history: np.ndarray) -> np.ndarray:
    return infer.predict_costs(weights, history)


def apply_policy_discount(predictions: np.ndarray, discount: float) -> np.ndarray:
    factors = np.linspace(1.0, 1.0 - discount, len(predictions))
    return predictions * factors


def validate_inputs(income: float, energy_cost: float, address: str) -> list[str]:
    errors = []
    if income <= 0:
        errors.append("월 소득은 0보다 큰 값을 입력해 주세요.")
    if income > 5_000_000:
        errors.append("월 소득은 500만원 이하로 입력해 주세요.")
    if energy_cost <= 0:
        errors.append("월 에너지 비용은 0보다 큰 값을 입력해 주세요.")
    if energy_cost > 300_000:
        errors.append("월 에너지 비용은 30만원 이하로 입력해 주세요.")
    if not address.strip():
        errors.append("주소를 입력해 주세요.")
    return errors


def render_poverty_chart() -> str:
    chart_path = os.path.join(BASE_DIR, "data", "poverty_chart.png")
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)

    regions = list(POVERTY_STATS["by_region"].keys())
    rates = list(POVERTY_STATS["by_region"].values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["#e74c3c" if r > POVERTY_STATS["national_rate"] else "#3498db" for r in rates]
    axes[0].barh(regions, rates, color=colors)
    axes[0].axvline(POVERTY_STATS["national_rate"], color="black", linestyle="--", label=f"전국 평균 {POVERTY_STATS['national_rate']}%")
    axes[0].set_xlabel("에너지 빈곤율 (%)")
    axes[0].set_title("지역별 에너지 빈곤율")
    axes[0].legend()

    quintiles = list(POVERTY_STATS["by_income_quintile"].keys())
    q_rates = list(POVERTY_STATS["by_income_quintile"].values())
    axes[1].bar(quintiles, q_rates, color="#2ecc71")
    axes[1].set_ylabel("에너지 빈곤율 (%)")
    axes[1].set_title("소득 분위별 에너지 빈곤율")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def page_input():
    st.header("정보 입력")
    st.markdown("가구의 소득·주거·에너지 사용 정보를 입력하면 맞춤형 복지 정책을 추천해 드립니다.")

    col1, col2 = st.columns(2)

    with col1:
        income_slider = st.slider(
            "월 소득 범위 (만원)",
            min_value=0,
            max_value=500,
            value=200,
            step=10,
            help="가구의 월평균 소득 범위를 선택하세요.",
        )
        income = st.number_input(
            "월 소득 (원)",
            min_value=0,
            max_value=5_000_000,
            value=income_slider * 10_000,
            step=10_000,
            format="%d",
        )
        members = st.selectbox("주거 인원", options=list(range(1, 7)), index=2)

    with col2:
        energy_cost = st.number_input(
            "월평균 에너지 소비량 (전기세+연료비, 원)",
            min_value=0,
            max_value=300_000,
            value=150_000,
            step=5_000,
            format="%d",
        )
        region = st.selectbox("지역", options=REGIONS, index=0)
        address = st.text_input("주소", placeholder="예: 서울특별시 ○○구 ○○동")

    if income > 0:
        ratio = energy_cost / income * 100
        st.info(f"**월평균 소득 대비 에너지 지출:** {ratio:.1f}% {'(에너지 빈곤 기준 10% 초과)' if ratio > 10 else '(기준 이하)'}")

    if st.button("추천 결과 보기", type="primary", use_container_width=True):
        errors = validate_inputs(income, energy_cost, address)
        if errors:
            for err in errors:
                st.error(err)
            return

        st.session_state["input_data"] = {
            "income": income,
            "members": members,
            "energy_cost": energy_cost,
            "region": region,
            "address": address,
            "energy_ratio": energy_cost / income if income > 0 else 1.0,
        }
        st.session_state["page"] = "추천 결과"
        st.rerun()


def page_results():
    st.header("추천 결과")

    if "input_data" not in st.session_state:
        st.warning("먼저 가구 정보를 입력해 주세요.")
        if st.button("입력 페이지로 이동"):
            st.session_state["page"] = "정보 입력"
            st.rerun()
        return

    data = st.session_state["input_data"]
    with st.spinner("모델을 불러오는 중입니다..."):
        weights = load_models()

    if weights is None:
        st.error(
            "학습된 모델 가중치가 없습니다. 터미널에서 `python setup_all.py` 실행 후 "
            "`python export_weights.py`를 실행하여 가중치를 생성해 주세요."
        )
        return

    st.subheader("입력 정보 요약")
    summary = pd.DataFrame(
        {
            "항목": ["월 소득", "주거 인원", "월 에너지 비용", "소득 대비 지출", "지역", "주소"],
            "값": [
                f"{data['income']:,.0f}원",
                f"{data['members']}인",
                f"{data['energy_cost']:,.0f}원",
                f"{data['energy_ratio'] * 100:.1f}%",
                data["region"],
                data["address"],
            ],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    risk = predict_risk(weights, data["income"], data["members"], data["energy_cost"])
    policies = recommend_policies(risk)

    st.subheader("에너지 빈곤 위험도 (MLP 예측)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("빈곤 위험도", f"{risk:.2%}")
    with col2:
        level = "높음" if risk >= 0.7 else ("중간" if risk >= 0.5 else "낮음")
        st.metric("위험 수준", level)
    with col3:
        st.metric("소득 대비 에너지 지출", f"{data['energy_ratio'] * 100:.1f}%")

    st.subheader("추천 복지 정책")
    policy_df = pd.DataFrame(
        [
            {
                "정책명": p["name"],
                "분류": p["category"],
                "지원 내용": p["support"],
            }
            for p in policies
        ]
    )
    st.dataframe(policy_df, use_container_width=True, hide_index=True)

    for p in policies:
        with st.expander(f"📋 {p['name']} 상세 설명"):
            st.markdown(p["description"])
            st.markdown(f"**지원 내용:** {p['support']}")
            st.markdown(f"**참고:** [{p['link']}]({p['link']})")

    st.subheader("정책 적용 후 월별 에너지 비용 예측 (LSTM)")
    history = synthesize_history(data["energy_cost"])
    baseline = predict_costs(weights, history)
    discount = get_max_discount(policies)
    after_policy = apply_policy_discount(baseline, discount)

    chart_df = pd.DataFrame(
        {
            "월": MONTH_LABELS + FUTURE_LABELS,
            "실제/기준": list(history) + [None] * 6,
            "정책 미적용 예측": [None] * 12 + list(baseline),
            "정책 적용 예측": [None] * 12 + list(after_policy),
        }
    )
    st.line_chart(
        chart_df.set_index("월")[["실제/기준", "정책 미적용 예측", "정책 적용 예측"]],
        use_container_width=True,
    )

    comparison = pd.DataFrame(
        {
            "예측 월": FUTURE_LABELS,
            "정책 미적용 (원)": [f"{v:,.0f}" for v in baseline],
            "정책 적용 (원)": [f"{v:,.0f}" for v in after_policy],
            "절감액 (원)": [f"{b - a:,.0f}" for b, a in zip(baseline, after_policy)],
        }
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    total_save = baseline.sum() - after_policy.sum()
    st.success(f"6개월 예측 기준 총 절감 예상액: **{total_save:,.0f}원** (약 {discount:.0%} 할인 효과 반영)")

    if st.button("← 정보 입력으로 돌아가기"):
        st.session_state["page"] = "정보 입력"
        st.rerun()


def page_info():
    st.header("정보 소개")

    tab1, tab2, tab3, tab4 = st.tabs(["복지 사업", "에너지 뉴스", "빈곤율 통계", "저소득층 안내"])

    with tab1:
        st.markdown(WELFARE_PROGRAMS)

    with tab2:
        for news in NEWS_ITEMS:
            st.markdown(f"#### {news['title']}")
            st.markdown(f"📅 {news['date']} | 출처: {news['source']}")
            st.markdown(news["summary"])
            st.divider()

    with tab3:
        st.markdown(f"### 전국 에너지 빈곤율: **{POVERTY_STATS['national_rate']}%**")
        st.markdown(
            "소득 대비 에너지 지출이 10%를 초과하는 가구 비율입니다. "
            "출처: 에너지경제연구원, 지표누리 (더미 통계 기반 시각화)"
        )
        chart_path = render_poverty_chart()
        st.image(chart_path, caption="지역별·소득 분위별 에너지 빈곤율", use_container_width=True)

    with tab4:
        st.markdown(LOW_INCOME_INFO)


def main():
    st.set_page_config(
        page_title="에너지 빈곤 복지 정책 추천 시스템",
        page_icon="⚡",
        layout="wide",
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "정보 입력"

    st.title("⚡ 데이터 기반 에너지 빈곤 맞춤형 복지 정책 추천 시스템")
    st.caption("MLP 기반 위험도 분석 · LSTM 기반 비용 예측")

    if not check_internet():
        st.warning("인터넷 연결을 확인할 수 없습니다. 일부 외부 링크 및 최신 정보가 제한될 수 있습니다.")

    with st.sidebar:
        st.markdown("### 메뉴")
        page = st.radio(
            "페이지 선택",
            options=["정보 입력", "추천 결과", "정보 소개"],
            index=["정보 입력", "추천 결과", "정보 소개"].index(st.session_state["page"]),
            label_visibility="collapsed",
        )
        st.session_state["page"] = page
        st.divider()
        st.markdown("**개발 목적**")
        st.markdown(
            "저소득 가구의 에너지 빈곤 문제 해결을 위해 "
            "딥러닝 기반 정책 추천 및 비용 변화 예측을 제공합니다."
        )

    if st.session_state["page"] == "정보 입력":
        page_input()
    elif st.session_state["page"] == "추천 결과":
        page_results()
    else:
        page_info()


if __name__ == "__main__":
    main()
