# 데이터 기반 에너지 빈곤 가구 맞춤형 복지 정책 추천 및 효과 예측 웹 시스템

MLP(에너지 빈곤 위험도)와 LSTM(월별 에너지 비용 예측)을 활용한 Streamlit 웹 애플리케이션입니다.

## 프로젝트 구조

```
project/
├── app.py              # Streamlit 메인 앱
├── train_mlp.py        # MLP 학습 스크립트
├── train_lstm.py       # LSTM 학습 스크립트
├── generate_data.py    # 더미 데이터 생성
├── setup_all.py        # 데이터 생성 + 모델 학습 일괄 실행
├── policies.py         # 정책 정보 및 추천 로직
├── requirements.txt
├── models/
│   ├── mlp_model.h5    # 사전 학습된 MLP 모델
│   └── lstm_model.h5   # 사전 학습된 LSTM 모델
└── data/
    └── dummy_data.csv  # 생성된 더미 데이터
```

## 설치 및 실행

### 1. Python 설치
Python 3.10 이상이 필요합니다. [python.org](https://www.python.org/downloads/)에서 설치 후 PATH에 추가하세요.

### 2. 패키지 설치

```bash
cd project
pip install -r requirements.txt
```

### 3. 데이터 생성 및 모델 학습

```bash
python setup_all.py
```

또는 개별 실행:

```bash
python generate_data.py
python train_mlp.py
python train_lstm.py
```

### 4. 웹 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

## 주요 기능

1. **정책 추천 (MLP)**: 소득, 주거 인원, 에너지 비용 기반 빈곤 위험도(0~1) 산출 후 정책 매칭
2. **비용 예측 (LSTM)**: 최근 12개월 패턴 기반 향후 6개월 에너지 비용 예측 및 정책 적용 효과 시뮬레이션
3. **정보 안내**: 복지 사업, 뉴스, 빈곤율 통계, 저소득층 안내

## 기술 스택

- Python, Streamlit, TensorFlow/Keras, pandas, scikit-learn, matplotlib

## 참고 자료

- [에너지경제연구원](https://www.keei.re.kr/)
- [대한민국 정책브리핑](https://www.korea.kr/)
- [국가정책연구포털](https://www.nkis.re.kr/main.do)
- [지표누리](https://www.index.go.kr)
