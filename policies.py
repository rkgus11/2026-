"""정책 정보 및 추천 로직"""

POLICIES = {
    "energy_voucher": {
        "name": "에너지 바우처",
        "category": "구입 비용 지원",
        "description": "저소득층 가구에 전기·도시가스·LPG·연탄·등유 등 에너지 구입비를 지원하는 제도입니다. "
        "기초생활수급자, 차상위계층 등 에너지 취약계층을 대상으로 합니다.",
        "support": "연간 최대 70만원(4인 가구 기준) 에너지 구입비 지원",
        "link": "https://www.korea.kr/news/policyNewsView.do?newsId=148881234",
        "discount_rate": 0.22,
    },
    "coal_coupon": {
        "name": "연탄 쿠폰",
        "category": "구입 비용 지원",
        "description": "연료비 부담이 큰 저소득 가구에 연탄 구입비를 지원하는 제도입니다. "
        "난방 연료로 연탄을 사용하는 가구를 우선 지원합니다.",
        "support": "가구당 연탄 구입비 할인·지원",
        "link": "https://www.korea.kr/",
        "discount_rate": 0.18,
    },
    "kerosene_voucher": {
        "name": "등유 바우처",
        "category": "구입 비용 지원",
        "description": "등유 난방을 사용하는 저소득 가구에 등유 구입비를 지원합니다. "
        "에너지 바우처와 연계하여 지원될 수 있습니다.",
        "support": "등유 구입비 일부 지원",
        "link": "https://www.korea.kr/",
        "discount_rate": 0.20,
    },
    "rate_discount": {
        "name": "요금 할인 제도",
        "category": "구입 비용 지원",
        "description": "전기·가스 요금에 대해 저소득층에게 할인 혜택을 제공하는 제도입니다. "
        "기초생활수급자, 차상위계층, 장애인 등이 대상입니다.",
        "support": "전기·가스 요금 30~50% 할인",
        "link": "https://www.korea.kr/",
        "discount_rate": 0.15,
    },
    "efficiency_program": {
        "name": "에너지 효율 개선 사업",
        "category": "효율 개선",
        "description": "노후 주택 및 저소득 가구의 단열·창호 교체 등 에너지 효율 개선을 지원하여 "
        "장기적으로 에너지 비용을 절감하는 사업입니다.",
        "support": "단열·창호 개선 공사비 일부 지원",
        "link": "https://www.nkis.re.kr/",
        "discount_rate": 0.12,
    },
    "led_distribution": {
        "name": "고효율 조명기기(LED) 보급",
        "category": "효율 개선",
        "description": "고효율 LED 조명기기를 보급하여 전력 소비를 줄이고 "
        "가계 전기요금 부담을 경감하는 사업입니다.",
        "support": "LED 조명기기 무상 또는 저가 보급",
        "link": "https://www.korea.kr/",
        "discount_rate": 0.05,
    },
}


def recommend_policies(risk_score: float) -> list[dict]:
    if risk_score >= 0.7:
        keys = ["energy_voucher", "coal_coupon", "kerosene_voucher"]
    elif risk_score >= 0.5:
        keys = ["rate_discount", "efficiency_program"]
    else:
        keys = ["led_distribution"]

    return [{**POLICIES[k], "id": k} for k in keys]


def get_max_discount(policies: list[dict]) -> float:
    if not policies:
        return 0.0
    return max(p["discount_rate"] for p in policies)


NEWS_ITEMS = [
    {
        "title": "2025년 에너지바우처 지원 대상 확대 검토",
        "date": "2025-05-12",
        "summary": "정부가 에너지 취약계층 지원을 강화하기 위해 에너지바우처 수혜 대상 확대를 검토 중입니다.",
        "source": "대한민국 정책브리핑",
    },
    {
        "title": "여름철 전력 요금 누진제 완화 논의",
        "date": "2025-04-28",
        "summary": "폭염 시기 저소득층 전기요금 부담 완화를 위한 누진제 조정 방안이 논의되고 있습니다.",
        "source": "에너지경제연구원",
    },
    {
        "title": "저소득층 에너지 효율 개선 사업 예산 증액",
        "date": "2025-03-15",
        "summary": "노후 주택 단열 개선 지원 예산이 전년 대비 15% 증액되어 더 많은 가구가 혜택을 받을 예정입니다.",
        "source": "국가정책연구포털",
    },
    {
        "title": "전국 에너지 빈곤율 8.2%…전년 대비 소폭 상승",
        "date": "2025-02-20",
        "summary": "에너지경제연구원 조사에 따르면 소득 대비 에너지 지출 10% 초과 가구 비율이 8.2%로 집계되었습니다.",
        "source": "지표누리",
    },
]

POVERTY_STATS = {
    "national_rate": 8.2,
    "by_region": {
        "서울": 5.1,
        "경기": 6.8,
        "인천": 7.2,
        "부산": 9.5,
        "대구": 10.1,
        "광주": 11.3,
        "대전": 8.7,
        "울산": 8.9,
        "강원": 12.4,
        "충북": 9.8,
        "충남": 8.5,
        "전북": 11.0,
        "전남": 13.2,
        "경북": 10.5,
        "경남": 9.1,
        "제주": 7.6,
    },
    "by_income_quintile": {
        "1분위(최저)": 28.5,
        "2분위": 15.2,
        "3분위": 6.8,
        "4분위": 2.1,
        "5분위(최고)": 0.4,
    },
}

LOW_INCOME_INFO = """
### 저소득층과 에너지 빈곤

**에너지 빈곤**이란 가구가 필요한 최소한의 에너지(난방, 냉방, 조명, 취사 등)를 
경제적으로 감당하지 못하는 상태를 말합니다. 국내에서는 일반적으로 **월 소득 대비 
에너지 비용이 10%를 초과**하는 경우 에너지 빈곤 가구로 분류합니다.

### 주요 대상 계층
- **기초생활수급자**: 국민기초생활보장법에 따른 수급 가구
- **차상위계층**: 기초생활수급자는 아니나 저소득 상태인 가구
- **에너지 취약계층**: 고령자, 장애인, 한부모 가구 등

### 에너지 빈곤의 영향
- 겨울철 난방 부족으로 인한 건강 악화
- 여름철 냉방 제한으로 인한 열사병 위험
- 아동 학습 환경 저하
- 가계 부채 증가 및 경제적 어려움 가중

### 참고 자료
- [에너지경제연구원 - 에너지 빈곤 지표 연구](https://www.keei.re.kr/)
- [대한민국 정책브리핑](https://www.korea.kr/)
- [국가정책연구포털](https://www.nkis.re.kr/main.do)
- [지표누리](https://www.index.go.kr)
"""

WELFARE_PROGRAMS = """
### 현재 시행 중인 주요 국가 복지 사업

#### 1. 에너지바우처
- **대상**: 기초생활수급자, 차상위계층 등
- **내용**: 전기·가스·연탄·등유 등 에너지 구입비 지원
- **신청**: 주소지 읍·면·동 행정복지센터 또는 복지로(bokjiro.go.kr)

#### 2. 전기·가스 요금 할인
- **대상**: 기초생활수급자, 차상위계층, 장애인 등
- **내용**: 전기·가스 요금 30~50% 할인
- **신청**: 한국전력, 지역 가스 공급사

#### 3. 에너지 효율 개선 사업
- **대상**: 저소득층, 노후 주택 거주 가구
- **내용**: 단열·창호 교체 등 에너지 효율 개선 지원
- **신청**: 지자체 에너지 관련 부서

#### 4. 연탄·등유 지원
- **대상**: 연탄·등유 난방 사용 저소득 가구
- **내용**: 난방 연료 구입비 지원
- **신청**: 주소지 행정복지센터
"""
