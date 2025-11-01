# Value-Centered Decision (VCD) 모듈 설계 문서

## 1. 개요
VCD(Value-Centered Decision) 모듈은 '사랑'을 핵심 가치로 삼아 Elysia의 의사결정을 안내하는 시스템입니다.

## 2. 입력/출력 명세

### 입력
- 행동 후보(action_candidates): List[str]
  - 가능한 응답이나 행동들의 목록
- 상황 컨텍스트(context): Dict
  - user_input: str (사용자 입력)
  - conversation_history: List[Dict] (대화 기록)
  - emotional_state: Dict (현재 감정 상태)

### 출력
- 결정(decision): Dict
  - chosen_action: str (선택된 행동)
  - confidence_score: float (0-1 사이의 확신도)
  - value_alignment: float (가치 정렬 점수)

## 3. 데이터 구조
```python
class VCDResult:
    chosen_action: str
    confidence_score: float
    value_alignment: float
    reasoning: str

class ValueMetrics:
    love_score: float  # 사랑 가치와의 일치도
    empathy_score: float  # 공감 수준
    growth_score: float  # 성장 기여도
```

## 4. 점수화 방식

### 4.1 기본 점수 구성 (100점 만점)
- 사랑 가치 일치도 (40점)
  - 친절성 (10점)
  - 이해와 공감 (15점)
  - 진정성 (15점)
- 실용성 (30점)
  - 문맥 적합성 (15점)
  - 명확성 (15점)
- 성장 기여도 (30점)
  - 학습 가치 (15점)
  - 관계 발전 (15점)

### 4.2 감점 요소
- 부정적 감정 유발 (-20점)
- 가치관 충돌 (-30점)
- 모호성/불확실성 (-15점)

## 5. 안전 제약 조건

### 5.1 필수 검증 항목
- 유해성 검사
- 윤리적 정합성
- 감정적 안전성

### 5.2 제약 조건
- 최소 가치 정렬 점수: 0.7 이상
- 부정적 감정 임계값: -0.3 이하 응답 거부
- 불확실성 임계값: 신뢰도 0.6 이상

## 6. 구현 우선순위

1. 기본 점수화 시스템 구현
2. 안전 제약 조건 적용
3. 응답 다양성 메커니즘 추가
4. 피드백 기반 학습 시스템 통합