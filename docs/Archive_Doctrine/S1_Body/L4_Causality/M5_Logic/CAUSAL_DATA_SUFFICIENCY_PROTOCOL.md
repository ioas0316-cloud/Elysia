# Causal Data Sufficiency Protocol (v1)

> **목적**: "데이터가 부족하다"를 감각이 아닌 측정 가능한 기준으로 관리한다.

본 문서는 `CAUSAL_CONSTRAINT_BLUEPRINT.md`와 `CAUSAL_PROCESS_ROADMAP.md`를
실행 가능한 데이터 수집/검증 프로토콜로 연결한다.

---

## 1. 왜 필요한가

인과 시스템의 실패는 대개 모델 구조가 아니라 데이터 결핍에서 시작된다.
특히 다음 3가지 결핍이 치명적이다.

1. **전이 결핍**: 원인-의도-행동-결과가 한 레코드로 묶이지 않음
2. **실패 결핍**: 거부 사유와 회복 이력이 누락됨
3. **장기 결핍**: 드리프트/자기수정 효과를 장기 추적하지 못함

---

## 2. 최소 데이터 단위 (Causal Event Unit)

```yaml
CausalEvent:
  event_id: string
  timestamp: int
  domain: string              # e.g. reasoning|planning|execution|dialogue
  cause_id: string
  intent_vector: [float]
  state_before_hash: string
  gate_result: bool
  rejection_reasons: [string]
  action_id: string
  observed_outcome: string
  outcome_score: float        # 0.0 ~ 1.0
  state_after_hash: string
  recovery_attempted: bool
  recovery_result: string     # success|partial|fail|na
  resonance_score: float
```

원칙:
- `cause_id`가 없으면 이벤트는 무효로 간주한다.
- `gate_result=false`라도 반드시 저장한다 (실패 데이터가 핵심 자산).

---

## 3. 데이터 충분성 지표 (DSS)

## 3.1 Coverage 지표

- **Domain Coverage (DC)**: 최소 4개 도메인에서 이벤트 수집
- **Path Coverage (PC)**: 성공/거부/회복 경로 모두 기록
- **Temporal Coverage (TC)**: 일 단위 + 주 단위 집계 동시 확보

## 3.2 Quality 지표

- **Causal Completeness (CC)** = `cause_id+intent+outcome` 완비 레코드 비율
- **Recovery Traceability (RT)** = 거부 이벤트 중 회복 이력 연결 비율
- **Audit Explainability (AE)** = 로그만으로 허용/거부 이유 설명 가능한 비율

## 3.3 Stability 지표

- **Decision Consistency (DCon)**: 동일 조건 판정 일관성
- **Neutral Retention (NR)**: trinary `neutral(0)` 점유율 하한 유지율
- **Drift Detectability (DD)**: 드리프트 탐지 규칙 발동률 및 유효성

---

## 4. 최소 합격선 (Week 1~4)

다음 기준을 충족하면 "데이터 충분성 초기 통과"로 본다.

- `CC >= 0.95`
- `RT >= 0.80`
- `AE >= 0.85`
- `DCon >= 0.90`
- `NR >= 0.15` (도메인 평균)
- 주차별 이벤트 수: `>= 10,000`
- 거부 이벤트 비중: `>= 0.10` (실패 학습 데이터 확보 목적)

주의:
- 거부율을 무조건 낮추는 것은 목표가 아니다.
- 초기에는 실패 데이터를 충분히 모으는 것이 더 중요하다.

---

## 5. 4주 수집 계획 (Pilot)

## Week 1 — 계측 정합
- 이벤트 스키마 강제
- 누락 필드 탐지기 배치
- 일일 무결성 리포트 생성

## Week 2 — 실패 데이터 확보
- 의도적으로 경계 조건 입력 비중 확대
- 거부 사유 코드 체계 고정
- 회복 재시도 규칙 v1 적용

## Week 3 — 장기 드리프트 관측
- 동일 과제 반복 실험
- 임계치 변경 전/후 비교
- 자기수정 안정성 분산 추적

## Week 4 — 합격선 판정
- DSS 지표 계산
- Go/No-Go 1차 판정
- 다음 4주 보정 계획 수립

---

## 6. Go/No-Go 규칙 (데이터 관점)

### Go
- 최소 합격선 6개 중 5개 이상 충족
- 치명 결함(`cause_id` 누락, 로그 불가 설명)이 없음

### Conditional Go
- 합격선 6개 중 4개 충족
- 단, 미충족 항목에 대한 개선 계획과 종료 조건이 명시됨

### No-Go
- 합격선 3개 이하 충족
- 또는 `AE < 0.70` / `CC < 0.90` / 감사 불가 상태 발생

---

## 7. 운영 원칙

1. "성능 향상"보다 "추적 가능성"을 우선한다.
2. 게이트 통과율 상승만으로 성공을 선언하지 않는다.
3. 월 1회 과잉낙관 방지 리포트를 필수 발행한다.
4. 문서-코드-로그의 용어를 단일 사전으로 관리한다.

---

## 8. 문서 연결

- 설계도: [CAUSAL_CONSTRAINT_BLUEPRINT.md](CAUSAL_CONSTRAINT_BLUEPRINT.md)
- 단계 로드맵: [CAUSAL_PROCESS_ROADMAP.md](CAUSAL_PROCESS_ROADMAP.md)
- 레이어 인덱스: [INDEX.md](INDEX.md)
- 루트 인덱스: [../../../../../INDEX.md](../../../../../INDEX.md)

