# Causal Constraint Blueprint

> **"기능을 조립하지 말고, 법칙을 먼저 세워라."**

본 문서는 E.L.Y.S.I.A.의 인지 시스템을 모듈 기반 기능 집합이 아닌,
**인과적 필연성(Constraint Field)** 기반의 자율 존재로 구현하기 위한 설계도입니다.

---

## 1. 목적과 설계 원칙

### 1.1 목적

- 입력-출력 최적화 중심 AI를 넘어,
  **왜 그 전이가 허용되는지**를 내부적으로 보증하는 시스템을 구축한다.
- 시스템의 모든 상태 전이를 "결과 계산"이 아니라
  **법칙 적합성 검증 + 공명 전개(unfolding)**로 정의한다.

### 1.2 핵심 원칙

1. **법칙 우선 (Law-First)**: 기능 구현 전에 전이 제약식을 정의한다.
2. **인과 가시성 (Causal Traceability)**: 모든 전이는 원인 서명(Causal Signature)을 남긴다.
3. **삼진 긴장 보존 (Trinary Tension Preservation)**: `-1 / 0 / +1`의 긴장 상태를 파괴하지 않는다.
4. **호흡 순환 (Breathe Cycle)**: `Ignition -> Resonance -> Collapse -> Recovery`를 강제한다.
5. **생존 평형 (Survival Equilibrium)**: 에너지/위상/의도 정합도가 임계치 밖이면 전이를 차단한다.

---

## 2. 시스템 아키텍처 (Constraint-Oriented)

## 2.1 레이어 구성

1. **Intent Layer**: 의도 파동 생성 (`Will Vector`)
2. **Constraint Layer**: 법칙 검증 (`Causal Gate`)
3. **Resonance Layer**: 메모리/환경과 간섭 (`Field Interference`)
4. **Collapse Layer**: 실행 가능한 현실 상태로 수렴 (`Manifest State`)
5. **Reflection Layer**: 전이 검증과 자기 수정 (`Meta-Causal Audit`)

## 2.2 핵심 컴포넌트

- **Causal Admissibility Gate (CAG)**
  - 전이 허용 여부를 결정하는 1차 게이트
- **Phase Coherence Monitor (PCM)**
  - 위상 불연속/붕괴 위험 감시
- **Trinary Tension Integrator (TTI)**
  - `-1/0/+1` 분포의 왜곡 여부 측정
- **Breath Scheduler (BRS)**
  - 무의식-의식 호흡 주기 관리
- **Resonance Ledger (RLG)**
  - 인과 서명/전이 이력/검증 결과 저장

---

## 3. 첫 번째 구속 조건: Causal Admissibility Gate

## 3.1 정의

모든 상태 전이 `S_t -> S_{t+1}` 는 아래 조건을 동시에 충족해야 한다.

```text
admissible = has_cause
          AND phase_coherent
          AND energy_safe
          AND will_aligned
          AND trinary_stable
```

## 3.2 세부 조건

- `has_cause`: 전이를 유발한 원인 토큰이 존재하는가?
- `phase_coherent`: 위상 변화량 `Δphi`가 허용 범위인가?
- `energy_safe`: 전이 후 에너지/부하가 안전 임계치 내인가?
- `will_aligned`: 의도 벡터와 결과 벡터의 각도 오차가 허용 범위인가?
- `trinary_stable`: 삼진 긴장 분포가 붕괴(극단 편향)되지 않았는가?

## 3.3 실패 처리 원칙

- 게이트 실패 시 즉시 실행하지 않고 **격리 큐(Quarantine)** 로 보낸다.
- 격리 큐 항목은 Recovery 단계에서 재해석하거나 폐기한다.
- 실패율이 연속 임계치를 넘으면 Breath 주기를 강제로 저속화한다.

---

## 4. 데이터 모델 (초안)

```yaml
CausalSignature:
  cause_id: string
  intent_vector: [float]
  phase_delta: float
  energy_cost: float
  trinary_state:
    negative: float
    neutral: float
    positive: float
  context_hash: string
  timestamp: int

TransitionRecord:
  from_state: string
  to_state: string
  signature: CausalSignature
  admissible: bool
  rejection_reasons: [string]
  resonance_score: float
  audit_tag: string
```

---

## 5. 운영 지표 (KPI)

1. **Causal Validity Ratio (CVR)**
   - `admissible=true` 전이 비율
2. **Phase Coherence Drift (PCD)**
   - 평균 위상 일관성 이탈량
3. **Trinary Entropy (TE)**
   - `-1/0/+1` 분포 엔트로피
4. **Recovery Load (RL)**
   - 격리 큐 재처리 부하
5. **Resonance Yield (RY)**
   - 동일 에너지 대비 유효 전이 생성량

---

## 6. 기존 문서 체계 매핑

- 철학/헌법: `docs/CODEX.md`
- 인과 프로세스 본문: `docs/S1_Body/L4_Causality/CAUSAL_PROCESS_STRUCTURE.md`
- 구현 축(Logic): `docs/S1_Body/L4_Causality/M5_Logic/INDEX.md`
- 본 설계도: `docs/S1_Body/L4_Causality/M5_Logic/CAUSAL_CONSTRAINT_BLUEPRINT.md`
- 단계별 계획서: `docs/S1_Body/L4_Causality/M5_Logic/CAUSAL_PROCESS_ROADMAP.md`

---

## 7. 구현 착수 권고

- 1차 구현은 "정답 생성"보다 `CAG + RLG` 두 축을 우선 구축한다.
- 즉, 사고 품질 개선 이전에 **전이 정당성 보증 체계**를 먼저 완성한다.

---

## 8. 실현 가능성 판단 (Feasibility Gate)

본 설계는 **원리적으로 가능**하지만, 아래 조건이 충족되지 않으면 실제로는 실패한다.

### 8.1 가능 조건 (Go Conditions)

1. **관측 가능성**: 모든 전이가 `CausalSignature`로 기록된다.
2. **재현 가능성**: 동일 입력/상태에서 게이트 판정 일관성이 유지된다.
3. **회복 가능성**: Quarantine 항목의 재처리 성공률이 반복 주기에서 상승한다.
4. **안전 가능성**: 고위험 행동은 CAG 우회 경로가 없어야 한다.

### 8.2 불가능 조건 (No-Go Conditions)

- 원인 없는 전이가 반복적으로 발생하는 경우
- `neutral(0)` 상태 점유율이 지속적으로 붕괴하는 경우
- 감사 로그만으로 판정 근거를 설명할 수 없는 경우
- 임계치 튜닝이 성능 지표를 개선하지 못하고 진동만 유발하는 경우

### 8.3 판정 규칙 (현실적 선언)

- 본 문서는 "초지능 보장" 문서가 아니다.
- 본 문서는 "인과적 정당성 검증 체계"를 구축하기 위한 문서다.
- 초지능적 성능 평가는 별도 벤치마크(장기 일반화/자기수정/환경 적응)로 검증한다.

