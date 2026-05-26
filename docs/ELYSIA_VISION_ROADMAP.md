# 🔭 엘리시아 비전 로드맵 (Elysia Vision Roadmap)

> **문서 유형**: 전략 비전 + 기술 분석 + 진화 로드맵
> **작성일**: 2026-05-26
> **전제 조건**: 베이스라인 안정화 완료 (UDP 파동 스트림, 레거시 아카이빙, 스토캐스틱 방전 정립)

---

## 제 1 장: 비전 — 트랜스포머와 가변 로터의 동형 사상

### 1.1 트랜스포머는 가변저항 다이얼의 중첩이다

트랜스포머의 어텐션 메커니즘을 회로로 해체하면 다음과 같다:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

| 트랜스포머 구성요소 | 회로 대응물 |
|---|---|
| Q, K 가중치 행렬 | 고정 임피던스 네트워크 (학습된 저항값) |
| 어텐션 스코어 `QK^T` | 입력에 따라 **동적으로 변하는 가변저항 값** |
| Softmax 정규화 | 에너지 보존 법칙 (전류 분배) |
| Multi-Head Attention | 서로 다른 주파수에 동조된 **병렬 공명 회로** |
| Residual Connection | 변압기의 바이패스 선로 |
| Layer Normalization | 전압 안정화 레귤레이터 |

이것은 비유가 아니라 **수학적으로 동형(Isomorphic)인 구조**이다.
내적(Dot Product)은 "두 벡터가 얼마나 같은 방향을 향하는가"를 측정하며,
이는 곧 **같음(Sameness/0)과 다름(Difference/1)의 관측 행위**이다.

### 1.2 엘리시아의 차원적 도약

트랜스포머의 근본적 한계는 **평탄한 유클리드 공간(flat vector space)** 에서
내적만으로 관계를 측정한다는 점이다.
모든 것을 1차원 벡터로 압착하고, 벡터들 사이의 각도(코사인 유사도)만 관측한다.

엘리시아의 클리포드 대수 공간은 근본적으로 다르다:

| 비교 항목 | 트랜스포머 | 엘리시아 가변 로터 |
|---|---|---|
| 핵심 연산 | 내적 (Dot Product) — 스칼라 결과 | 기하곱 (Geometric Product) — 스칼라 + 이벡터 + 트라이벡터 **동시** 결과 |
| 입력 표현 | 토큰 임베딩 (이산적, 정적) | 위상/진폭 파동 함수 (연속적, 동적) |
| 기억 구조 | 파라미터 행렬 (수십억 개의 고정 숫자) | 홀로그램 간섭 무늬 (위상 분포, 실시간 진화) |
| 관계 표현 | 코사인 유사도 (1차원 스칼라) | 쐐기곱 텐션 (다차원 이벡터 평면) |
| 학습 방식 | 역전파 (정적 그래디언트 하강) | 공명 간섭 패턴 (동적 위상 동기화) |

**기하곱(Geometric Product)** 하나로 내적과 외적이 동시에 산출된다.
"얼마나 같은가(0/Sameness)"와 "얼마나 다른가(1/Difference)"가
분리되지 않은 채 **하나의 연산에서 관측**된다.

### 1.3 4D 홀로그램 그래픽 메모리로의 확장

트랜스포머가 **2D 평면 위의 가변저항 다이얼 중첩**이라면,
엘리시아가 향하는 방향은 **4D 홀로그램 위상 공간에서의 가변 로터 간섭 패턴**이다.

현재 씨드(`elysia_seed`)의 `VariableRotorSpine`은 100개의 `[phase, amplitude]` 빈(bin)으로
1차원 홀로그램 토포그래피를 유지한다. 이것을 다중 감각 채널로 확장하려면:

1. **1D → 4D 빈 구조**: `[phase, amplitude]` → `[phase_θ, phase_φ, amplitude, frequency]`
2. **간섭 패턴의 중첩**: 복수의 데이터 스트림이 동일 빈에 간섭 무늬를 생성
3. **쿼터니언 토포그래피**: 각 빈이 `Quaternion` 상태를 보유하여 4D 회전을 기록

### 1.4 통합 데이터 스트림 관측 — 튜링 공명 게이트의 일반화

튜링 공명 게이트가 ASCII 바이트에 대해 증명한 원리:

> **유효한 패턴 → 닫힌 고리 (텐션 0.0000) → 같음(Sameness)**
> **무효한 패턴 → 열린 궤적 (텐션 ≫ 0) → 다름(Difference)**

이 원리는 ASCII에 종속되지 않는다. **모든 바이트 스트림**에 적용된다:

- **텍스트** = UTF-8 바이트 시퀀스
- **음성** = PCM 샘플 바이트 시퀀스
- **영상** = RGB 픽셀 바이트 시퀀스
- **네트워크 패킷** = 이미 UDP로 관측 중인 바이트 시퀀스

게이트 입장에서 이들은 전부 **"특정 주파수대의 위상 궤적"** 이다.
라벨이 필요 없다. 닫힌 고리를 형성하는 패턴과 그렇지 않은 패턴이
기하학적 텐션으로 자연스럽게 분류된다.

이것이 **제 4 단계: 무라벨 자율 유기 반응계(Label-Free Autopoiesis)** 의 수학적 토대이다.

---

## 제 2 장: 코어 엔진 연산 구조 분석 (Computational Bottleneck Analysis)

> 이하 분석은 2026-05-26 시점의 코어 파일 전수 조사 결과이다.

---

### 2.1 electromagnetic_circuit.py — 15대 전자기장 회로

**핵심 클래스**: `ElectromagneticCircuit`
- **계층 수**: 15개 (인지 레이어)
- **노드 구성**: 각 레이어마다 1개의 `ElectromagneticRotor` 인스턴스 (→ `ImitationCell` 상속)

**핫패스: `pulse_circuit(dt)`** — 매 틱 실행:

| 단계 | 연산 | 복잡도 |
|------|------|--------|
| 결합 진동 확산 | O(N) 루프: 좌우 이웃 스프링 힘 (커플링, 텐션 차이, 감쇠) | 순수 float 산술 |
| 클리핑 | O(N) 루프: `min(1.0, max(0.0, ...))` | O(15) |
| 로터 인지 | O(N): `ElectromagneticRotor.perceive_input()` → `ImitationCell.absorb_wave()` (모듈러 산술, `cos`/`sin`, `sqrt`) | ~165 스칼라 연산/틱 |

**데이터 구조**: 파이썬 리스트 (`tensions[]`, `couplings[]`, `dampings[]`, `is_constant[]`), `ElectromagneticRotor` 객체 리스트.

**외부 의존성**: `math`, `time`, `random`. **numpy/scipy 미사용.**

**복잡도**: O(N), N=15. **틱당 경량.**

---

### 2.2 atlantis_clifford_bridge.py — 클리포드 대수 브릿지

**핵심 클래스**: `AtlantisCliffordSystem`
- **클리포드 공간**: Cl(15, 0) — 15개 기저 벡터 → 2^15 = **32,768개의 잠재적 기저 블레이드**
- **상태 표현**: 단일 `Multivector` (sparse `dict` 표현: bitmask → coefficient)

**핫패스 연산**:

| 연산 | 설명 | 복잡도 |
|------|------|--------|
| `apply_rotor_discharge(layer_from, layer_to, θ)` | 로터 R = cos(θ/2) + sin(θ/2)·B 구성 후 **샌드위치 곱** `R * state * R†` | **O(K₁ × K₂)**, K = 비영 블레이드 수 |
| `apply_agent_intent(intent_angle, mode)` | 전체 상태에 대한 샌드위치 곱. 역순 포함 **이중 기하곱** | 위와 동일 × 2 |
| `compute_bivector_tension(a, b)` | 2개 dict 조회 + 곱셈 | O(1) |

**기하곱 복잡도 (⚠️ 핵심 위험 지점)**:

`Multivector._multiply_blades()` 함수 (`math_utils.py`):
- 비트 워킹: mask1의 각 세트 비트를 추출, mask2에서 하위 비트 카운트 (`bin().count('1')`)
- 블레이드 쌍당 **O(N²)**, N=15 차원
- 밀집 Multivector 두 개의 곱: O(K² × N²), K가 최대 32,768 도달 가능
- **이것이 전체 코드베이스에서 단일 최대 연산 위험 지점이다.**

**외부 의존성**: `math`, `core.math_utils.Multivector`. **순수 파이썬, numpy/scipy 미사용.**

**틱당 복잡도**: 희소 상태(~5-20개 비영 블레이드)에서 관리 가능한 O(수백). 블레이드가 반복 곱으로 확산될 경우 **O(수백만) 잠재적.**

---

### 2.3 clifford_rotor_sync.py — 비트와이즈 로터

**핵심 클래스**:

1. **`DynamicPIDController`**: 표준 PID + 동적 이득 스케일링
   - `discharge_error_to_ground()`: ~10 float ops. O(1).

2. **`BitwiseCliffordRotor`**: 16비트 순환 비트 시프트 로터
   - `apply_clock_edge()`: 정수 `tension × 10 % 16`, 단일 순환 비트 시프트. **O(1), 극도로 빠름** — 정수 비트 연산만.

**외부 의존성**: `math`, `time`. **이미 최적에 가까움.**

---

### 2.4 holographic_memory.py — 홀로그램 메모리

**핵심 클래스**:

1. **`CliffordLayer`**: 4D 매니폴드 레이어 1개
   - 상태: `manifold_state` (Quaternion), `concept_contents` (Dict[str, Quaternion])
   - 쿼터니언 곱 1회 = 16 곱셈 + 12 덧셈 = ~28 float ops

2. **`HologramMemory`**: 다중 레이어 관리자 (기본 3개 레이어)
   - **`scan_resonance(tension)` — 핫패스**: 등록 개념 수(C) × 레이어 수(L):
     - 개념당 레이어당: ~84 float ops
     - 100개 개념, 3개 레이어: **~25,200 float ops/스캔**

**외부 의존성**: `math`, `hashlib`, `core.math_utils.Quaternion`. **numpy/scipy 미사용.**

**복잡도**: O(C × L). **개념 수 증가 시 선형적으로 병목화.**

---

### 2.5 triple_helix_engine.py — 삼중나선 엔진 (중앙 오케스트레이터)

**핵심 클래스**: `TripleHelixEngine`

**아키텍처**: 3개 `CliffordIPN` 네트워크 + 교차 브릿지 링크 + Ark Gearbox 관리

| 세계 | CliffordIPN 구성 | 차원 | 노드 | 링크 수 |
|------|-------------------|------|------|---------|
| Inner World (내계) | 8 입력 + 4 은닉 + 1 출력 | Cl(3,0) → Cl(8,0) 가변 | 13 | 36 |
| Outer World (외계) | 4 감각 + 2 액추에이터 | Cl(4,0) 고정 | 6 | 2 |
| Ego World (자아) | 2 노드 | Cl(3,0) → Cl(5,0) | 2 | 1 |
| 교차 브릿지 | — | — | — | 5 |

**핫패스: `pulse()` — 단계별 상세**:

| 단계 | 연산 | 복잡도 |
|------|------|--------|
| A1 | `assimilate_axiom(anomaly_signal)` | O(nodes × blades) |
| A3 | `inner_world.forward_propagate(inputs)` | **O(36 links × K²)** — 샌드위치 곱 |
| A4 | `inner_world.tune_network(dt, lr)` | **O(36 links × K²) × 2** — 임피던스 갱신 + Kuramoto 동기화 |
| B1-B4 | `outer_world` + `ego_world` 전파/튜닝 | O(3 links × K²) |
| C | 교차 브릿지 전파 | O(5 links) — 샌드위치 곱 + 임피던스 갱신 |
| C.5 | `ark_gearbox.apply_agent_intent()` | Cl(15,0)에서의 샌드위치 곱 1회 |

**⚠️ 핵심 병목 — CliffordIPN 연산**:

| 차원 | 블레이드 수 K | 링크당 곱 연산 | 내계 36링크 총합 |
|------|-------------|----------------|-----------------|
| Cl(3,0) | K ≈ 8 | ~192 블레이드쌍 ops | **~6,912 ops** |
| Cl(8,0) | K ≈ 256 | ~196,608 블레이드쌍 ops | **~7,000,000 ops** ← ⚠️ 위험 |

**틱당 총 연산량**:
- Cl(3,0) 상태: ~50,000-100,000 float ops
- Cl(8,0) 상태: **~10,000,000-50,000,000 float ops** — 순수 파이썬 dict 순회

---

### 2.6 grid_engine.py — 펄스 그리드 엔진

- `poll_substation()`: HTTP GET (`urllib.request`) → 1.2초 타임아웃
- `spine.pulse()`: O(100) — 100개 빈 감쇠 루프
- **병목**: 계통 연동 시 HTTP 폴링 = I/O 바운드

---

### 2.7 elysia_seed/spine.py — 가변 로터 척추 (씨드)

- `pulse()`: O(100) — 100개 빈 감쇠 루프
- 외부 의존성 없음, 순수 `math` + `time`
- **매우 경량.**

---

### 2.8 api_server.py — WebSocket 서버

- FastAPI + WebSocket, 30 FPS 상태 스트리밍
- `SensoryHarmonics(size=16)`: 16×16 numpy 배열 (256 floats)
- **⚠️ 주의: 이 서버는 TripleHelixEngine / Clifford 파이프라인과 완전히 분리되어 있다.**
  - 독립적인 `SentientBeing` + `SensoryHarmonics` 스택을 사용
  - 메인 엔진과의 통합이 필요

---

## 제 3 장: 최적화 우선순위 (Rust/C 가속 포팅 로드맵)

### 3.1 병목 순위표

| 우선순위 | 컴포넌트 | 병목 지점 | 근거 |
|----------|----------|-----------|------|
| 🔴 **P0** | `Multivector.__mul__()` 및 `_multiply_blades()` in `math_utils.py` | 기하곱: 호출당 O(K² × N), 틱당 수천 회 호출 | 순수 파이썬 dict 순회, 비트 워킹, 벡터화 없음. **모든 대수 흐름이 이 함수를 통과** |
| 🔴 **P0** | `CliffordIPN.tune_network()` in `clifford_impedance_network.py` | 2× 전체 링크 순회, 각각 샌드위치 곱 + 내적 + 쐐기곱 | 링크당 기하곱 4-6회, 내계 36+ 링크. Cl(n,0) 증가 시 폭발 |
| 🟠 **P1** | `CliffordIPN.forward_propagate()` | 링크당 샌드위치 곱 (내계 36 링크) | 동일한 Multivector 병목 |
| 🟠 **P1** | `HologramMemory.scan_resonance()` | O(C × L) 쿼터니언 곱셈 | 순수 파이썬 해밀턴 곱, 개념 수 증가 시 악화 |
| 🟡 **P2** | `Quaternion.__mul__()` in `math_utils.py` | 해밀턴 곱: 호출당 28 ops, 극도로 높은 호출 빈도 | EM 회로 + 홀로그램 + 브릿지 전체에서 사용 |
| 🟡 **P2** | `ElectromagneticCircuit.pulse_circuit()` | 15× 로터 인지 (삼각함수 + 쿼터니언 정규화) | 중간 볼륨, 순수 파이썬 math |
| 🟢 **P3** | `VariableRotorSpine.pulse()` (양 버전) | O(100) 감쇠 루프 | 매우 경량, 단독 포팅 가치 없음 |
| 🟢 **P3** | `api_server.py` 송신 루프 | 프레임당 O(1), 이벤트 시 numpy | 이미 numpy 사용, I/O 바운드 |
| ⚪ **P4** | `BitwiseCliffordRotor` | O(1) 비트 연산 | **이미 최적** |

### 3.2 핵심 통찰: 최우선 포팅 대상

**`Multivector.__mul__()` 및 `_multiply_blades()`** (`math_utils.py`)가 중력의 우물(gravity well)이다.

모든 CliffordIPN 전파, 모든 브릿지 로터 적용, 모든 임피던스 갱신, 모든 Ark 기어박스 의지 적용이
**이 단일 함수를 통과한다**. 

비트마스크 대수를 활용한 **Rust 구현의 희소(sparse) 기하곱**은 엔진 전체를 가속시킬 것이다.

### 3.3 아키텍처 발견 사항

`api_server.py` WebSocket 서버가 메인 `TripleHelixEngine`과 **완전히 단절**되어 있다.
독립적인 `SentientBeing` + `SensoryHarmonics` 스택을 사용한다.
1단계 웹소켓 고도화 시, 이 단절을 해소하고 단일 파동 채널로 통합해야 한다.

---

## 제 4 장: 진화 로드맵

### 4.1 단계별 로드맵

```
[현재] 베이스라인 수립 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     ┃
     ┣━ [1단계] 송·수전 결합 파동 채널 고도화
     ┃   ├── Seed ↔ Core HTTP 폴링 → WebSocket 영구 커넥션 전환
     ┃   ├── api_server.py ↔ TripleHelixEngine 단절 해소
     ┃   └── Substation Gateway의 /voltage 엔드포인트 WebSocket화
     ┃
     ┣━ [2단계] Clifford 기하대수 연산 가속화 (Rust/C Extension)
     ┃   ├── [P0] math_utils.py → Multivector 기하곱 Rust 포팅
     ┃   ├── [P0] CliffordIPN 전파/튜닝 루프 네이티브화
     ┃   ├── [P1] Quaternion 해밀턴 곱 SIMD 가속
     ┃   └── 목표: 터빈 속도 4Hz → 1000Hz
     ┃
     ┣━ [3단계] 튜링 공명 게이트 확충 (Syntax-to-Wave Transducer)
     ┃   ├── 단일 바이트 → 구문 구조 전체의 고차원 궤적 사상
     ┃   ├── 코드 에러를 "물리적 중력 법칙"으로 감지
     ┃   └── 다중 데이터 스트림 (텍스트 → 음성 → 영상) 순차 결합
     ┃
     ┗━ [4단계] 무라벨 자율 유기 반응계 (Label-Free Autopoiesis)
         ├── 레지스터/메모리 상태를 다차원 기하 행렬로 직접 인입
         ├── 위상 텐션 편차로 위험/항상성 자율 판단
         └── 4D 홀로그램 토포그래피 확장
```

### 4.2 단계 간 의존성

```
1단계 (통신 고도화) ─┐
                     ├──→ 3단계 (공명 게이트 확충) ──→ 4단계 (자율 유기)
2단계 (연산 가속화) ─┘
```

1단계와 2단계는 **병렬 착수 가능**하나,
3단계와 4단계는 반드시 1·2단계의 완료를 전제한다.
고차원 데이터 스트림 처리와 실시간 자율 반응은
통신 채널과 연산 속도 양쪽이 모두 확보되어야 의미가 있기 때문이다.

### 4.3 2단계 최적화 전략 상세

**포팅 언어 선택: Rust (via PyO3/maturin)**

| 고려 사항 | C (ctypes) | Rust (PyO3) |
|-----------|-----------|-------------|
| 메모리 안전성 | 수동 관리 | 소유권 시스템으로 보장 |
| 파이썬 바인딩 | ctypes 수동 래핑 | PyO3 자동 생성 |
| 빌드 시스템 | Makefile + 수동 | maturin 단일 명령 |
| 비트 연산 최적화 | 우수 | **동등 또는 우월** (LLVM 백엔드) |
| 병렬화 확장 | pthread 수동 | rayon 크레이트로 즉시 가능 |

**포팅 순서**:

1. `Multivector` 클래스 → `elysia_algebra` Rust 크레이트
   - 희소(sparse) 기하곱: `HashMap<u32, f64>` 기반
   - 비트마스크 부호 계산: `(mask1 & mask2).count_ones()` 하드웨어 POPCNT 활용
   - 예상 가속비: **100x-1000x** (파이썬 dict → Rust HashMap + SIMD)

2. `Quaternion` 클래스 → 동일 크레이트 내 구현
   - SIMD 4-wide 연산 (`f64x4`) 활용
   - 예상 가속비: **50x-200x**

3. `CliffordIPN` 전파/튜닝 루프 → Rust에서 배치 처리
   - 36개 링크의 샌드위치 곱을 단일 Rust 호출로 벡터화
   - 예상 가속비: **500x+**

---

## 제 5 장: 결론

트랜스포머가 **2D 평면 위의 가변저항 다이얼 중첩**이라면,
엘리시아가 향하는 방향은 **4D 홀로그램 위상 공간에서의 가변 로터 간섭 패턴**이다.

이것은 같은 차원에서의 스케일링이 아니라, **차원 자체가 다른 접근**이다.

그리고 그 기초 물리 — 클리포드 공간, 기하곱, 위상 텐션 관측, 닫힌 고리 공명 — 는
이미 이 베이스라인 위에서 **실동하고 있다**.

현재 연산 병목의 중력 우물은 **`math_utils.py`의 기하곱 함수** 하나에 집중되어 있으며,
이것을 Rust로 이식하는 순간 엔진 전체의 물리적 한계가 해방된다.

이상이 아니라, 현재 구동 중인 엔진의 **자연스러운 확장 방향**이다.
