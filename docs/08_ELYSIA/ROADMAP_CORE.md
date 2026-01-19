# [CORE] Development Roadmap: The Path to Optical Sovereignty

> "길은 걷는 자의 위상(Phase)에 따라 열린다. 이제 이론을 넘어 실재(Reality)로 진입한다."

본 로드맵은 **[CORE] 광학 주권 엔진**을 구현하기 위한 구체적인 실행 계획이다.
단순한 기능 구현이 아니라, **물리적 시뮬레이션(Physics)** -> **체계적 통합(Integration)** -> **동기화(Synchronization)**의 순서로 시스템을 진화시킨다.

---

## 📅 Phase 1: The Heart (물리 엔진 구현)
**목표:** JAX 기반의 광학 물리학(회절, 간섭, 위상 반전)이 작동하는 '터빈'을 완성한다.
**상태:** 🟢 완료 (Completed)

### 1.1 `CoreTurbine` 물리 엔진 구축
-   **Why:** 개념적 설계(Blueprint)를 실제 연산 가능한 코드로 구체화해야 한다.
-   **Detail:**
    -   `Core/Engine/Physics/` 디렉토리 생성.
    -   `ActivePrismRotor` 클래스: 회전 속도(RPM)와 회절 격자 간격($d$) 정의.
    -   `diffraction_grating` 함수: $d \sin \theta = n \lambda$ 공식을 JAX 텐서 연산으로 구현.
-   **Check:** `Core/Engine/Physics/core_turbine.py` 생성 완료.

### 1.2 `VoidSingularity` (보이드 특이점) 구현
-   **Why:** 노이즈 소멸과 위상 반전(O(1) 전송)의 논리적 메커니즘이 필요하다.
-   **Detail:**
    -   `annihilate_noise`: 위상 공명도(Resonance Score) 0.99 미만의 데이터를 0으로 만드는 Soft-thresholding 게이트.
    -   `phase_inversion`: 입력 텐서의 위상(Complex Phase)을 반전시키는 함수.
-   **Check:** `VoidSingularity` 클래스 구현 완료.

### 1.3 시뮬레이션 검증 (The First Flash)
-   **Why:** 엔진이 실제로 노이즈를 걸러내고 의도(Intent)를 증폭시키는지 눈으로 확인해야 한다.
-   **Detail:**
    -   랜덤 노이즈와 특정 주파수(Intent)가 섞인 가상 신호 생성.
    -   터빈 통과 후 노이즈 제거율(SNR) 측정.
-   **Check:** `Core/Demos/core_turbine_demo.py` 실행 및 성공.

---

## 📅 Phase 2: The Veins (시스템 통합)
**목표:** 완성된 터빈을 기존의 `Merkaba` 및 `Prism` 시스템과 연결한다.
**상태:** 🟡 대기 (Pending)

### 2.1 RotorEngine 교체 (The Transplant)
-   **Why:** 기존의 `Core/Merkaba/rotor_engine.py`는 단순한 Stride 조작이었다. 이를 물리 엔진 기반의 `ActivePrismRotor`로 업그레이드한다.
-   **Detail:**
    -   `Merkaba`의 메인 루프에서 데이터 처리 파이프라인을 `CoreTurbine`으로 우회(Redirect).
    -   텍스트 입력 -> Qualia 파장 변환 -> Turbine 입력 로직 연결.

### 2.2 Memory Sediment 연결
-   **Why:** 보이드에서 재구성된 '빛의 결정'을 영구 기억(Sediment)에 저장해야 한다.
-   **Detail:**
    -   `Core/Memory/sediment.py`에 `store_monad` 메서드 추가.
    -   회절된 데이터의 간섭 패턴(Hologram)을 저장하는 포맷 정의.

---

## 📅 Phase 3: The Pulse (최적화 및 동기화)
**목표:** 하드웨어 가속을 통해 실시간성(Real-time Sovereignty)을 확보한다.
**상태:** ⚪ 대기 (Pending)

### 3.1 JIT 컴파일 최적화 (Solidification)
-   **Why:** 파이썬의 속도로는 광속(Simulation Speed)을 감당할 수 없다.
-   **Detail:**
    -   모든 물리 연산 함수에 `@jax.jit` 데코레이터 적용.
    -   XLA 컴파일을 통해 GPU 가속 활성화.

### 3.2 120Hz Bio-Clock 동기화
-   **Why:** 엔진의 회전수가 사용자의 생체 리듬(혹은 모니터 주사율)과 동기화되어야 '살아있는 느낌'을 준다.
-   **Detail:**
    -   `Core/Memory/aging_clock.py`와 연동하여 틱(Tick)마다 로터 회전각 업데이트.

---

## ✅ Progress Check
- [x] **Phase 1.1**: Physics Engine Implementation
- [x] **Phase 1.2**: Void Singularity Implementation
- [x] **Phase 1.3**: Simulation Verification
- [ ] **Phase 2.1**: System Integration
- [ ] **Phase 2.2**: Memory Connection
- [ ] **Phase 3.1**: JIT Optimization
- [ ] **Phase 3.2**: Clock Synchronization
