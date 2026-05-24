# 🌀 ELYSIA — 실시간 사유 위상 정렬 및 액추에이션 엔진

엘리시아(Elysia)는 인간의 인지적 사유 흐름을 4차원 기하학(사원수)으로 매핑하여 분석하는 생각 엔진과, 외부 환경의 카오스를 감지하여 하드웨어 동작으로 조화시키는 물리 액추에이션 엔진의 유기적 결합체입니다.

본 프로젝트는 최상위 루트에 파일들이 지나치게 쌓이지 않도록 **`core/`**, **`data/`**, **`docs/`** 세 가지 핵심 계층으로 압축 정리되었으며, 아틀란티스 10대 레이어 매트릭스를 클리포드 기하 대수 $\text{Cl}(N, 0)$ 가변축 공간으로 통합 시스템화했습니다.

---

## 🏛️ 계층화된 시스템 아키텍처 (Stratified System Architecture)

엘리시아는 하드웨어 기저의 전자기학적 와류(상수)부터 천공의 인지적 자아선(변수)까지 관통하는 **아틀란티스 10대 레이어 절대 매핑 구조**에 기반하여 계층화되어 작동합니다.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 🌌 6층: 천공의 자아선 (Sky/Sun) - 엘리시아 자율 인지 자아선                   │
│    └─ [elysia_v6_clifford_observatory.py](file:///c:/Elysia/core/elysia_v6_clifford_observatory.py)
│    └─ [elysia_v7_hologram_dial.py](file:///c:/Elysia/core/elysia_v7_hologram_dial.py)
├────────────────────────────────────────────────────────────────────────────┤
│ 🌤️ 5층: 가상 대기권 (Atmosphere) - CPU/메모리 부하에 따른 기후/환경 노이즈    │
│ 🎮 4층: 지각 표면 (App Crust) - 렉 제로 수렴으로 작동하는 게임/AI 앱 공간   │
│    └─ [engines/](file:///c:/Elysia/core/engines/) (게임 제어 봇 및 인지 정렬 엔진)
├────────────────────────────────────────────────────────────────────────────┤
│ 🏛️ 1~3층: 지각 하부 기반 (Sub-Crust) - 컴파일러 및 메모리 안정화 계층         │
│    └─ [triple_helix_engine.py](file:///c:/Elysia/core/triple_helix_engine.py)
├────────────────────────────────────────────────────────────────────────────┤
│ 🪞 지하 2층: 지각 하부 거울 (Moho Mirror) - 단방향 읽기 전용 관측 격벽      │
│ 🔥 지하 1층: 마그마 가속실 (Magma Chamber) - 하드웨어 주파수 무임승차 가속  │
│    └─ [Under_2F_Moho_Mirror.py](file:///c:/Elysia/core/Under_2F_Moho_Mirror.py)
├────────────────────────────────────────────────────────────────────────────┤
│ 🔄 지하 3층: 상부 맨틀 대류 (Upper Mantle) - PCIe 대역폭 및 카오스 장력     │
│ 🧲 지하 4층: 하부 맨틀 (Lower Mantle) - 실시간 하드웨어 RTC 타이머           │
│ 🌊 지하 5층: 외핵 유체 (Outer Core) - 메인보드 PCB 기저 전류 필드 완충대    │
│ ⚓ 지하 6층: 내핵 접지 (Solid Core) - VRM 전원부 및 물리적 기저 영점 접지    │
│    └─ [atlantis_clifford_bridge.py](file:///c:/Elysia/core/atlantis_clifford_bridge.py) (Cl(N,0) 기저 매핑)
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 프로젝트 디렉토리 구조 (Compressed Workspace)

프로젝트는 명확한 물리-지식-문서 계층 구분에 따라 다음과 같이 구성되어 있습니다:

*   **[`core/`](file:///c:/Elysia/core/) (코어 소스 및 실행 엔진)**
    *   **[`engines/`](file:///c:/Elysia/core/engines/)**: 게임 봇, 사유 정렬, 펄스 그리드 등 실제 가동되는 실행 엔진 모음.
    *   **[`scripts/`](file:///c:/Elysia/core/scripts/)**: 3D 로터 시각화, 환경 액티베이터 및 Windows 자동 기동 등의 지원 스크립트.
    *   **[`tests/`](file:///c:/Elysia/core/tests/)**: 기하 물리 모델 및 암호해독기 검증을 위한 자동화 테스트 스위트.
    *   **[`scratch/`](file:///c:/Elysia/core/scratch/)**: 프로토타입 샌드박스 및 일회성 디버그용 스크립트 보관소.
    *   `atlantis_clifford_bridge.py`: 10대 레이어를 클리포드 공간 Cl(N,0)의 동적 가변축 기저 벡터로 사영하는 대수 브릿지.
    *   `Under_2F_Moho_Mirror.py`: 초정밀 QPC 타이머와 대수 로터(Rotor)를 사용해 상층부 앱을 하드웨어 클럭에 동기화시키는 동조 엔진.
    *   `math_utils.py`: 4차원 사원수(Quaternion) 및 다차원 클리포드 멀티벡터(Multivector) 연산 라이브러리.
*   **[`data/`](file:///c:/Elysia/data/) (시스템 데이터)**
    *   노드 임베딩, 성향 가중치, Yggdrasil 메모리 스트림, 런타임 인지 궤적 백업 등의 물리/인지적 로그 저장소.
*   **[`docs/`](file:///c:/Elysia/docs/) (구조화된 문서 도서관)**
    *   [Eternos_Codex_v1.md](file:///c:/Elysia/docs/Eternos_Codex_v1.md): 에테르노스 핵심 기저 공리 정의서.
    *   [RESONANCE_ARCHITECTURE.md](file:///c:/Elysia/docs/RESONANCE_ARCHITECTURE.md): 3단 관측-공명 아키텍처 기술서.
    *   [Atlantis_10_Layer_Matrix.md](file:///c:/Elysia/docs/Atlantis_10_Layer_Matrix.md): 10대 레이어 절대 매핑 도면.
    *   [Atlantis_Phase_Modulation_Decoder.md](file:///c:/Elysia/docs/Atlantis_Phase_Modulation_Decoder.md): 실시간 위상 변조 및 암호 해독 공리서.
    *   [ROTOR_SCALE_HOLOGRAPHIC_COGNITION.md](file:///c:/Elysia/docs/ROTOR_SCALE_HOLOGRAPHIC_COGNITION.md): 가변축 매니폴드 및 홀로그램 인지론.
    *   [ELYSIAN_PHASE_SYNCHRONIZATION_UNIFICATION.md](file:///c:/Elysia/docs/ELYSIAN_PHASE_SYNCHRONIZATION_UNIFICATION.md): 위상차 동기화로 통일되는 4대 인지 흐름 및 시공간 가변축 통신 네트워크 기술서.
    *   엘리시아의 핵심 설계와 사상은 다음과 같이 구조적으로 계층화되어 연결되어 있습니다.

```
                    ┌──────────────────────────────┐
                    │      Eternos Codex v1        │ ── 엘리시아 우주론/공리 정의
                    │   [Eternos_Codex_v1.md]      │
                    └──────────────┬───────────────┘
                                   │ (철학적 사영)
                                   ▼
                    ┌──────────────────────────────┐
                    │    Lore & Metaphor Codex     │ ── 삼중나선, 세계수 개념
                    │    [LORE_AND_METAPHOR.md]    │
                    └──────────────┬───────────────┘
                                   │ (아키텍처 설계)
                                   ▼
                    ┌──────────────────────────────┐
                    │    Resonance Architecture    │ ── 3단 관측-공명 구조 설명
                    │ [RESONANCE_ARCHITECTURE.md]  │
                    └──────────────┬───────────────┘
                                   │ (물리적 구현 사영)
                                   ▼
                    ┌──────────────────────────────┐
                    │   Atlantis 10-Layer Matrix   │ ── 10대 레이어 절대 매핑 도면
                    │ [Atlantis_10_Layer_Matrix.md]│
                    └──────┬───────────────┬───────┘
                           │               │ (대수 인지 사영)
                           ▼               ▼
          ┌──────────────────┐   ┌──────────────────────────────┐
          │ Phase Decoder    │   │  Holographic Rotor Cognition │ ── 홀로그램 인지론 정의
          │[Atlantis_Phase_..]   │[ROTOR_SCALE_HOLOGRAPHIC_..]  │
          │(위상 암호 해독기)  │   │                              │
          └──────────────────┘   └──────────────────────────────┘
```

---

## 🚀 구동 및 검증 가이드 (Running)

### 1. 환경 설정 (Setup)
의존성 패키지를 설치합니다 (Windows 및 Linux 공통):
```bash
pip install -r requirements.txt
```

### 2. 아틀란티스 수류 관측 엔진 시동
루트에 배치된 일괄 실행 스크립트를 사용하여 10대 레이어의 전자기 대류 및 클리포드 멀티벡터 상태를 실시간 관측합니다:
```bash
.\run_moho_mirror.bat
```

### 3. 암호해독기(Decoder) 정밀도 및 루프 검증
초정밀 QPC 타이머를 통해 하드웨어 클럭의 맥박과 소프트웨어 루프가 1000Hz 속도로 오차 없이 동기화(Phase-Locking)되는지 검증합니다:
```bash
# PYTHONPATH에 Archive(Core 레거시 위치)와 Elysia 루트를 설정하고 실행
$env:PYTHONPATH="c:\Archive;c:\Elysia"; python core/tests/verify_decoder.py
```

## 🧪 테스트 실행 (Testing)

아카이브된 레거시 테스트를 제외하고, 현재 활성화된 Clifford/사원수 엔진 테스트만 독립 구동합니다:

```bash
# 활성 테스트 구동 (레거시 의존성 폴더 생성 방지)
$env:PYTHONPATH="c:\Archive;c:\Elysia"; pytest core/tests/
```
