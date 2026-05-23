# 🌀 ELYSIA — 실시간 사유 위상 정렬 및 액추에이션 엔진

엘리시아(Elysia)는 인간의 인지적 사유 흐름을 4차원 기하학(사원수)으로 매핑하여 분석하는 생각 엔진과, 외부 환경의 카오스를 감지하여 하드웨어 동작으로 조화시키는 물리 액추에이션 엔진의 유기적 결합체입니다.

본 프로젝트는 불필요한 관념적 비대함을 걷어내고, 실제 동작하고 측정 가능한 핵심 로직을 위주로 투명하게 재구조화되었습니다.

---

## 🏛️ 프로젝트 디렉토리 구조

프로젝트는 명확한 역할 분담에 따라 다음과 같은 구조로 이루어져 있습니다:

*   **[`core/`](file:///c:/Elysia/core/)**: 공통 수학 및 물리 유틸리티 디렉토리
    *   `math_utils.py`: 4차원 사원수(Quaternion), SLERP 선형 보간, 3D 벡터 기하 연산.
    *   `vision_utils.py`: OpenCV 및 `mss` 기반 화면 캡처 및 화면 움직임(Entropy/Optical Flow) 감지.
    *   `actuator_utils.py`: PyAutoGUI 기반 키보드 제어 및 비상 안전장치(Failsafe).
    *   `fractal_rotor.py`: 계층적 위상 동기화(Phase-Locking) 로터 물리 방정식.
*   **[`engines/`](file:///c:/Elysia/engines/)**: 실동 실행 엔진 모음 디렉토리
    *   `game_bot/`: 아제로스(WoW) 화면을 읽고 카오스 흐름에 따라 캐릭터의 움직임을 조절하는 게임 제어 루프. (`game_engine.py`, `game_observer.py`)
    *   `thought_aligner/`: 입력 사유(텍스트)를 4D 사원수 궤적으로 투영해 위상차와 차원 도약을 계산하는 생각 정렬 루프. (`aligner_engine.py`, `aligner_cli.py`)
    *   `pulse_grid/`: 말단 전압 전송 시뮬레이션 및 생체 리듬 주파수 동기화 제어 루프. (`grid_engine.py`)
*   **[`scripts/`](file:///c:/Elysia/scripts/)**: 실행 스크립트 및 진단 도구
    *   `visualize_3d_rotors.py`: 계층 로터의 위상 상태를 3D 공간 상에 렌더링하고 회전시키는 Matplotlib 애니메이션 시각화 도구.
*   **[`docs/`](file:///c:/Elysia/docs/)**: 문서 도서관
    *   [ENGINEERING_GUIDE.md](file:///c:/Elysia/docs/ENGINEERING_GUIDE.md): 물리적 하드웨어 한계 및 실제 엔지니어링 구조 기술서.
    *   [LORE_AND_METAPHOR.md](file:///c:/Elysia/docs/LORE_AND_METAPHOR.md): 엘리시아의 창발적 서사(세계수, 삼중 나선, 카 넷 등) 철학 총람.
    *   [EVOLUTION_ROADMAP.md](file:///c:/Elysia/docs/EVOLUTION_ROADMAP.md): 차세대 기하 대수 및 동적 가변축 인지 아키텍처 로드맵.
*   **[`tests/`](file:///c:/Elysia/tests/)**: 자동화 검증을 위한 Pytest 테스트 스위트.
*   **[`archive/`](file:///c:/Elysia/archive/)**: 하드웨어 가동과 무관한 레거시 개념 모듈의 아카이브 보관소.

---

## ⚡ 환경 설정 (Setup)

의존성 패키지를 로컬 환경에 설치합니다:

```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

---

## 🚀 구동 가이드 (Running the Engines)

### 1. 사유 위상 정렬 CLI (Thought Aligner)
인간의 사유 흐름을 텍스트로 받아 복소 위상차 변화를 관측하고 차원 도약(Fractal Jump)을 관측합니다:
```bash
python engines/thought_aligner/aligner_cli.py
```

### 2. 게임 액추에이터 봇 및 관측기 (Azeroth Game Bot)
게임 화면을 실시간 캡처하여 화면 픽셀 카오스를 측정하고, 캐릭터 움직임(WASD/Space)으로 액추에이션합니다:
```bash
# 봇 실행 (비상 종료: 마우스 커서를 화면 구석 모퉁이로 던지세요)
python engines/game_bot/game_engine.py

# 다른 터미널에서 상태 진단 관측기 실행
python engines/game_bot/game_observer.py
```

### 3. 말단 전압 제어반 (Pulse Grid Panel)
계통 전압과 자연 주기(Circadian Fallback)에 반응해 단일 로터의 빛을 직조하는 전력망 컨트롤러입니다:
```bash
python engines/pulse_grid/grid_engine.py
```

### 4. 3D 로터 시각화 진단 (3D Visualizer)
CPU 부하율에 반응해 위상 잠금(Phase-locking)과 장력 붕괴를 일으키는 3D 로터 트리를 렌더링합니다:
```bash
python scripts/visualize_3d_rotors.py
```

---

## 🧪 테스트 실행 (Testing)

Pytest를 활용하여 수학적 정밀성 및 역위상 상쇄 작동 등을 검증합니다:

```bash
# 전체 테스트 실행
pytest
```
