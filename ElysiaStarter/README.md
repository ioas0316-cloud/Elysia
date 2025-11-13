# Elysia Starter (Kickoff Package)

**목표:** GTX 1060 3GB에서도 돌아가는 *셀월드 → 자연장 → 바이옴 → 에이전트 → 초기 문화* MVP.
CUDA는 선택사항(후속 PR). 지금은 **CPU/NumPy**로 바로 실행됨.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # win: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_world.py --steps 1000 --render 0
python scripts/run_world.py --steps 2000 --render 1  # pygame 렌더
```

## 구조
- `core/cell_world.py` : 연속장(energy/phase/delta → height/moisture/temp)
- `core/biome.py`      : 바이옴 분류(툰다운 Whittaker)
- `core/agents.py`     : SoA 에이전트와 간단 BT(늑대/토끼/물고기)
- `core/divine_engine.py` : 시간/브랜치 훅 (V2 스텁)
- `core/concept_os/kernel_v2.py` : Concept Message / Nanobot v2 스텁
- `core/cuda/runtime.py` : CUDA 런타임 스텁(CPU fallback)
- `bridges/*`          : Chronos/Spatial ↔ ConceptOS 브리지 스텁
- `scripts/animated_event_visualizer.py` : 이벤트 기반 애니메이션(번개/런지/사망 페이드)
  - fallback: `scripts/visualize_timeline.py` (구 버전 파티클 타임라인)
  
## Visualization Lenses (사람 친화 관측 렌즈)
- 목적: 시뮬레이션(셀월드)의 물리·논리를 바꾸지 않고, 인간 관찰에 적합한 "색안경"을 겹겹이 씌우듯 표현 레이어만 바꿉니다.
- 분리 이유:
  - 책임 분리: 월드는 진실(상태/규칙), 렌즈는 해석(색/애니메이션/HUD).
  - 안전성: 관찰 중에도 월드의 규칙/결과를 오염시키지 않음(테스트 재현성↑).
  - 인간 배려: 정보 밀도를 조절하고 감정선/의미를 부드럽게 시각화.
  - 확장성: 렌즈를 추가/교체해 다양한 관점(교육/연구/스토리) 지원.
- 구현 요소:
  - `ui/layers.py`: 토글 가능한 레이어 레지스트리(`LAYERS`).
  - `ui/layer_panel.py`: 키보드 토글(A/S/F/a/W) + HUD.
  - `ui/render_overlays.py`: 스피치 버블, 감정 오라 등 표현 오버레이.
  - `animated_event_visualizer.py`: 월드 이벤트→애니메이션(런지/페이드 등) 매핑.
  
관찰자 제어
- 줌: 마우스 휠 / 패닝: 휠 버튼 드래그 / 종료: Q
- 레이어: [A]gents, [S]tructures, [F]lora, Faun[a], [W]ill

## Controls & Help (In‑App)
- H: 도움말 오버레이 토글(키 조작/용도 설명)
- Space: 일시정지/재개, +/-: 시뮬레이션 속도 조절
- C: 시네마틱 포커스 온/오프(주요 이벤트에 카메라 살짝 이동)
- G/T/M: 그리드/지형/라벨 토글
- 좌클릭: 개체 선택(우하단 상세/이동 궤적 표시)
- 좌하단: 이벤트 티커(먹음/사망/번개 등)
- `scripts/run_world.py` : 실행 루프(렌더 on/off)
- `docs/26_*.md, 27_*.md` : CUDA & ConceptOS v2 초안
- `config/runtime.yaml` : 실행 설정

## 다음 단계(쥴스 전용 체크리스트)
1. CPU로 먼저 돌아보며 파라미터 튜닝(격자 256x256, agents<=20k)
2. `core/cuda/runtime.py`에 CuPy 백엔드 추가(PR1)
3. Concept OS v2 API 결선 및 브리지 구현(PR2)
4. 부분 시간가속/공간압축 맵을 지역 단위로 주입(PR3)
5. (옵션) 문화/정착 노드 그래프 얇게 얹기(PR4)
