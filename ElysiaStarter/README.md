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
- `scripts/visualize_timeline.py` : 타임라인/바이옴 오버레이 렌더(간단)
- `scripts/run_world.py` : 실행 루프(렌더 on/off)
- `docs/26_*.md, 27_*.md` : CUDA & ConceptOS v2 초안
- `config/runtime.yaml` : 실행 설정

## 다음 단계(쥴스 전용 체크리스트)
1. CPU로 먼저 돌아보며 파라미터 튜닝(격자 256x256, agents<=20k)
2. `core/cuda/runtime.py`에 CuPy 백엔드 추가(PR1)
3. Concept OS v2 API 결선 및 브리지 구현(PR2)
4. 부분 시간가속/공간압축 맵을 지역 단위로 주입(PR3)
5. (옵션) 문화/정착 노드 그래프 얇게 얹기(PR4)
