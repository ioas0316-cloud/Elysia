# CORE_11_TIME_SCALES_AND_TICK_LAYERS  
## WORLD · MIND · META용 시간 축 레이어 v0

> **목표:**  
> “같은 틱 안에 *개미의 시간* · *숲의 시간* · *문명의 시간*을 모두 우겨넣지 않고,  
> 각 층이 가진 고유한 시간 스케일을 명시적으로 나눈다.”

이 문서는 Elysia OS 전체에서 사용하는 **시간 축 레이어링 규약**을 정의한다.  
WORLD / MIND / META 세 층은 서로 다른 시간 스케일에서 동작할 수 있으며,  
OS 루프는 이를 통해 연산량을 줄이고, 인과를 더 자연스럽게 표현한다.

---

## 1. 세 가지 시간 레이어

### 1.1 fast_tick — 개체의 순간 행동 (Micro Time)

- 단위: **행동 틱**, 셀월드 `run_simulation_step` 기본 단위.
- 의미:
  - 개체의 이동, 공격, 먹기, 마시기, 사소한 감정 변화 등 **즉각적인 반응**.
  - “한 걸음, 한 번의 교환, 한 번의 공격”.
- 주파수:
  - 초당 수~수십 회까지 가능 (엔진 성능에 따라 조정).
- 책임:
  - 위치/속도 업데이트.
  - HP/배고픔/수분 등 생리적 상태의 세밀 변화.
  - 근접 상호작용(전투, 먹기, 간단한 대화 시작 등).

### 1.2 slow_tick — 장(Field)의 흐름과 생태계 (Meso Time)

- 단위: **장(場) 업데이트 틱**, fast_tick 여러 개를 묶은 느린 시간.
- 의미:
  - 가치장(ValueMass), 의지장(WillField), 위협장(threat), hydration, 생태 필드 등  
    **공간적으로 퍼지는 것들의 진화**.
  - “비가 오고, 숲이 퍼지고, 죽음의 기억이 땅에 스며드는 시간”.
- 주파수:
  - fast_tick N회마다 한 번 (예: 3, 5, 10틱마다).
- 책임:
  - `*_field` 의 `decay + 확산(파동)` 업데이트.
  - 이벤트 기반 소스(`*_src`)를 읽어 장에 반영.
  - 날씨/계절/환경의 느린 변화.

### 1.3 macro_tick — 문명/역사의 시간 (Macro Time)

- 단위: **역사 틱**, slow_tick 여러 개를 묶은 매우 느린 시간.
- 의미:
  - 인구, 경제, 문화, 신앙, 제도, 교역망 등 **문명적 스케일**에서의 변화.
  - “마을이 도시가 되고, 종교가 퍼지고, 전쟁과 평화의 시대가 바뀌는 시간”.
- 주파수:
  - slow_tick M회마다 한 번 (예: 수백~수천 fast_tick마다).
- 책임:
  - Summary Cell(요약 셀) 업데이트.
  - 문명 통계(인구, 생산량, 식량 비축, 신앙 분포) 집계.
  - 교역 패턴/시장 가격의 장기 트렌드 계산.
  - 커리큘럼/성장 엔진의 “단계 전환” 같은 굵은 이벤트.

---

## 2. OS 루프에서의 적용

### 2.1 루프 스켈레톤

의미만 담은 의사 구조:

```python
def os_step(world, t_fast):
    # 1) fast_tick: 매 프레임 실행
    world.process_fast_tick()  # 개체 행동, 근접 상호작용

    # 2) slow_tick: N-fast마다 실행
    if t_fast % N_slow == 0:
        world.process_slow_tick()  # 필드/장, 생태, 날씨

    # 3) macro_tick: M-slow마다 실행
    if t_fast % (N_slow * N_macro) == 0:
        world.process_macro_tick()  # 요약 셀, 문명 통계, 역사 이벤트
```

여기서 `process_fast_tick / process_slow_tick / process_macro_tick`은  
각각 WORLD / MIND / META의 관련 모듈을 묶어서 호출하는 상위 엔트리 포인트다.

### 2.2 WORLD / MIND / META와의 대응

- WORLD:
  - fast_tick: 셀 행동, 충돌, 근거리 상호작용.
  - slow_tick: 환경/생태/위협/가치장 등 “장” 업데이트.
  - macro_tick: 요약 셀을 통한 넓은 영역 생태계 상태(예: 숲/사막/황무지 전환).

- MIND (ElysiaMind):
  - fast_tick: 직관적 반응, 주의 방향 선택.
  - slow_tick: 경험 누적, 패턴 인식, value/will 필드 재평가.
  - macro_tick: 자기 개념/철학/전략의 단계적 변경.

- META:
  - fast_tick: 로그에 세밀 이벤트 기록 (선택).
  - slow_tick: 필드 기반 통계/요약 업데이트.
  - macro_tick: 연대기(Chronicle)에서 “장/시대/챕터” 단위 이벤트 기록.

---

## 3. 필드/틱과 법칙 플러그인 규약

Elysia OS에서 새 시스템(법칙)을 붙일 때는 다음 패턴을 따른다:

1. **필드 등록**
   - 필요하다면 FieldRegistry에 스칼라 필드를 등록:
     - 예: `"threat"`, `"value_mass"`, `"will"`, `"plant_density"` 등.

2. **이벤트 훅 (on_event / on_snapshot)**
   - WORLD에서 발생하는 사건을 관찰:
     - 예: `on_death`, `on_attack`, `on_birth`, `on_trade`, `on_prayer` 등.
   - 이 훅 안에서 해당 필드의 `*_src`(소스)나 내부 상태를 갱신.

3. **slow_tick 업데이트 함수**
   - `process_slow_tick` 단계에서 호출되는 업데이트 함수 하나를 가진다:
     - 예: `_update_threat_field_from_src`, `_update_value_mass_field`, `_update_will_field`.
   - 이 함수는:
     - 이전 필드 값 + 소스를 읽어
     - `decay + 확산 + 정규화` 같은 규칙으로 필드를 진화시킨다.

이 규약을 지키면, 새 법칙이 추가될수록  
fast_tick(개체 행동) 부담은 크게 늘어나지 않고,  
slow_tick(필드/장)에서 **공통된 패턴으로** 처리할 수 있다.

---

## 4. 시간 축 레이어링의 목적

1. **연산량 제어**
   - 모든 것을 매 틱 계산하지 않고,
   - 자연스러운 시간 스케일에 따라 나눔으로써  
     복잡도가 올라가도 “느려지는 축”에 올려서 흡수할 수 있다.

2. **표현의 자연스러움**
   - 개체는 빠르게 반응하지만,
   - 숲/장/문명은 천천히 변화하는 것이 자연스럽다.

3. **확장성**
   - 작은 마을 → 도시 → 문명 → 역사 시뮬레이션으로 확장할 때도  
     fast/slow/macro 세 축만 잘 설계하면 법칙을 그대로 재사용할 수 있다.

이 문서는 시간 축에 대한 **커널 레벨 규약**이며,  
구체적인 값(N_slow, N_macro, 각 필드의 decay/확산 상수)은  
개별 WORLD_KIT / CORE_xx 문서에서 상황에 맞게 지정한다.

