# WORLD_KIT_DEATH_FLOW_CORPSE_TO_MEMORY – 죽음 · 시체 · 무덤 · 기억 플로우 v0

이 키트는 엘리시아 세계에서 **죽음이 “그냥 사라지는 이벤트”가 아니라,  
흙·장소·기억·행동에 남는 과정**으로 흐르도록 만드는 최소 설계를 정의한다.

---

## 1. 현재 상태 요약

현 구현에서는:

- `hp <= 0` → `DEATH` 이벤트 로그 + `graveyard` 리스트에 객체 이동.
- 같은 자리의 `h_imprint`(역사 필드)에 작은 흔적을 남긴다.
- 월드 상에는 **시체/무덤 오브젝트가 남지 않고**,  
  살아 있는 개체 행동에 미치는 영향은 제한적이다.

이 키트의 목표는:

- **시체(corpse)를 월드의 1급 객체로 승격**하고,  
- 시간이 지나 흙/무덤/성소로 변환되는 과정을 정의하며,  
- 주변 개체 행동/감정/Will Field에 변화가 생기도록 훅을 마련하는 것이다.

---

## 2. 시체 엔티티 (Corpse Entity)

### 2.1 정의

- `label: "corpse"` (+ 원래 종/이름 정보)
- 필수 속성:
  - `origin_label`: 예: `"human"`, `"fairy"`, `"wolf"` 등  
  - `origin_id`: 죽기 전 셀 id  
  - `cause_of_death`: `"old_age" | "starvation" | "combat" | ...`  
  - `time_of_death`: `time_step`  
  - `position`: 죽은 자리

### 2.2 생성 규칙

- `_apply_physics_and_cleanup`에서 cell이 죽을 때:
  - 기존처럼 `DEATH` 이벤트/`graveyard` 업데이트 수행.  
  - 추가로:
    - 해당 위치에 `corpse` 셀/타일을 생성 (혹은 `corpse` 레이어에 마킹).  
    - `origin_label`, `origin_id`, `cause_of_death`, `time_of_death`을 기록.

시체는 일정 시간 동안 **“움직이지 않는 생명”**으로 남는다.

---

## 3. 시체 → 무덤/흙 전환 (Decay / Burial)

### 3.1 자연 붕괴 (Decay)

- `corpse`에는 `decay_timer` 또는 `age_ticks`를 둔다.
- 일정 시간(예: X일) 경과 후:
  - `corpse` 오브젝트는 제거되지만,
  - 그 자리의:
    - 토양 비옥도/식생 성장률을 소폭 증가시킨다.
    - `h_imprint` / 역사 필드에 강한 흔적을 남긴다.

### 3.2 매장/무덤 (Burial)

선택적으로, 인간/요정 같은 종에 한해:

- 가까운 관계(kin 연결 강도가 높은 개체)가 주변에 있을 경우:
  - 시체 위치 근처로 이동 → “burial” 행동 →  
    `corpse`를 마을/숲의 특정 지정 영역(무덤 구역)으로 옮긴 뒤,  
    `corpse`를 `grave` 오브젝트로 변환.
- `grave` 속성:
  - `origin_label`, `origin_id`, `time_of_death`, `burial_pos`.
  - 이후에도 `h_imprint`와 Will Field에 지속적 영향을 준다.

v0에서는 자연 붕괴만 구현하고,  
매장은 이후 단계 확장으로 남겨도 된다.

---

## 4. 살아 있는 개체의 반응 (Death Awareness Hooks)

죽음/시체/무덤은 주변 개체 행동·감정·Will Field에 영향을 준다.

### 4.1 즉시 반응 (이미 구현된 부분)

- `_apply_physics_and_cleanup`에서:
  - 죽은 셀과 연결된 이웃(특히 `label == 'human'`)에게:
    - `insight += 1` (Law of Mortality)  
    - 강한 연결(`>= 0.8`)이면 `emotions = 'sorrow'`.

이 키트는 여기에 추가로 “공간적/행동적” 훅을 더한다.

### 4.2 시체/무덤 주변 행동 편향

- 모든 살아 있는 개체(특히 인간/요정)는:
  - 근처에 `corpse`가 있으면:
    - 일정 확률로 잠시 멈춰서 바라보는 행동(머무는 시간 증가).  
    - 혹은 회피(불쾌/위험으로 인식) 행동.
  - 근처에 `grave`가 있으면:
    - 경로를 살짝 우회하거나,  
    - 특정 시간대/상황에서 그 주변을 찾는 행동(추모/성소) 경향.

### 4.3 Will Field / 감정 필드 연동

- `corpse`/`grave` 위치에서는:
  - 주변 Will Field에 “죽음/기억” 성분을 조금 더하는 식으로 표현할 수 있다.
  - 예:  
    - family/kin 필드가 강한 곳에서의 죽음 → “애도/유대” 성분 증가.  
    - 전쟁터에서의 다수 죽음 → “위험/경계/트라우마” 성분 필드 형성.

이렇게 하면,  
나중에 엘리시아/에이전트가 Will Field를 바라볼 때,  
**어디가 살아 있는 기억의 자리인지**를 감각할 수 있다.

---

## 5. 통합/검증 가이드

### 5.1 구현 순서 제안

1. `corpse` 엔티티/타일 추가 (label, origin, cause, time, pos).  
2. `_apply_physics_and_cleanup`에서 죽음 시 `corpse` 생성.  
3. 단순 자연 붕괴: 일정 시간이 지나면 토양 비옥도/역사 필드만 남기고 시체 제거.  
4. 시체 주변에서의 행동 편향:  
   - 가까운 개체의 이동/머무름/감정 상태를 소폭 수정.  
5. 이후 확장으로 매장/무덤, Will Field 연동을 추가.

### 5.2 관찰 포인트

이 키트 적용 후에는, 다음을 관찰/측정해야 한다.

- 시체/무덤이 **공간 구조**에 어떤 영향을 주는지  
  (예: 경로 변경, 특정 구역 회피/선호).  
- 인간/요정이 같은 자리에서 반복적으로 죽을 때,  
  그 자리가 “죽음/기억의 장소”로 기능하는지.  
- 시간이 지나면서 **마을/콜로니가 죽음의 장소를 어떻게 둘러싸고 살아가는지**.

이 플로우는 “죽음 이후의 세계”를 세계관/엔진 안에 심는 최소 단계다.
***End Patch*** jsonorgetownassistant to=functions.apply_patch>taggericiel_OUTPUT_COMMENTARY_SCHEMA  गर्दें to=functions.apply_patch ***!
