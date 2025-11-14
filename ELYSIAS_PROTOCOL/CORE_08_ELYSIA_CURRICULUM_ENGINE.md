# CORE-08: ELYSIA 성장 커리큘럼 엔진

엘리시아가 **세계(CellWorld + Render)**를 경험하면서  
1레벨(세계의 기초) → 5레벨(창조)까지 성장하도록 관리하는 **커리큘럼 엔진**의 설계 및 적용 구조다.

- 이 문서는 “아이디어”가 아니라 **쥴스가 바로 붙여 넣어서 쓸 수 있는 형태**를 목표로 한다.
- 구현은 두 층으로 나뉜다.
  - **데이터**: JSON 커리큘럼 정의
  - **코드**: Python `CurriculumEngine`

---

## 0. 한 줄 개념 정리

- **셀월드(CellWorld)**  
  “무슨 일이 일어났는지”가 기록되는 인과·상태 세계

- **렌더 엔진(Godot/Pygame 등)**  
  “어떻게 보이는지”를 담당하는 형상 세계

- **커리큘럼 엔진(Curriculum Engine)**  
  그 사이에서

  > “지금 엘리시아가 무엇을 배워야 하는가?”

  를 관리하는 **교장 선생님 계층**.

---

## 1. 파일/모듈 구조 제안

프로젝트 루트 기준 예시:

```text
Elysia/
  ELYSIA_CURRICULUM/
    curriculum_index.json      # 전체 커리큘럼 카탈로그
    level_1_world_basics.json  # 레벨 1: 세계의 기초
    level_2_causality.json     # 레벨 2: 인과
    level_3_meaning.json       # 레벨 3: 의미
    level_4_will_field.json    # 레벨 4: 의지장
    level_5_creation.json      # 레벨 5: 창조
  core/
    curriculum_engine.py       # 커리큘럼 엔진
    mind.py                    # ElysiaMind (perceive / decide 등)
    lens.py                    # 셀월드 ↔ 렌더 변환
  cell_world/
  render_engine/ (Godot 등)
```

- `ELYSIA_CURRICULUM/`  
  → “수업 계획서” 폴더.
- `curriculum_index.json`  
  → 어떤 레벨이 있는지, 순서, 잠금/해제 조건.
- 각 `level_X_*.json`  
  → 해당 레벨에서 **무엇을 경험하고, 언제 ‘배웠다’고 인정할지** 정의.

실제 코드 트리에 맞춰 `core/` 위치는 조정 가능하다. 중요한 것은 **커리큘럼 전용 디렉터리와 엔진 모듈을 분리**하는 것이다.

---

## 2. 커리큘럼 JSON 스펙

### 2.1 레벨 파일 공통 구조

각 레벨 파일(예: `level_1_world_basics.json`)은 대략 다음과 같다.

```json
{
  "id": "L1_WORLD_BASICS",
  "name": "Level 1: World Basics",
  "description": "공간, 시간, 기본 객체들을 감지하고 구분하는 단계.",
  "unlock_condition": {
    "type": "always"
  },
  "goals": [
    {
      "id": "SEE_DAY_NIGHT",
      "type": "experience_count",
      "event": "world_cycle.day_night",
      "min_count": 3,
      "description": "낮/밤 전환을 3회 이상 관찰한다."
    },
    {
      "id": "SEE_STATIC_VS_DYNAMIC",
      "type": "distinct_entities",
      "tags": ["static_object", "living_entity"],
      "min_static": 3,
      "min_living": 3,
      "description": "정적인 것/움직이는 것을 최소 3종씩 구분한다."
    }
  ],
  "success_condition": {
    "type": "all_goals_completed"
  },
  "unlocks": [
    "L2_CAUSALITY"
  ],
  "reward": {
    "type": "flag",
    "set_flags": ["HAS_WORLD_SCHEMA_V1"]
  }
}
```

**핵심 필드**

- `unlock_condition`  
  - 언제 이 레벨을 **시작할 수 있는지**  
  - 예: 이전 레벨 완료, 특정 플래그 설정 등.
- `goals[]`  
  - “이걸 경험하면 통과” 조건 모음. 이벤트/상태 기반.
- `success_condition`  
  - “이제 이 레벨은 끝났다”를 판단하는 규칙.
- `unlocks`  
  - 완료 후 열릴 다음 레벨 ID 리스트.
- `reward`  
  - 플래그 설정, 내부 파라미터 변경 등.

### 2.2 커리큘럼 인덱스 (`curriculum_index.json`)

```json
{
  "levels": [
    { "id": "L1_WORLD_BASICS", "file": "level_1_world_basics.json" },
    { "id": "L2_CAUSALITY",    "file": "level_2_causality.json" },
    { "id": "L3_MEANING",      "file": "level_3_meaning.json" },
    { "id": "L4_WILL_FIELD",   "file": "level_4_will_field.json" },
    { "id": "L5_CREATION",     "file": "level_5_creation.json" }
  ],
  "start_level": "L1_WORLD_BASICS"
}
```

에이전트 입장에서는:

> “이 인덱스를 읽어서, 현재 레벨 JSON만 로드해서 CurriculumEngine에 넣는다”

라고 이해하면 된다.

---

## 3. 커리큘럼 엔진 파이썬 스켈레톤

`core/curriculum_engine.py`에 들어갈 수 있는 형태의 뼈대다.  
(실제 세부 구현은 프로젝트 구조에 맞춰 채우면 된다.)

```python
# core/curriculum_engine.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import json
from pathlib import Path


@dataclass
class GoalState:
    id: str
    progress: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False


@dataclass
class LevelState:
    id: str
    goals: Dict[str, GoalState] = field(default_factory=dict)
    completed: bool = False


class CurriculumEngine:
    """
    Elysia 성장 커리큘럼을 관리하는 엔진.
    - 현재 레벨 로드
    - 이벤트/상태 피드백으로 목표 진행
    - 레벨 완료 판단 → 다음 레벨 언락
    """

    def __init__(self, curriculum_dir: Path, index_file: str = "curriculum_index.json"):
        self.curriculum_dir = curriculum_dir
        self.index = self._load_index(index_file)
        self.current_level_id: str = self.index["start_level"]
        self.current_level_spec: Dict[str, Any] = self._load_level(self.current_level_id)
        self.level_state = self._init_level_state(self.current_level_spec)
        self.flags: Set[str] = set()

    def _load_index(self, filename: str) -> Dict[str, Any]:
        path = self.curriculum_dir / filename
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_level(self, level_id: str) -> Dict[str, Any]:
        entry = next(l for l in self.index["levels"] if l["id"] == level_id)
        path = self.curriculum_dir / entry["file"]
        return json.loads(path.read_text(encoding="utf-8"))

    def _init_level_state(self, spec: Dict[str, Any]) -> LevelState:
        goals = {
            g["id"]: GoalState(id=g["id"])
            for g in spec.get("goals", [])
        }
        return LevelState(id=spec["id"], goals=goals)

    # --- 외부에서 호출되는 핵심 API ---

    def on_world_event(self, event: Dict[str, Any]) -> None:
        """
        셀월드/렌더에서 온 이벤트를 커리큘럼에 전달한다.
        event 예시:
        {
          "type": "world_cycle.day_night",
          "payload": {...}
        }
        """
        if self.level_state.completed:
            return

        for goal_spec in self.current_level_spec.get("goals", []):
            goal_state = self.level_state.goals[goal_spec["id"]]
            if not goal_state.completed:
                self._update_goal_from_event(goal_spec, goal_state, event)

        # 레벨 완료 체크
        if self._check_level_completed(self.current_level_spec, self.level_state):
            self._apply_reward(self.current_level_spec.get("reward", {}))
            self._advance_level()

    def on_world_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        주기적으로 전체 월드 상태를 보고하는 경우에 사용.
        상태 기반 목표(distinct_entities 등)에 활용 가능.
        """
        # 필요 시 구현
        pass

    # --- 내부 로직 ---

    def _update_goal_from_event(self, spec: Dict[str, Any], state: GoalState, event: Dict[str, Any]) -> None:
        gtype = spec["type"]

        if gtype == "experience_count":
            if event.get("type") == spec["event"]:
                count = state.progress.get("count", 0) + 1
                state.progress["count"] = count
                if count >= spec["min_count"]:
                    state.completed = True

        elif gtype == "distinct_entities":
            # 예: static/living 태그를 모아서 개수 채우기
            tag = event.get("tag")
            seen_static = state.progress.get("static_ids", set())
            seen_living = state.progress.get("living_ids", set())

            if tag == "static_object":
                seen_static.add(event["entity_id"])
            elif tag == "living_entity":
                seen_living.add(event["entity_id"])

            state.progress["static_ids"] = seen_static
            state.progress["living_ids"] = seen_living

            if len(seen_static) >= spec["min_static"] and len(seen_living) >= spec["min_living"]:
                state.completed = True

        # 필요하면 다른 goal type도 여기에 추가

    def _check_level_completed(self, spec: Dict[str, Any], state: LevelState) -> bool:
        cond = spec.get("success_condition", {"type": "all_goals_completed"})
        if cond["type"] == "all_goals_completed":
            return all(g.completed for g in state.goals.values())
        # 다른 조건 타입도 확장 가능
        return False

    def _apply_reward(self, reward_spec: Dict[str, Any]) -> None:
        if reward_spec.get("type") == "flag":
            for f in reward_spec.get("set_flags", []):
                self.flags.add(f)

    def _advance_level(self) -> None:
        self.level_state.completed = True
        unlocks = self.current_level_spec.get("unlocks", [])
        if not unlocks:
            return  # 마지막 레벨

        # 단순히 첫 번째 unlock으로 진행
        next_id = unlocks[0]
        self.current_level_id = next_id
        self.current_level_spec = self._load_level(next_id)
        self.level_state = self._init_level_state(self.current_level_spec)
```

---

## 4. Runtime에서의 연결 (예시)

월드 루프에서 커리큘럼 엔진을 사용하는 방식 예시:

```python
# core/runtime_loop.py (예시)

from pathlib import Path
from core.curriculum_engine import CurriculumEngine

curriculum = CurriculumEngine(curriculum_dir=Path("ELYSIA_CURRICULUM"))

def main_loop():
    while True:
        # 1) 셀월드/렌더에서 이벤트 수집
        events = collect_world_events()

        # 2) 각 이벤트를 커리큘럼에 전달
        for ev in events:
            curriculum.on_world_event(ev)

        # 3) 엘리시아 의사결정에 커리큘럼 상태를 참고시키고 싶다면:
        #    ElysiaMind.update(
        #        curriculum_state=curriculum.level_state,
        #        flags=curriculum.flags
        #    )

        # 4) 나머지 월드 처리 ...
```

이렇게 하면 **“그냥 세계를 돌리기만 해도”**  
엘리시아는 자연스럽게 레벨 1 → 2 → 3 … 으로 성장할 수 있다.

---

## 5. 레벨 2 이후 예시 (인과 레벨)

레벨 2 (인과) 예시:

```json
{
  "id": "L2_CAUSALITY",
  "name": "Level 2: Causality",
  "description": "행동과 결과의 연결을 감지하고 패턴을 인식한다.",
  "unlock_condition": {
    "type": "flag_required",
    "flag": "HAS_WORLD_SCHEMA_V1"
  },
  "goals": [
    {
      "id": "SEE_ACTION_REACTION",
      "type": "experience_count",
      "event": "interaction.push_response",
      "min_count": 5,
      "description": "어떤 존재가 다른 존재를 밀었을 때, 그 반응을 5회 이상 관찰한다."
    }
  ],
  "success_condition": {
    "type": "all_goals_completed"
  },
  "unlocks": ["L3_MEANING"],
  "reward": {
    "type": "flag",
    "set_flags": ["HAS_CAUSAL_SCHEMA_V1"]
  }
}
```

이런 식으로 **엘리시아가 겪었으면 하는 경험들**을  
각 레벨 JSON의 `goals`로 하나씩 정의하면 된다.

---

## 6. 강덕(사용자)이 지금 당장 할 수 있는 일

1. 에이전트/빌더에게 다음만 전달하면 된다.
   - `ELYSIA_CURRICULUM/` 폴더 생성
   - `curriculum_index.json` 및 `level_1_world_basics.json`부터 구현
   - 위 `CurriculumEngine` 스켈레톤을 적절한 `core/` 위치에 추가
   - 셀월드/렌더 루프에서 이벤트를 `on_world_event()`로 넘기도록 연결

2. 자신은
   - “엘리시아가 겪어야 할 레벨 1~5 경험 목록”을 자연어로 계속 정리  
   - 이 문서와 함께, 에이전트가 JSON 골격으로 변환해 줄 수 있다.

---

## 7. 요약

- 셀월드 = 학교  
- 렌더 엔진 = 교실/운동장  
- 커리큘럼 엔진 = 시간표/교장  
- 엘리시아 = 그 학교에 다니며 자라나는 아이

> **세계는 이미 만들어졌다.  
> 이제 “어떻게 자라게 할 것인가”를 정의하는 단계다.**

