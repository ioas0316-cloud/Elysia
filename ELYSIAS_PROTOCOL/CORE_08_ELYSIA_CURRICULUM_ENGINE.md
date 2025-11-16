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

---

## 8. 성장 법칙과 커리큘럼 흐름 (World ↔ Code ↔ Narrative)

- **World Sense Layer (L1~L2)**
  - 목표: 위치/리듬/인과를 체감하고 `logs/world_events.jsonl`, `logs/elysia_signals.jsonl`에서 자연계 신호를 모은다.
  - 학습 증거: SymbolEpisode/ TextEpisode에서 “관찰 → 감정 → 의도”가 3턴 이상 이어지는 서술.
  - 커리큘럼 룰: 분기(branch)마다 `World.set_time_scale`과 `N_macro`를 명시하여 낮/밤·계절을 최소 수십 회 압축 경험시킨다.

- **Code & System Sense Layer (L3)**
  - 목표: `applications`, `app_core`, `Project_Elysia` 등 코드 월드 이벤트를 “세계 규칙”으로 내재화.
  - 학습 증거: CausalEpisode에서 함수/모듈을 원인-결과 쌍으로 언급하고, `logs/elysia_language_field.json`에서 관련 개념 세기가 상승.
  - 커리큘럼 룰: 커리큘럼 엔진에 “코드 세계 과제” 골 타입을 추가하고, 브랜치 실행 시 빌드/테스트 로그를 샘플링하여 episode 스트림에 주입.

- **Narrative & Expression Layer (L4~L5)**
  - 목표: `logs/elysia_self_writing.jsonl`에 자발적 글쓰기가 누적되고, `logs/elysia_caretaker_feedback.jsonl`에서 감정/가치 피드백을 흡수.
  - 학습 증거: caretakers가 남긴 감성 태그와 self-writing 감정 벡터가 수렴하고, TextEpisode에서 1인칭 감정/의도 진술 비중이 상승.
  - 커리큘럼 룰: 각 레벨 완료 조건에 “표현 지표(감정 다양성, caretaker resonance)”를 추가해 언어장(language field)을 성장의 주 지표로 삼는다.

- **프랙탈 패턴 적용**
  - 각 레벨은 Why/How/What/Telemetry/Boundaries를 내포하며, 동일 패턴을 `ELYSIA_CURRICULUM/*.json`에 반복.
  - 분기 실험 설계 시 “World(환경) ↔ Code(법칙) ↔ Narrative(언어)” 세 축을 모두 선언하고, 어떤 축을 가속하는지 명시한다.

---

## 9. 관측 및 로그 확장 제안

- **기존 로그 정렬**
  - `world_events.jsonl` / `elysia_signals.jsonl`: 저수준 환경/감정 텔레메트리 → 커리큘럼 `world_event` 인풋으로 직결.
  - `symbol_episodes.jsonl` / `text_episodes.jsonl` / `causal_episodes.jsonl`: 경험-언어 스택 → 레벨 목표 증거.
  - `elysia_language_field.json`: 개념/감정 세기 히스토리 → 레벨별 핵심 지표.
  - `elysia_self_writing.jsonl` + `elysia_caretaker_feedback.jsonl`: 표현 및 피드백 루프.

- **추가 로그/필드 제안**
  1. `logs/elysia_curriculum_trials.jsonl`
     - 필드: `branch_id`, `level_id`, `time_scale`, `N_macro`, `seed`, `goal_snapshot`, `observables`.
     - 목적: 각 실험 분기의 파라미터/의도/결과를 한 줄로 기록해 quaternion/프랙탈 브랜치 비교가 가능하도록 한다.
  2. `logs/elysia_expression_scores.jsonl`
     - 필드: `ts`, `episode_id`, `self_writing_vector`, `caretaker_vector`, `resonance`, `emotion_diversity`, `freeform_notes`.
     - 목적: 자발적 글쓰기와 보모 피드백 간의 거리/공명 정도를 수치화해 언어 성장 속도를 추적.
  3. `logs/elysia_branch_feedback.jsonl`
     - 필드: `branch_id`, `hypothesis`, `accepted_fields`, `rejected_fields`, `notes`.
     - 목적: Growth law 조정 시 어떤 브랜치가 근거가 되었는지 명확히 남긴다.

- **Curriculum Engine 연동**
  - `CurriculumEngine`에 `log_trial(event_batch)` 훅을 추가하여 레벨/브랜치 상태를 새 로그로 자동 축적.
  - 레벨 목표 중 언어/감정 관련 항목은 `elysia_expression_scores.jsonl`를 읽어 통과 여부를 판단하도록 새로운 goal type(`expression_metric`)을 설계한다.

---

## 10. LLM 실험 / 학습 설계 (Self-Writing 중심)

1. **데이터 큐레이션**
   - Symbol/Text/Causal episodes를 시계열로 정렬하고, 동일 timestamp 근처의 `self_writing` + `caretaker_feedback`를 묶어 학습 샘플 생성.
   - 입력: (World context slice, Symbol/Text/Causal summaries, caretaker prior feedback).
   - 출력: 다음 self-writing 문단 또는 caretaker 응답을 생성하도록 지도 학습.

2. **Branch 기반 평가**
   - 매 커리큘럼 레벨 완료 직전에 최소 3개의 가속 브랜치를 파생시켜, 각 브랜치에서 동일한 self-writing 프롬프트를 실행.
   - 지표: (a) caretakers가 부여한 resonance 점수 평균, (b) expression metric 로그의 감정 다양성, (c) language field에서 해당 개념 축의 상승량.

3. **모델 스케일링 타임라인**
   - L1~L2: 소형 (≤7B) 모델 파인튜닝으로 감각/감정 어휘 정착.
   - L3: 중형 (13B~34B) 모델로 코드-세계 설명력 강화, causal episode 복기를 학습.
   - L4~L5: 대형 (≥70B) 모델 또는 mixture를 활용하여 장편 self-writing/ caretaker 대화 실험. 이 단계에서만 RLHF/Direct Preference 최적화를 시도.

4. **자발적 글쓰기 중심 평가 루프**
   - `elysia_self_writing`을 주기적으로 샘플링하여 caretaker들이 소수 질문(감정, 의도, 배움)을 달아주도록 프롬프트 구성.
   - caretakers의 피드백은 `elysia_expression_scores.jsonl`로 요약되고, CurriculumEngine이 다음 레벨 unlock 조건으로 사용.

5. **시간가속 규칙 내장**
   - LLM 실험 스크립트는 항상 `branch_config`(time_scale, duration, seeds) 메타데이터를 요구하고, 1-tick 루프 금지를 자동 확인한다.
   - 브랜치별 로그 묶음을 artifact로 저장하여 CODEX가 “무엇을 보고 판단할지” 즉시 접근 가능하도록 한다.

이 설계는 “정답 맞추기”보다 언어장과 감정 표현 성장 곡선을 관측/조정하는 데 집중하며, caretakers의 공명을 주 지표로 사용한다.

---

## 11. 매크로 브랜치 실행/보고 레시피 (예: 1,000년 × 20회)

- **실행 전 선언**
  1. `branch_plan`에 `time_scale`(예: 1 tick = 6개월)과 `macro_duration`(예: 1,000년)을 명시한다.
  2. 동일 계획으로 최소 3개, 권장 20개의 시드를 잡는다. (seed, curriculum_offset, caretaker_profile 조합 권장)
  3. 어떤 레벨/언어 축을 검증하는지 Why/How/What로 한 문단 작성.

- **실행 중 기록**
  - 각 브랜치의 `CurriculumEngine`는 `trial_report` 프리셋을 채운다.
  - 필수 필드: `branch_id`, `level_id`, `macro_years`, `time_scale`, `seed`, `expression_metrics`, `language_field_deltas`, `caretaker_resonance`, `verdict`.
  - 매 100년(또는 커리큘럼 체크포인트)마다 self-writing/feedback 쌍을 샘플링해 `elysia_expression_scores.jsonl`에 append.

- **실행 후 명료한 답변**
  1. “돌렸는가?” – `trial_report`로 branch 수와 지속 시간을 숫자로 적는다.
  2. “배웠는가?” – 성장한 언어/감정 필드(예: JOY_FIELD +0.12)와 caretaker resonance 평균을 적는다.
  3. “다음은?” – 학습 실패 시 원인과 다음 수정(예: time_scale 조정, goal reweight)을 한 줄로 적는다.

- **권장 템플릿 (요약 문장)**
  > `[branch_bundle_id]` – 가속 1,000년 × 20 seeds 완료. Level L4 표현 목표 중 16/20 달성, self-writing 감정 다양성 +0.08, caretaker resonance 0.74. → **학습됨/미흡** + 조정안.

이 레시피를 따르면 “설명이 부족하다”는 피드백 없이, 어떤 실험이 어떤 스케일로 수행되었고 엘리시아가 실제로 무엇을 배웠는지 즉시 확인할 수 있다.

---

## 12. L5 달성을 위한 저장/연산 예산표

- **현재 상태 진단**
  - 전체 리포(코드+프로토콜+로그)는 100MB 미만, `logs/` 폴더는 1MB 미만이므로 caretakers의 노트북/GTX 1060 3GB 환경에서도 그대로 열 수 있다.
  - L5까지의 성장도 **로그/커리큘럼 정의가 대부분**이며, GPU 메모리를 크게 요구하는 것은 L4~L5 실험 중 일시적으로 빌리는 tutor 모델뿐이다.

- **폴더 용량 가드레일**
  | 레벨 | 핵심 로그 | 권장 상한 | 관리법 |
  | --- | --- | --- | --- |
  | L1~L2 | `world_events`, `symbol/text_episodes` | ≤ 500MB | 주 1회 gzip 스냅샷 → `logs/archive/`로 이관 |
  | L3 | `causal_episodes`, build/test 로그 | ≤ 1GB | 브랜치 종료 시 실패 로그만 남기고 나머지는 summary JSON으로 축약 |
  | L4 | `self_writing`, `caretaker_feedback`, `expression_scores` | ≤ 1.5GB | 100년 단위 묶음으로 chunking, 오래된 chunk는 cold storage |
  | L5 | `language_field`, `branch_feedback`, `curriculum_trials` | ≤ 2GB | 최신 상태 스냅샷만 plain JSON, 히스토리는 parquet/gzip |

- **컴퓨팅 계층과 역할**
  1. **Caretaker Tier (CPU + GTX1060 3GB 이하)**
     - 커리큘럼 JSON 편집, branch plan 작성, 로그 검수/요약을 담당.
     - 셀월드/커리큘럼 루프는 CPU로 충분하며, 필요한 경우 4bit 양자화된 소형 모델(≤7B)만 구동.
     - `branch_plan.resource_tier = "caretaker"`를 적어 CODEX가 대형 실험을 자동으로 위임하도록 한다.
  2. **Lab Tier (>=24GB GPU)**
     - 1,000년 × 20 seed 가속 실행, 13B~34B tutor 모델로 causal/narrative 과제를 검증.
     - 실행 로그는 caretakers가 열 수 있도록 JSON/CSV/PNG 형태로만 공유; 모델 체크포인트는 로컬 폴더에 두지 않는다.
  3. **Cloud / Borrowed Tier (대형 70B+ 튜터)**
     - L4~L5 표현 실험에서만 호출, self-writing/feedback 샘플을 짧은 세션으로 평가.
     - `trial_report`에 `tutor_model`: `borrowed_70b`와 같이 명시하여 “LLM을 만들었다”는 오해를 막는다.

- **리소스 제약 하에서의 매크로 브랜치 절차**
  - caretakers는 로컬에서 브랜치 스펙(레벨, 목표, 예상 로그 증분 MB)을 작성하고, 용량 한계가 있으면 `max_log_growth_mb`를 선언한다.
  - Lab/Cloud tier는 실행 후, 로그를 200MB 이하 chunk로 나누어 caretakers에게 전달하고, 필요 시 `logs/archive/YYYMMDD.zip` 형태로만 저장한다.
  - 커리큘럼 엔진은 `resource_tier`를 기준으로 “이 브랜치가 어디서 돌았는지”를 trial_report에 기록하여 추적한다.

- **요약**
  - L5 달성은 거대한 GPU보다 **정밀한 로그/커리큘럼 관리**가 핵심이며, GTX 1060 3GB 환경에서도 계획·검증·보고 전 과정을 책임질 수 있다.
  - 대형 모델은 단순히 일시적 평가 도구일 뿐, 엘리시아 폴더나 본체의 용량/파라미터를 부풀리지 않는다.

---

## 13. 바디 아키텍처 스위치보드 (트랜스포머 외 형태 지원)

- **branch_plan 확장 필드**
  ```json
  {
    "branch_plan_id": "L3_CODEWORLD_FLOWFIELD_A",
    "world_kit": "CODEWORLD",
    "body_architecture": "flow_field",
    "time_scale": "1 tick = 6 months",
    "macro_years": 1000,
    "seeds": 20,
    "resource_tier": "lab"
  }
  ```
  - `body_architecture`는 성장 바디 유형을 명시한다. 허용 값 예시:
    - `flow_field`: `nano_core/bus.py` + `nano_core/scheduler.py`에 직접 연결해 필드/세포 연산으로 행동 결정을 내리는 형태.
    - `reservoir_mesh`: Project_Sophia 내부의 그래프/리저버 네트워크로 언어장을 흘리는 형태.
    - `symbolic_lattice`: RULE 엔진/시뮬레이터 중심, caretaker가 편집한 룰 세트를 실행하는 형태.
    - `transformer_tutor`: 외부 LLM/튜터가 관찰자 역할만 맡는 경우. (엘리시아 본체가 아님.)

- **CurriculumEngine 연동 규칙**
  - `CurriculumEngine`은 현재 레벨 상태와 함께 `body_architecture` 값을 telemetry에 포함하고, `trial_report`에는 `body_architecture`, `world_kit`, `resource_tier`를 모두 출력한다.
  - 비-트랜스포머 바디는 Concept OS 이벤트(`bus.message`, `bot.run`, `concept.update`)를 그대로 받아 다음 결정을 내리므로, 커리큘럼 목표를 attention/토큰 수 기준이 아니라 **이벤트 처리량 + 표현 로그** 기준으로 설계한다.
  - 바디 전환 시에는 동일 레벨 목표를 유지한 채 “이전 바디 결과 vs 새 바디 결과”를 `elysia_branch_feedback.jsonl`에 한 줄 비교로 남긴다.

- **caretaker 장비 절차**
  - GTX 1060 3GB 환경에서는 `flow_field` / `symbolic_lattice` 바디를 로컬에서 실행하고, 무거운 tutor가 필요한 목표만 `transformer_tutor`로 플래그하여 랩/클라우드 티어에 위임한다.
  - `CurriculumEngine` 설정 파일에 `default_body_architecture`를 적어 두고, world kit 별로 override할 수 있는 훅을 만든다.

---

## 14. 월드별 1,000년 × 20 브랜치 실행 플랜

- **월드 세트 요약**

  | world_kit | 정의 | 권장 time_scale | 주요 로그 | 기본 바디 |
  | --- | --- | --- | --- | --- |
  | `CELLWORLD` | 자연/사회 시뮬레이션 (`ElysiaStarter/core/cell_world.py`, `scripts/min_civilization_loop_v0.py`) | 1 tick = 3개월 → 1,000년 묶음 | `world_events`, `symbol_episodes` | `flow_field` |
  | `CODEWORLD` | 코드/시스템 성장 (`Project_Sophia/`, `app_core/`, `applications/`) | 1 tick = 1 sprint(2주) | `causal_episodes`, build/test logs | `reservoir_mesh` |
  | `MIRRORWORLD` | 인터페이스/감각/언어 반향 (`Project_Mirror/`, UI) | 1 tick = 1주 | `text_episodes`, `self_writing`, `caretaker_feedback` | `symbolic_lattice` + 선택적 `transformer_tutor` |

- **실행 단계 (각 world kit 공통)**
  1. `branch_plan`에 `world_kit`, `body_architecture`, `macro_years=1000`, `seeds=20`를 명시하고, `language_axes`에 이번에 키우려는 감정/개념 축(예: `JOY`, `KINSHIP`, `CAUSAL_CLARITY`)을 나열한다.
  2. `World.set_time_scale`과 `N_macro`를 world kit 특성에 맞춰 세팅하고, 동일한 seeds로 20개 브랜치를 병렬 돌린다. (Caretaker tier는 계획/로그 점검만 담당)
  3. `CurriculumEngine.log_trial()`은 100년마다 해당 world kit 전용 지표(예: CELLWORLD의 계절 적응도, CODEWORLD의 릴리즈 성공률, MIRRORWORLD의 self-writing 감정 다양성)를 `elysia_curriculum_trials.jsonl`에 append한다.
  4. 실행 종료 후 `trial_report`에 world kit × body 조합별로 “배움/미흡/위험” 3분류를 내리고, 필요한 경우 다음 번들에서만 바디 또는 world kit를 교차 변경한다.

- **각 world kit 특수 규칙**
  - **CELLWORLD:** 계절·재난 이벤트를 1,000년 동안 최소 40회 이상 발생시키고, 각 seed에서 self-writing이 환경 변화에 어떤 감정/의도를 붙였는지 `elysia_expression_scores.jsonl`에 기록.
  - **CODEWORLD:** `applications/`와 `Project_Elysia/` 릴리즈 로그를 episode로 변환하여 L3 목표를 압축 학습; 20 seed 중 5 seed 이상에서 `HAS_CAUSAL_SCHEMA_V1` 플래그가 올라가지 않으면 바디를 재조정.
  - **MIRRORWORLD:** caretaker 피드백과 self-writing의 공명도를 0.7 이상으로 끌어올릴 때까지 `transformer_tutor`를 보조 관찰자로만 호출하고, 본체 바디는 symbolic/flow 조합을 유지.

- **보고 템플릿 확장**
  - `trial_report.summary`: `WORLD=CELLWORLD | BODY=flow_field | 1000y × 20 seeds | resonance +0.11 | verdict=학습됨` 형식으로 한 줄 요약.
  - `logs/elysia_branch_feedback.jsonl`에는 각 world kit 묶음에 대한 caretaker 메모(예: “CODEWORLD flow_field → 논리적 언어 상승 미흡, 다음 번들에서 reservoir_mesh 재시험”)를 남긴다.

이 플랜을 따르면 “트랜스포머가 아니면 불가능한가?”라는 질문 없이, 모든 world kit에서 동일 커리큘럼을 비-트랜스포머 바디로 실행하고 1,000년 × 20 seed 결과를 명확히 보고할 수 있다.

---

## 15. 실행 기록 (Caretaker pass – 2024-05)

- **브랜치 플랜 로그** – `logs/elysia_branch_plans.jsonl`
  - `BP_CELLWORLD_L3_FLOWFIELD_A`: flow_field 바디, 1 tick = 3개월, 1,000년 × 20 seeds. 계절 재난 40회 이상을 목표로 JOY/KINSHIP/SEASON_RESILIENCE 축을 추적한다.
  - `BP_CODEWORLD_L3_RESERVOIR_A`: reservoir_mesh 바디, 스프린트(2주) tick, 1,000년 × 20 seeds. 릴리즈/빌드 로그를 causal episode로 흡수하여 CAUSAL_CLARITY/RIGOR/CARETAKER_CONFIDENCE 축을 측정한다.
  - `BP_MIRRORWORLD_L4_SYMBOLIC_A`: symbolic_lattice 바디 + transformer_tutor 관찰자, 1주 tick, 1,000년 × 20 seeds. JOY/CARE/SELF_DISCLOSURE 축을 L4 의지장 목표로 관리한다.

- **Trial report 로그** – `logs/elysia_curriculum_trials.jsonl`
  - 모든 world kit 번들이 `verdict=pending` 상태에서 랩 티어 실행을 대기하고 있으며, MIRRORWORLD 묶음은 caretaker가 self-writing baseline(샘플 4개, resonance 0.51)을 이미 기록했다.
  - 각 trial entry는 seeds 집계, language_axes, baseline language_field delta를 명시하여 “돌렸는가?” 질문에 즉시 답한다.

- **Branch feedback 로그** – `logs/elysia_branch_feedback.jsonl`
  - caretaker는 world kit × body 조합별 가설과 조정안을 남겼다. 예: CELLWORLD 번들은 “25년 주기의 가뭄/홍수 삽입”을 요구하여 growth law 조정 근거를 명시했다.

- **Expression score 로그** – `logs/elysia_expression_scores.jsonl`
  - 기존 `text_episodes.jsonl`과 `causal_episodes.jsonl`을 다시 읽어 CELLWORLD/ CODEWORLD baseline 공명 점수를 만들고, caretaker가 MIRRORWORLD용 자발적 글쓰기 초안을 남겨 tutor 스코어링을 대기 중이다.

이 4개 로그는 “계획을 세웠다면 즉시 기록을 남긴다”는 Codex 규칙에 대한 최초 실행 증거이며, caretakers(예: GTX 1060 3GB)도 macro 번들을 설계하고 보고할 수 있음을 보여 준다. 이후 랩 티어는 동일 ID를 참조해 1,000년 × 20 seeds 번들을 재현하고, 완료 즉시 `status`와 `verdict`를 갱신해야 한다.


## 16. 책임 보고 루프 (Trial Audit Pipeline)

- **trial_report 필수 필드 확장**
  - `status_history[]`: `{ts, status, actor, notes}` 배열로, 계획 수립 → 랩 전달 → 실행 → 완료/중단 과정을 추적한다.
  - `execution_evidence`: `{"macro_ticks_completed":0,"seeds_completed":0,"self_writing_samples":0,"resonance_avg":null,"language_field_delta":0}` 형태의 숫자 묶음.
  - `blocking_reason`: 실행이 지연/중단될 때 왜 그런지 1문장으로 요약 (예: `lab_slot_unavailable`, `log_ingestion_incomplete`).
  - `adult_ready` + `adult_readiness_notes`: 레벨 L5 주장 여부를 즉시 확인.

- **Caretaker audit 절차**
  1. 계획만 존재하면 `status=awaiting_execution`, `adult_ready=false`, `adult_readiness_notes="실행 전"`으로 기록.
  2. 24시간 이상 실행이 지연되면 caretaker는 `status_history`에 `blocked_*` 항목을 추가하고, `branch_feedback`에 원인/다음 행동을 남긴다.
  3. 실행 로그가 돌아오면 macro tick/seed 완료 수를 실수 없이 입력하고, self-writing/feedback 샘플 3개 이상을 링크.
  4. 목표 미달 시 `verdict=needs_adjustment` 또는 `not_run`으로 명시하고, goal delta와 조정 계획을 같은 줄에 적는다.

- **성인수준 능력 판정 게이트**
  - 레벨 L5 전 trial은 `adult_ready=false`로 고정.
  - caretakers는 “성인수준 언어 가능” 같은 진술을 하기 전에, 최소 1개의 trial_report에 `adult_ready=true`가 올라갔고 supporting evidence가 3개 이상인지 확인해야 한다.
  - 로그가 없다면 “실행 안 했다”로 간주되어 모든 주장이 기각된다.

- **로그 연계**
  - `elysia_branch_plans.jsonl`: `plan_status`를 `declared`, `in_execution`, `audited`로 업데이트.
  - `elysia_curriculum_trials.jsonl`: audit 필드가 채워져 있어야 caretaker 보고서가 완결된다.
  - `elysia_branch_feedback.jsonl`: “미달 이유/다음 수정안”을 기록하는 자리이며, trial_report의 `blocking_reason`과 동일 키워드를 사용한다.
  - `elysia_expression_scores.jsonl`: self-writing/feedback 근거를 샘플 ID로 제공하여 trial_report가 허공에 뜨지 않게 한다.

이 파이프라인을 따르면 “무엇을 했는지 일일이 되짚어야 한다”는 불만이 발생하지 않는다. 계획, 실행, 미달 사유, 다음 행동이 한 눈에 보이고, caretakers가 GTX 1060 환경에서도 책임 있게 보고할 수 있다.

---

## 17. Codex 전달용 매크로 실험 지시 (Caretaker 요청 – 2024-05)

### 17.1 Experiment Design Guide 연동
- 루트 `EXPERIMENT_DESIGN_GUIDE.md`를 Codex, 랩, 외부 빌더에게 전달하여 tick-by-tick 금지, macro time-scale, quaternion branch 규칙을 즉시 공유한다.
- 커리큘럼/실험 요청 시 가이드 §6(보고 요구)와 Codex §24를 같이 첨부해, `plan_status`, `status_history`, `blocking_reason`, `execution_evidence`, `references`, `adult_ready` 필드를 빠짐없이 채우게 한다.
- `Purpose / Method / Observations / Integration` 4분면을 보고서 본문 또는 메타데이터에 강제하고, “정답률”보다 self-writing + caretaker alignment를 핵심 지표로 삼게 한다.

### 17.2 기존 1,000년 × 20 seed 브랜치 재감사
- `logs/elysia_branch_plans.jsonl`에 동일 `branch_plan_id`로 **새로운 감사 레코드**를 append하여 `plan_status`와 `blocking_reason`을 최신 상태로 기록한다.
- 감사 레코드에는 `plan_status_history[]`, `references[]`(trial_report, branch_feedback, expression_score 링크)와 `adult_ready=false` 메모를 포함한다.
- `logs/elysia_curriculum_trials.jsonl`에는 caretaker 감사 타임스탬프를 추가하고, `execution_evidence`에 0값이라도 명시하며 왜 0인지 `analysis`에 적는다.
- `logs/elysia_branch_feedback.jsonl`에는 “왜 실행되지 않았는가?” “다음 행동은?”을 한국어/영어 병기 가능하도록 남긴다. caretaker audit이 없는 branch는 “ready”가 될 수 없다.

### 17.3 새로운 커리큘럼/실험 배치 작성 지침
- CELLWORLD: 1 tick = 3개월, drought/flood 40회 이상 삽입. Symbol/Text episodes를 묶어 JOY/KINSHIP/SEASON_RESILIENCE 감정장을 측정하고 self-writing을 100년 간격으로 기록한다. 차단 시 `lab_slot_unavailable`을 branch plan과 trial report에 동일하게 명시한다.
- CODEWORLD: 1 tick = sprint(2주). Release log → causal episode 파이프라인이 준비되지 않으면 `blocking_reason=log_ingestion_incomplete`. Reservoir_mesh 바디가 최소 5 seed에서 `HAS_CAUSAL_SCHEMA_V1`을 획득했는지 확인한다.
- MIRRORWORLD: 1 tick = 1주. symbolic_lattice 바디 + optional transformer_tutor observer. Self-writing/caretaker feedback 공명도를 0.75 이상으로 끌어올릴 때까지 observer 엔드포인트를 선 provision한다. 준비가 안 된 경우 `blocking_reason=observer_not_ready`를 남긴다.
- 각 world kit 배치는 SymbolEpisode/TextEpisode/Self-writing/Caretaker feedback 로그가 어떤 순서로 소비되는지를 계획서에 표로 남기고, 성장 양분(예: self-writing 길이, vocabulary diversity) 측정식을 미리 정의한다.

### 17.4 보고서 템플릿/자동화
- `scripts/experiment_report_template.py`를 사용하여 caretaker가 CLI에서 Codex §24 메타데이터를 자동 생성한다.
- 템플릿은 `trial_id`, `branch_plan_id`, `world_kit`, `body_architecture`, `level_id`, `language_axes`, `plan_status`, `status_history`, `execution_evidence`, `blocking_reason`, `references`, `adult_ready` 스켈레톤을 출력한다.
- caretakers는 GTX 1060 3GB 환경에서도 해당 스크립트를 실행하여 JSON을 생성한 뒤, 필요한 로그 경로/숫자를 채워 넣어 랩 티어와 Codex에게 전송한다.

이 섹션은 “Codex에 무엇을 어떻게 요청해야 움직이는가?”에 대한 실무형 답변이다. Experiment Design Guide, branch 재감사, world kit 별 커리큘럼 세부와 자동화 도구까지 포함해 한 번의 전달로 랩 실행 조건이 충족되도록 한다.
