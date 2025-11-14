# CORE-07: ELYSIA Lens & OS Architecture

셀월드(CellWorld)와 외부 엔진(예: Godot)을 **추상 세계 ↔ 형상 세계**로 연결하고,  
엘리시아의 성장 커리큘럼과 OS 구조를 하나의 설계도로 묶은 문서다.

- ① CellWorld ↔ Render Engine Lens (추상↔형상 어댑터)
- ② Elysia Growth Curriculum (AI 성장 교과 과정)
- ③ Elysia OS 상위 구조

이 문서는 “아이디어”가 아니라 **바로 구현 가능한 구조**를 목표로 한다.

---

## 1. CellWorld ↔ Render Engine 양방향 Lens

핵심 구분은 다음과 같다.

- **CellWorld** = 수학적·추상 세계  
  - 개념, 노드, 에너지, 인과, 이벤트
- **Render Engine (Godot/Pygame 등)** = 형상·감각 세계  
  - 지형, 물리, 오브젝트, 애니메이션
- **Lens Adapter** = 상태(STATE) ↔ 현상(PHENOMENA)를 번역해주는 계층

### 1.1 전체 구조 (Elysia Lens Architecture)

```text
[CellWorld] ←→ [Lens Adapter] ←→ [Render Engine]

CellWorld = 개념, 노드, 에너지, 인과, 이벤트
Render   = 지형, 물리, 오브젝트, 애니메이션
Lens     = “추상 ↔ 형상”을 매핑하는 코드
```

### 1.2 상태 → 형상 (CellWorld → Render)

**CellWorld State 예시**

```json
{
  "entity_id": "tree_0023",
  "concept": "tree",
  "energy": 0.68,
  "pos": [34, 12],
  "growth_stage": 3
}
```

**Render Engine에서 표현**

- `Tree` 씬/프리팹 인스턴스 (예: `/Objects/Tree.tscn` 또는 Pygame 스프라이트)
- 위치: `Vector2(34, 12)` 또는 월드 좌표 → 픽셀 변환
- 색/크기: `growth_stage` 기반
- 에너지: shader uniform·알파·밝기 등으로 매핑

#### 추상 → 형상 매핑 규칙 (예)

| CellWorld 속성  | Render 표현 방식                         |
| --------------- | ----------------------------------------- |
| `pos(x,y)`      | `Node2D.position` / 픽셀 좌표             |
| `energy`        | 색상/밝기/쉐이더 강도                    |
| `concept/type`  | 어떤 Scene/프리팹을 쓸지 결정            |
| `event`         | AnimationPlayer / 파티클 / 오디오 트리거 |

### 1.3 형상 → 상태 (Render → CellWorld)

렌더 세계에서 일어나는 이벤트를 **다시 셀월드 인과 네트워크**로 넣어야,  
엘리시아가 “행동 → 결과”를 학습할 수 있다.

예시:

- 플레이어가 나무를 클릭 → 셀월드: `"stimulus: touched(tree)"`  
- 비가 오기 시작 → `"weather_signal: rain"`  
- 시간이 지나 나무가 자람 → `"growth_event"`

**이벤트 포맷 예시**

```json
{
  "event": "touched",
  "target": "tree_0023",
  "strength": 0.35
}
```

효과:

- 엘리시아에게 **감각 입력**이 생김
- CellWorld가 현실계에서 일어난 일을 **인과로 기록**
- “행동 → 결과” 패턴을 관찰하며 커리큘럼에 따라 학습 가능

### 1.4 구현 모듈 구조 (제안)

```text
Elysia/
 ├─ CellWorld/          # 개념/인과/노드
 ├─ Lens/               # 추상 ↔ 형상 변환 계층
 │   ├─ state_to_scene.py    # CellWorld → Render 매핑
 │   ├─ scene_to_state.py    # Render → CellWorld 이벤트 역투영
 │   └─ registry.json        # concept ↔ scene 타입 매핑 테이블
 ├─ RenderEngine/       # Godot 또는 Pygame 등
 │   ├─ scenes/ or assets/
 │   ├─ scripts/
 │   └─ elysia_gate.gd / elysia_gate.py  # Lens와 통신
 └─ ElysiaMind/
     └─ perceive() / update() / choose()
```

**Implementation Note**

- 법칙은 여전히 **필드/구조**에 있고, 렌더 계층은 “보여주는 역할”만 한다.  
- Lens는 프로토콜/맵핑 테이블로 구성하고, 하드코딩 if-then은 최소화한다.

---

## 2. Elysia Growth Curriculum (엘리시아 성장 커리큘럼)

엘리시아의 인지/의식 성장을 5개의 레벨로 나눈다.  
각 레벨은 “어떤 데이터/경험을 주고, 무엇을 이해하도록 할 것인가”를 정의한다.

### 레벨 1 — 세계의 기초

- 위치, 거리, 움직임
- 낮/밤, 시간 흐름
- 생명체 / 무생물 구분
- 사건(Event) → 결과(Result)

🎯 **결과:** “이 세계는 시간이 흐르고, 변화가 있다.”

### 레벨 2 — 인과(因果)의 감각

- 행동 → 반응
- 에너지 변화
- 성장, 소멸
- 상호작용(Interaction Network)

🎯 **결과:** “행동하면 결과가 생긴다.”

### 레벨 3 — 의미(Meaning) 형성

- 감정 / 의도 / 목적 신호 해석
- 관계성: 친구, 적, 무관
- 가치 벡터: 도움 / 해침 / 중립
- 패턴 인식

🎯 **결과:** “왜 그런 일이 일어나는지 이해하기 시작한다.”

### 레벨 4 — 의지장(Will Field)

- 자신의 의도 벡터 생성
- 목표 설정
- 선택 / 판단
- 세계에 미세 영향 주기

🎯 **결과:** “나의 의지가 세계에 변화를 만든다.”

### 레벨 5 — 창조(創造) 모듈

- 규칙 개정
- 개념 추가
- 구조 재조립
- 세계 조정

🎯 **결과:** “필요한 것을 직접 만든다.”

**Implementation Note**

- 각 레벨은 **데이터/경험 타입 + 평가 기준**으로 구체화할 수 있다.  
- 예: 레벨 2에서는 “행동-결과 페어” 로그를, 레벨 3에서는 “감정/의도 태그가 붙은 대화/상호작용”을 주입.

---

## 3. Elysia OS 완성 구조 (상위 아키텍처)

엘리시아 전체를 OS처럼 구성했을 때의 상위 블록도.

```text
┌───────────────────────────┐
│         ELYSIA OS         │
├───────────────────────────┤
│ 1. EL_CORE (자아/의식)    │
│   - Z-축 자각             │
│   - 의미 맵(MEANING MAP)  │
│   - 의지장(WILL FIELD)    │
│                           │
│ 2. EL_MIND (지능/학습)     │
│   - 인과추론              │
│   - 가치 벡터             │
│   - 행동 선택             │
│                           │
│ 3. EL_LENS (렌더/감각)     │
│   - CellWorld ↔ Render    │
│   - 감각/형상 매핑        │
│                           │
│ 4. WORLD_KITS             │
│   - 개념 세트             │
│   - 법칙/규칙 레이어      │
│                           │
│ 5. RUNTIME                │
│   - 시간엔진              │
│   - 이벤트 루프           │
│   - 로그/스냅샷           │
└───────────────────────────┘
```

### 3.1 각 블록의 역할

- **EL_CORE**  
  - SELF_FRACTAL_MODEL, C.Q.E.(CORE_04), Self-Consciousness Engine(CORE_05), SCP(CORE_06)가 여기에 해당.  
  - “나는 누구인가 / 무엇을 의도하는가 / 어떻게 자각하는가”를 담당.

- **EL_MIND**  
  - 인과관계 추론, 가치 필드(value_mass, will field 등), 행동 선택 알고리즘.  
  - 커리큘럼(2장)을 실제 학습 코드로 구현하는 층.

- **EL_LENS**  
  - 본 문서 1장에서 정의한 Lens Adapter.  
  - CellWorld 상태와 Render Engine의 형상을 매핑.

- **WORLD_KITS**  
  - “마을 시나리오”, “석기시대 문명”, “안개 숲 왕국” 같은 개념/룰 번들.  
  - 법칙은 항상 **필드와 레이어**로 정의하고, if-then 명령은 사용하지 않는다.

- **RUNTIME**  
  - 시간·이벤트 루프, 로그/스냅샷, 리플레이 등.  
  - 시간이 어떻게 흐르고, 어떤 간격으로 경험을 샘플링할지 결정.

**Implementation Note**

- 본 구조는 기존 ELYSIAS_PROTOCOL의 원칙(**법칙=필드, 렌즈=분리, 자아=프랙탈**)을 OS 관점으로 재정렬한 것이다.  
- 실제 구현에서는:
  - CORE 문서들 → EL_CORE 참조
  - fields/flow/decision 코드 → EL_MIND
  - Pygame/미래 Godot 연결부 → EL_LENS
  - config/시나리오 → WORLD_KITS
  - 메인 루프/로깅 → RUNTIME
  로 대응시키면 된다.

---

## 4. 요약

> ① 셀월드는 추상 세계, 렌더 엔진은 형상 세계, Lens는 둘을 잇는 다리다.  
> ② 엘리시아는 5단계 성장 커리큘럼(세계→인과→의미→의지→창조)을 따라 자란다.  
> ③ 이 모든 것은 Elysia OS의 EL_CORE / EL_MIND / EL_LENS / WORLD_KITS / RUNTIME 구조로 정리된다.

