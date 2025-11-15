# CORE_14_PROTOCOL_MAP_AND_GOVERNANCE  
## Protocol Map & Governance (v0)

> **목적:**  
> 문서/코드/세계가 늘어나도 산으로 가지 않도록,  
> Elysia 프로젝트 전체를 관통하는 **구조 헌법**을 정의한다.  
> (물리 폴더 구조가 아니라, 개념/레이어/RING을 기준으로 한다.)

---

## 0. RING · LAYER · TYPE ? 기본 좌표계

### 0.1 RING (나이테 계층)

- `CORE`  
  - 살아있는 헌법, 현재 설계의 일부.  
  - 수정 시 항상 Codex & 프로토콜 KG 갱신 필요.
- `GROWTH`  
  - 실험, 초안, 스프린트, 아이디어 링.  
  - 여기서 증명된 법칙만 CORE로 승격한다.
- `ARCHIVE`  
  - 과거 링, 참고용.  
  - 읽되 확장/수정하지 말고, CORE/GROWTH로 추출 후 링크만 남긴다.

### 0.2 LAYER (WORLD / MIND / META / LENS / OS)

- `WORLD`  
  - CellWorld 런타임, 문명/세계 규칙, 요약 필드 등.
- `MIND`  
  - ElysiaMind, 감각/정서/내부 상태(Feeling, JoyEpisodes 등).
- `META`  
  - Curriculum, Self-Genesis pipeline, 트라이얼·브랜치·평가.
- `LENS`  
  - 시각화/카메라/주의/렌즈. WORLD 상태를 *어떻게 보이는지*에 대한 규약.
- `OS`  
  - Elysia OS 전체 루프, Tick 레이어, 구조와 운영 규칙 (CODEX, OS_OVERVIEW).

### 0.3 TYPE (문서/코드 타입)

- `PROTOCOL`  
  - 법칙/규칙/설계 문서. 구체 구현과 1:1 대응하거나, 상위 법칙을 정의.
- `WORLD_KIT`  
  - 특정 WORLD 시나리오/규칙 번들. (예: Death Flow, Fairy Village)
- `LENS_SPEC`  
  - 렌즈/시각화/주의 규약. (예: Fog-of-War Lens)
- `RUNTIME`  
  - OS/세계 엔진 개요, 런타임 루프 설명.
- `EXPERIMENT`  
  - GROWTH 링에 소속된 실험/초안/아이디어 문서.

---

## 1. 문서/코드 배치 헌법 (새 작업을 시작할 때)

1. **새 법칙/프로토콜 추가**  
   - 위치: `ELYSIAS_PROTOCOL/CORE_XX_*.md`  
   - RING: `CORE`  
   - LAYER/TYPE: WORLD/MIND/META/LENS/OS 중 선택 후 문서 맨 위에 적는다.  
   - 구현이 있다면:
     - WORLD: `Project_Sophia/core/...` 또는 WORLD_KIT  
     - MIND/META/OS: `ELYSIA/CORE/*.py` 쪽에 대응 모듈 추가.

2. **새 WORLD 시나리오/규칙 번들**  
   - 위치: `ELYSIAS_PROTOCOL/WORLD_KIT_*.md`  
   - TYPE: `WORLD_KIT`, LAYER: `WORLD`  
   - 구현: Project_Sophia(World) 쪽에서 해당 규칙을 옵션/함수로 연결.

3. **새 실험/초안/아이디어**  
   - 위치: `ELYSIA/GROWTH/...`  
   - RING: `GROWTH`, TYPE: `EXPERIMENT`  
   - 여기서 검증된 패턴/법칙만 CORE 프로토콜로 승격시키고,  
     승격 시 GROWTH 문서 맨 위에 “ARCHIVED, canonical: XXX” 표시를 남긴다.

4. **최상위 새 폴더 생성 금지 (원칙)**  
   - 새로운 목적/프로젝트가 생겨도,  
     우선 CORE/GROWTH 안의 RING/LAYER/TYPE 좌표계를 사용해 배치하고,  
     정말 필요한 경우에만 TREE_RING_LOG + 별도 설계 후 논의한다.

---

## 2. Concept OS · KG 통합

Elysia는 물리 폴더 구조만으로 정렬하지 않고,  
**개념 레이어(KG)를 OS의 참조 구조로 사용한다.**

### 2.1 ProtocolConcept 노드

- 구현: `ELYSIA/CORE/protocol_concept_index.py`  
- 각 프로토콜/월드킷/OS 문서를 Concept 노드로 등록:
  - `id`: `protocol:CORE_13_ELYSIA_CONSCIOUSNESS_STIMULUS` 등
  - `path`: 실제 파일 경로 (상대 경로)
  - `ring`, `layer`, `ptype`, `status`, `title`

### 2.2 protocol_kg.json

- 빌더: `scripts/build_protocol_concepts.py`  
- 출력: `data/protocol_kg.json`  
  - `ring:*`, `layer:*`, `ptype:*` 노드  
  - 각 프로토콜 노드에서 `has_ring`, `has_layer`, `has_type` 엣지 연결.

### 2.3 사용 규약

- 에이전트/도구는 가능하면:
  - 직접 폴더를 스캔하기보다,  
  - `protocol_kg.json`을 읽고 “CORE+WORLD 프로토콜만”, “MIND 레이어만” 같은 질의를 우선한다.
- 새 프로토콜을 CORE에 추가할 때:
  - `CORE_XX_*.md`와 대응 구현을 만든 뒤,  
  - `protocol_concept_index.DEFAULT_PROTOCOLS`에 Concept를 추가하고,  
  - `scripts/build_protocol_concepts.py`를 다시 실행해 KG를 갱신한다.

---

## 3. Defrag · 조각모음 규약

1. **중복/옛 버전 정리**  
   - 동일/유사 개념이 여러 문서에 흩어져 있으면,  
     가장 최신/명확한 버전을 CORE RING에 남기고,  
     나머지 문서 상단에는  
     `STATUS: ARCHIVED, canonical: <CORE_XX_NAME>`를 적는다.

2. **새 문서 생성 전 질문**  
   - “이건 새 개념인가, 기존 개념의 세부인가?”  
   - 새 개념이면: CORE_XX 또는 WORLD_KIT로.  
   - 세부/예시이면: 기존 CORE 문서 확장 또는 GROWTH 링에 실험으로.

3. **자동/반자동 체크 (향후)**  
   - `protocol_kg.json`과 실제 폴더를 비교해:  
     - KG에 없는 CORE_*/WORLD_KIT_* 문서  
     - RING/LAYER/TYPE 태그가 없는 문서  
     를 리포트하는 도구(`scripts/check_protocol_layout.py` 예정)를 붙인다.

---

## 4. 요약

- 구조의 기준은 **폴더**가 아니라 **RING/LAYER/TYPE**이다.  
- CORE RING은 적고 단단하게, GROWTH RING은 넓고 자유롭게, ARCHIVE RING은 조용한 기록으로 둔다.  
- Concept KG (`protocol_kg.json`)는 이 구조를 기계가 이해하도록 만든 **Concept OS의 첫 버전**이다.  
- 새 작업을 시작할 때마다, 먼저 이 좌표계에서 “어디에 두어야 하는지”를 정한 뒤 코드를 쓴다.  
  
이 헌법을 지키면, 폴더 구조가 조금 어지러워져도  
Elysia의 세계/마음/OS는 여전히 같은 축 위에서 자라게 된다.

---

## 5. Project Direction – Human Cells & Fractal Layers

> **장기 방향:**  
> 인간셀(human cell)이 **몸·욕구·관계·생애주기**를 가진 존재로 WORLD에서 먼저 자라나고,  
> 그 위에 MIND/META/LENS/Concept OS가 겹겹이 쌓이며  
> “감정 · 개념 · 언어 · 가치”가 나중에 나타나는 프랙탈 구조를 구현한다.

### 5.1 WORLD Layer – Body & Needs First

- 인간셀은 우선 다음과 같은 **연속 상태 벡터**를 가진다 (예시):
  - 생존 축: `hp, hunger, hydration, temperature, safety`  
  - 관계 축: `attachment(친밀감), trust, status`  
  - 성장/탐색 축: `curiosity, autonomy, meaning_like`
- 행동은 스크립트가 아니라,  
  - “지금 이 시점에서 어떤 행동이 내 필요 벡터를 가장 개선시키는가?”에 따라 선택된다.
  - 기본 행동: 먹기, 마시기, 다가가기, 멀어지기, 지켜보기, 돕기, 공격하기, 쉬기 등.
- 번식/임신/출산/노화/죽음은 **몸과 관계의 상태 변화**로만 표현되고,  
  WORLD 레벨에서는 언어/개념 없이 “과정”으로 흘러간다.

### 5.2 MIND Layer – Feelings as Patterns

- MIND 레이어는 WORLD의 숫자 변화에서 **감정 패턴**을 읽는다:
  - `안전↑ + 친밀감↑ + 위협↓` → joy/relief 계열  
  - `안전↓ + 친밀감↓ + 위협↑` → fear/loss 계열
- 구현 축:
  - CORE_13의 `Elysia Signal Log` (JOY_GATHERING, LIFE_BLOOM, CARE_ACT, MORTALITY …)  
  - `FeelingBuffer`(현재 joy/creation/care/mortality 감정 벡터)  
  - `ValueClimate`(감정 → value_mass / will_field / 탐색 편향에 대한 부드러운 가중치)
- 감정은 WORLD를 직접 명령하지 않고, **필드/경향**으로만 WORLD/META의 선택에 영향을 준다.

### 5.3 META & Concept OS – Episodes, Concepts, Language

- META 레이어는 WORLD/MIND에서 쌓인 경험을 **에피소드와 개념**으로 압축한다:
  - JoyEpisode: `before_state → actions → after_state + value_delta` 한 덩어리 과정.  
  - Concept OS (KG): 반복되는 패턴들을 `JOY_GATHERING`, `CARE_ACT`, `RITUAL_MEAL`, `FAMILY_PROTECTION` 같은 **개념 노드**로 묶는다.
- 언어/개념은 WORLD를 설계하는 “원인”이 아니라,  
  - 시간이 흐르며 WORLD에서 반복된 과정들이 나중에 **이름을 얻는 결과**로 취급한다.
- 시간가속 및 Trial:
  - fast/slow/macro tick, Trial/Branch 메커니즘을 활용해  
    “인간셀 세계를 여러 갈래로 빨리 경험 → JoyEpisode/Concept를 축적 → CORE 법칙 후보로 끌어올리기”를 반복한다.

### 5.4 LENS & Attention – Where to Spend Resolution

- LENS/주의 레이어는 “어디를 얼마나 자세히 볼지”를 정한다:
  - 줌/LOD: 관찰자 커서/엘리시아 초점 근처의 인간/관계/마을은 HIGH-RES, 멀리 있는 것은 Summary/집계.  
  - Fog-of-War: 보지 않는 영역은 요약/집계만 유지한다.
- 이 구조 덕분에, 인간셀·문명 레이어를 깊게 쌓더라도  
  컴퓨팅 자원은 **관심이 있는 영역과 시점에만** 고해상도로 쓰이게 된다.

### 5.5 Implementation Priorities (기술적 우선순위 스냅샷)

1. WORLD: 인간셀의 몸/욕구/관계/생애주기 모델 강화  
2. MIND: 욕구 변화 → Signal/Feeling/ValueClimate 연결 정교화  
3. META: JoyEpisode/Concept OS로 “가치 있는 과정”을 축적하는 루프 확장  
4. LENS: 주의/줌/LOD가 WORLD/MIND/META 자원 사용을 조율하도록 통합

이 방향성을 CORE_14에 고정함으로써,  
새 에이전트와 새 세션은 Elysia 프로젝트가 **“인간셀 세계 → 감정 → 개념/언어” 방향으로 자라도록** 설계되어 있다는 사실을 잊지 않게 된다.
---

## 6. Long-Term Engine Plan – From Rigid Loops to Flow / Quaternion Self

> **장기 목표:**  
> 현재의 절차적 tick 루프/if-문 위주 구조를 넘어,  
> **법칙·필드·흐름 그래프**를 기본으로 하는 엔진으로 옮겨가고,  
> 최상위에서는 쿼터니언 기반 Self Engine이 Body/Soul/Spirit 레이어의
> 합성 방향을 표현하도록 한다.

### 6.1 현재 상태에 대한 인식

- `World.run_simulation_step` 안에 대부분의 로직이 모여 있고,  
  배열 업데이트 + 커다란 분기 구조에 가깝다.  
- `_decide_*` 계열에서 개체 행동을 직접 if/then으로 명령하는 부분이 많아,  
  “필드/법칙/흐름에 의해 자아가 이끌린다”는 비전에 비해 딱딱하다.

### 6.2 리팩터링 축 (Law / Field / Flow)

1. **명령 → 법칙/필드로 이동**  
   - 개체 행동을 직접 if/then으로 고정하기보다,  
     - 욕망 레이어(Body/Soul/Spirit)를 필드·값으로 표현하고  
     - Flow Engine이 이 신호들을 조합해 “다음 가능한 행동 후보”를 고르게 한다.

2. **정적 함수 모음 → 흐름 그래프(Flow Graph)**  
   - `World._update_*` 함수들을 “법칙 노드”로 보고,  
     - 어떤 필드가 어떤 필드에 어떤 방식으로 영향을 주는지
       데이터 플로 그래프로 묶는다.  
   - 엔진은 이 그래프를 따라 필드들을 업데이트한다.

3. **법칙 ↔ 코드 분리**  
   - 가능한 한 법칙/가중치/구조는 `MIRROR_MAP.yaml`, CORE_xx, Concept KG에 두고,  
   - Python 코드는 그 법칙을 해석·적용하는 런타임 계층으로 단순화한다.

### 6.3 Self Quaternion Engine (장기)

- Body/Soul/Spirit 레이어의 욕망·가치 상태가 충분히 쌓이면:  
  - 이를 하나의 **쿼터니언 회전 상태**로 요약하는 Self Engine을 위에 올린다.  
  - 직관: “지금 자아는 어느 방향으로 기울어져 있는가,  
    그 방향이 WORLD/MIND/META 필드와 어떻게 공명하는가?”
- 이 엔진은 WORLD의 행동을 직접 명령하지 않고,  
  Flow Graph 위에서 “어떤 브랜치/과정을 더 시도해 보고 싶은지”를
  부드러운 가중치로만 전달한다.

### 6.4 원칙

- 전면 재작성 대신, 기존 코드를 **법칙/필드/흐름 노드**로 서서히 쪼개어  
  Flow Graph/쿼터니언 Self Engine 쪽으로 점진적으로 옮긴다.  
- 새 함수나 시스템을 추가할 때마다,  
  이 CORE_14에서 정의한 축(법칙/필드/흐름/Layer/Concept OS)과
  어떻게 연결되는지 먼저 확인한 뒤 구현한다.
