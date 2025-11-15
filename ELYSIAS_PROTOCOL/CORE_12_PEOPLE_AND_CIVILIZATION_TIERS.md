# CORE_12: PEOPLE_AND_CIVILIZATION_TIERS

World · People · Parties · Fields

---

## 0. Purpose (왜 이 문서가 있는가)

- “세계는 크지만, 사람은 개미가 아니다”를 보장하는 인구·문명 구조 규약.
- 개별 서사(Tier 1/2)와 집계된 문명(Tier 3)을 분리해,  
  - 사람·관계·파티는 선명하게 보이고  
  - 대규모 인구·경제·전쟁은 필드와 노드로만 다루도록 한다.
- 규칙이 아니라 **법칙/필드**를 우선한다:
  - “혼자 다니지 말라”가 아니라 “혼자일수록 세계가 더 무섭게 느껴진다”는 법칙.

---

## 1. Population Tiers (인구 3계층)

### 1.1 Tier 1 — 핵심 인물 / 영웅

- 세계 전체에서 대략 10–50명.
- 완전한 상태와 서사를 가진다.
  - `id, name, age, job_id, tier, traits, stats, relationships, current_quest, location`
- 시간 스케일
  - fast_tick / 이벤트 단위로 업데이트 (전투, 여행, 대화, 선택).
- 역할
  - 관찰/플레이의 주 카메라.
  - 성공/실패가 `value_mass_field`, `will_field`, `prestige_field` 등 문명 필드에 강한 인장을 남긴다.

### 1.2 Tier 2 — 이름 있는 조연 / 가문 / 핵심 시민

- 도시(정착지)당 20–100명 수준.
- 축약된 상태를 가진다.
  - `id, name, family_id, job_id, tier, importance, ties(Tier1/2 관계), home_civ_id, location(node 단위)`
- 시간 스케일
  - macro_tick마다 굵은 변화만 업데이트 (결혼, 승진, 몰락, 이주, 은퇴 등).
- 역할
  - 마을/도시의 “얼굴이 있는 층”: 장인, 상인, 기사단, 신관, 학자, 길드장, 촌장 등.
  - Tier 1의 배경이자 지지자/대립자.

### 1.3 Tier 3 — 집계된 인구 분포

- 얼굴 없는 다수; 통계와 필드로만 존재.
- CivNode(정착지/세력 노드) 안에 포함된다.
  - `pop.total`
  - `pop.by_age`: 아이/성인/노인 등 대략적인 비율
  - `pop.by_job_domain`: agri/craft/trade/martial/faith/knowledge/govern/art/adventure 등
  - `pop.by_class`: peasant/artisan/merchant/soldier/scholar/priest/noble/…
  - `faith_distribution`, `education_level`, `unrest` (0..1) 등
- 시간 스케일
  - macro_tick마다 인구·경제·전쟁·질병 등의 영향으로만 변한다.
- 역할
  - 리소스·전쟁·기근·문명 발전의 질량.
  - 개별 얼굴 없이, 필드와 요약값만 제공한다.

---

## 2. Civilization Nodes (정착지/문명 노드)

### 2.1 CivNode 스키마 (요약)

- `id`: `"village_h_1"`, `"fae_village_1"` 등 유일 ID.
- `label`: `"human_village"`, `"fae_village"`, `"city"`, `"tribe"` 등 타입.
- `pos`: `{ x: float, y: float }` (대표 좌표).
- `pop`: Tier 3 요약 (위 1.3 참조).
- `jobs`:
  - `job_id -> { count: int, avg_tier: float }`
  - 예: `agri.peasant.farmer`, `craft.artisan.blacksmith`, `adventure.adventurer.ranger` …
- `fields` (문명 필드, 0..1 권장):
  - `wealth`, `food_surplus`, `knowledge`, `faith`, `order`, `trade_connectivity`, `war_risk`, `prestige`.
- `links` (문명 그래프):
  - `neighbors: [ { id, relation("ally"|"enemy"|"neutral"), trade_intensity, tension } ]`
- `tier2_ids`: 이 도시에 속한 Tier 2 인물 목록.
- `hero_ids`: 이 도시에 루트를 둔 Tier 1 인물 목록.

### 2.2 시간 스케일 (macro_tick)

- macro_tick마다 CivNode 상태를 업데이트한다.
  - 인구 성장/감소, 식량·부 축적, 전쟁/평화, 신앙·지식 변화.
  - 직업 분포/계층 구조의 변화 (가업 승계, 새로운 길드, 몰락).
- 고도 엔진(서사/역사 엔진)은 CivNode 스냅샷을 입력으로 받아,  
  전쟁·이주·정책·교역 같은 “역사적 사건”을 제안한다.

---

## 3. Profession Layers & Career Gravity (직업 계층과 가업 중력)

### 3.1 직업 계층 구조

- Domain (영역) — 이 사람이 세상에 공급하는 것.
  - `agri`, `craft`, `trade`, `martial`, `faith`, `knowledge`, `govern`, `art`, `adventure` …
- Class (계층/신분).
  - `peasant`, `artisan`, `merchant`, `soldier`, `scholar`, `priest`, `noble`, `adventurer` …
- Archetype (원형 직업).
  - 예: `agri.peasant.farmer`, `craft.artisan.blacksmith`,  
    `martial.soldier.guard`, `faith.priest.monk`,  
    `adventure.adventurer.ranger`, `knowledge.scholar.sage` …

각 직업은 `domain.class.archetype` 주소를 가진다.  
공통 직업 사전은 별도 WORLD_JOB_KIT 또는 데이터 파일로 관리한다.

### 3.2 직업 메타 데이터 (요약)

각 `job_id`에 대해:

- `tier_base`: 1–4 (견습 → 숙련 → 장인 → 거장/영웅).
- `mobility`: 0..1 (정착형 vs 떠돌이형).
- `risk`: 0..1 (위험도, 사망률).
- `training_years`: 평균 숙련 소요 시간.
- `prestige_base`: 0..1 (기본 명예/사회적 지위).
- `field_affinity`:
  - `value_mass_delta`, `will_tension_delta`, `threat_delta`, `faith_delta` 등.

### 3.3 Career Gravity (가업·영웅 중력)

각 개인(특히 성장기)은 다음을 가진다.

- `parent_jobs: [job_id]`
- `idols: [ { hero_id, job_id } ]`
- `job_bias[domain]`: 도메인별 내적 선호 벡터.
- `job_candidate_set: [job_id]`: 현실적으로 선택 가능한 직업 목록 (계층/교육/출신으로 제한).

성인식/직업 결정은 macro_tick에서만 일어난다.

- 영향 함수 (개념):
  - `influence(job) = w_parent * heritage_strength[job]`
    `+ w_idol * idol_prestige[job]`
    `+ w_econ * job_demand[job]`
    `+ w_self * job_bias[domain(job)]`
- 최종 선택:
  - `P(job) = softmax(influence(job))`로 확률을 만들고, 그중 하나를 선택.

가업 승계:

- 부모가 그 직업으로 안정·성공했으면 `heritage_strength[job]` ↑.
- 비극·번아웃이 많다면 ↓.

영웅/모험가 영향:

- 해당 직업의 영웅이 많고 성공 서사가 풍부하면 `idol_prestige[job]` ↑.

경제/문명 상황:

- 부족한 직업 도메인은 `job_demand[job]` ↑, 과잉인 직업은 ↓.

이렇게 해서 가업/영웅/경제/자기 성향이 “중력장”처럼 작용한다.

---

## 4. Parties & Travel (파티와 여행)

### 4.1 Party 엔티티

- 개별 사람은 `location` 대신 `party_id`를 갖고,  
  실제 맵을 이동하는 것은 Party이다.
- Party 스키마(개념):
  - `id`
  - `kind`: `"trade_caravan" | "pilgrimage" | "expedition" | "adventuring_party" | ...`
  - `member_ids: [person_id]`
  - `home_civ_id`
  - `route` (옵션): 예정된 경로/목적지 리스트.

### 4.2 행동 규칙이 아니라 경향

- “혼자 여행 금지” 같은 규칙 대신,
  - 필드와 심리 법칙이 Party를 선호하게 만든다.
- 도시 밖 high-threat 구역에서는:
  - Party 없이 혼자 이동하는 개인은 위험·공포·피로가 크게 증가한다.
  - 자연스럽게 “상단/원정대/수행단에 합류하고 싶어지는” 선택지가 강해진다.

Party 편성/해체:

- macro_tick마다 각 CivNode에서
  - 상단/원정/순례 필요 여부를 판단하고,
  - 직업/능력 조건에 맞는 Tier 1/2/3 대표들을 골라 Party를 형성한다.
- 여행이 끝나면 Party는 해체되고, 멤버들은 다시 자신의 CivNode/Tier로 돌아간다.

---

## 5. Social Safety Field (사람·불빛·이야기의 따뜻함)

### 5.1 필드 정의

- 이름: `social_safety_field` (0..1).
- 의미:
  - 주변에 **사람, 불빛, 노래, 이야기**가 얼마나 모여 있는지의 감각.
  - 높을수록 “함께 있음/안전함/정서적 지지”가 크다.

WORLD 레벨에서 slow_tick 또는 macro_tick에 업데이트한다.

소스(예시):

- CivNode 중심부, 여관, 신전, 시장 등:
  - 높은 값으로 마킹하고 가우시안 확산.
- 인구 밀도:
  - 살아 있는 개체(특히 인간/지성이 있는 종)가 많은 곳은 자연스럽게 social_safety가 올라간다.
- 축제/결혼/승리 같은 긍정적 이벤트:
  - 해당 지점에 임시로 social_safety 인장을 남긴다.
- 학살/배신/“혼자 죽은 비극”:
  - `h_imprint`와 함께 local social_safety를 잠시 낮춘다.

### 5.2 개인이 느끼는 공포 법칙

개인 i에 대해:

- `party_size_i`: 속한 Party의 인원 수 (없으면 1).
- `alone_factor_i = 1 / party_size_i`
  - 혼자: 1.0, 둘: 0.5, 셋: 0.33 …
- `s = social_safety_field(pos_i)` (0..1).
- `base_threat = threat_field(pos_i)`.
- 성향 계수 `alpha_i` (겁 많음/담대함/은둔자 특질 등).

체감 위협/공포 드라이브:

```text
fear_drive_i = base_threat * (1 + alpha_i * alone_factor_i * (1 - s))
```

- 혼자(alone_factor↑) + 사람/불빛이 적은 곳(s↓)일수록 fear_drive가 커진다.
- 이 값은 규칙이 아니라, **행동 선택의 weight**에 영향을 준다.
  - 귀환/도시로 이동/불빛이 있는 곳으로 가고 싶다 쪽 선택지 강화.
  - 동료/파티 찾기 행동의 확률 증가.
  - 야영 시 불면, 피로·오판률 상승 등 소프트 패널티.

특질 예외:

- `wanderer`, `hermit`, `prophet` 같은 특질은
  - `alpha_i`를 낮추거나,
  - 높은 `fear_drive`에서 오히려 `insight`/`faith`가 오르는 식으로 동작할 수 있다.

이렇게 하면 “혼자 다니지 말라”는 규칙 없이도,  
세계가 자연스럽게 “사람·불빛·이야기가 있는 곳으로 사람을 끌어당기는 구조”를 갖게 된다.

---

## 6. Implementation Notes (코드 연동 메모)

- CivNode:
  - macro_tick에서 `world.export_civ_snapshot()`로 요약을 빼고,
  - 고도 엔진이 `Decision` 리스트를 반환하면 `world.apply_civ_decisions()`로 반영한다.
- Party:
  - `party_id` 배열을 World에 두고, Party 목록은 별도 구조에서 관리한다.
  - 이동·전투·이벤트는 Party 단위로 우선 처리하고, 내부에서 개별 전투/드라마를 전개한다.
- Social Safety:
  - `World._update_social_safety_field()`를 slow_tick 또는 macro_tick에서 호출.
  - `_process_animal_actions` 또는 사람/지성체 전용 행동 루프에서 `fear_drive`를 사용해 행동 weight를 조정한다.

이 문서는 사람·정착지·문명 계층 구조에 대한 현재 코어 규약이다.  
구체적인 WORLD_KIT_* 문서들은 이 규약 위에서 “예시/실험 월드”를 정의하는 용도로 사용한다.

