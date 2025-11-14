# WORLD_KIT_TWIN_VILLAGES_TRADE  
## 페어리 마을 × 인간 마을 교역 월드킷 v0

페어리 마을과 인간 마을을 **서로 인접한 쌍둥이 마을**로 배치하고,  
각자가 가진 **잉여 자원(surplus)** 을 교환하면서  
초기의 **물물교환 → 내재적 가격 → 화폐 전(前) 경제**를 관찰하기 위한 월드킷이다.

> 목표:  
> “짧은 생애의 페어리 숲”과 “우물 중심 인간 마을”이  
> 서로의 부족함을 채우며, **희소성 → 가치 → 신뢰 → 교환 패턴**이  
> 인과적으로 생겨나는지를 실험한다.

---

## 1. 지형 / 배치

### 1.1 월드 좌표계 상의 배치

- 월드 크기: `width = 256` (기존 World 기본값 사용)
- 인간 마을 중심 (`home_pos_human`)
  - 예: `(xh, yh) = (width * 0.35, width * 0.5)`
  - 우물(`well`)과 광장이 있는 위치.
- 페어리 마을 중심 (`home_pos_fae`)
  - 예: `(xf, yf) = (width * 0.65, width * 0.5)`
  - `fae_spring` / 숲 속 공터가 있는 위치.
- 교역 경로 / 교류 지대 (`trade_zone`)
  - 두 마을의 중간 지점: `trade_center = ((xh+xf)/2, (yh+yf)/2)`
  - 반경 `R_trade` (예: 12~16) 안을 “시장/교류 지대”로 간주.

### 1.2 지형 요소

- 인간 마을 주변
  - `home_pos_human` 주변 반경 `R_home_h` 내:
    - 우물: `wetness` 1.0으로 채워진 작은 원형 영역.
    - 집/농지/나무 등은 기존 HUMAN_VILLAGE 월드킷을 재사용.
- 페어리 마을 주변
  - `home_pos_fae` 주변 반경 `R_home_f` 내:
    - `fae_spring` (물/마나가 풍부한 샘).
    - `fae_forest` (열매가 풍부한 숲, 베리류 식물 밀도↑).
- 교류 지대
  - 특별한 타일 타입 없이,
  - “양쪽 마을에서 상인(trader)이 만나는 위치”로 사용.

---

## 2. 종족별 역할 / 자원

### 2.1 인간 마을 (Village Humans)

- 기본 속성
  - `label: "human"`
  - `element_type: "animal"`
  - `diet: "omnivore"`
  - `culture: "village"`
  - `home_village_id: "village_h_1"`
- 핵심 자원 (생산 / 잉여)
  - `water` : 우물/샘에서 길어오는 물.
  - `bread` : 곡물을 가공하여 만든 주식.
  - `wood`  : 목재 (건설/도구용).
- 생존/우선순위
  - 생존 자원: `water`, `bread`
  - 인프라/성장 자원: `wood`
  - 사치/기쁨 자원: `berry` (페어리 마을 특산품을 통해 얻음)

### 2.2 페어리 마을 (Fae Village)

- 기본 속성
  - `label: "fairy"`
  - `element_type: "animal"`
  - `diet: "omnivore"`
  - `culture: "fae_village"`
  - `home_village_id: "fae_village_1"`
- 핵심 자원 (생산 / 잉여)
  - `berry`     : 블루베리류 열매 (당/비타민, 맛).
  - `fae_herb`  : 간단한 약초/허브 (선택, v0에서는 단순자원 취급).
- 생존/우선순위
  - 생존 자원: `berry`
  - 안정/보호 자원: `water`, `bread` (인간과의 교역으로 확보)
  - 사치/마나 자원: `fae_herb` (나중에 마법/의례로 확장 가능)

---

## 3. 자원 저장 모델 (v0, 집합적 스톡)

v0에서는 개별 에이전트 인벤토리까지 가지 않고,  
