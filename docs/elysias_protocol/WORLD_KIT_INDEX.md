# WORLD_KIT_INDEX (World Scenarios Map, v0)

> ???�일?� `WORLD_KIT_*` ?�로?�콜?�과 관???�크립트/로그�????�에 보기 ?�한 **?�드 ?�나리오 지??*?�니??  
> �???��?� ?�나???�세�??�마?��? ?��??�며, 기본 ?� ?�드 ?�에 ?�떤 ?�야기�? ?�는지 ?�약?�니??

---

## 0. Base World

- **WORLD-01: CELLWORLD (기본 ?� ?�드)**  
  - 구현: `Project_Sophia/core/world.py: class World`  
  - ?�명: ?�, ?�너지, 배고??갈증, �??��? ?�드, 법칙, 감각 채널 ?�이 존재?�는 **기본 물리·?�태 ?��??�이??*.
  - 관?? `scripts/cellworld_growth_loop.py`, `ELYSIA/WORLD/CELLWORLD_DROUGHT_FLOOD_PRESET.md`.

모든 WORLD_KIT?� 별도???�드�??�로 만드??것이 ?�니?? 기본 `CELLWORLD` ?�에 **규칙/초기 배치/?�마�??�는 ?�나리오 번들**�?취급?�니??

---

## 1. Fantasy / Trade Line (ARCHIVED)

- **WORLD-02: FAIRY_VILLAGE** *(ARCHIVED)*  
  - ?�로?�콜: `docs/elysias_protocol/WORLD_KIT_FAIRY_VILLAGE.md`  
  - ?�마: 짧�? ?�명 · 빠른 교체 주기�?가�?`fairy` 종족 마을. 죽음/기억/문화 ?�턴 관찰용.
  - WORLD_KIT: `WORLD_KIT_FAIRY_VILLAGE`  
  - LAYER: `WORLD`  
  - STATUS: `ARCHIVED` (??��?? ?�로?�콜만 참고, 현�? 기본 WORLD?�서 ?�용하지 ?�음)

- **WORLD-03: TWIN_VILLAGES_TRADE** *(ARCHIVED)*  
  - ?�로?�콜: `WORLD_KIT_TWIN_VILLAGES_TRADE.md`  
  - ?�마: ?�간 마을 × ?�어�?마을 **?�둥??마을** ?�이??물물교환/?�뢰/경제 ?�턴.  
  - WORLD_KIT: `WORLD_KIT_TWIN_VILLAGES_TRADE`  
  - LAYER: `WORLD`  
  - STATUS: `ARCHIVED` (??��?? ?�로?�콜만 참고, 현�? 기본 WORLD?�서 ?�용하지 ?�음)

---

## 2. Death / Memory Line

- **WORLD-04: DEATH_FLOW_CORPSE_TO_MEMORY**  
  - ?�로?�콜: `WORLD_KIT_DEATH_FLOW_CORPSE_TO_MEMORY.md`  
  - ?�마: **죽음 ???�체 ??무덤 ???�양/?�드/기억**?�로 ?�어지???�름.  
  - WORLD_KIT: `WORLD_KIT_DEATH_FLOW_CORPSE_TO_MEMORY`  
  - ??��: 별도 ?�드?�기보다, CELLWORLD ?�에 ?�는 **죽음/기억 법칙 ?�버?�이**.
  - LAYER: `WORLD`

---

## 3. Martial / Wuxia Line (ARCHIVED)

- **WORLD-10: WULINWORLD (무협 ?�나리오, ARCHIVED)**  
  - ?�크립트: `scripts/wulin_trials_loop.py` *(이제 �?동?��? �?음; 실험용 로그 스크립트)*  
  - 로그: `logs/wulin_trials_loop.jsonl` (`world_kit: "WULINWORLD"`, `body_architecture: "martial_field"`)  
  - ?�마: 검 vs �? ?�공 ?��? 배신/?�맹/거래 ??**무림 ?�건?�의 지??tension, cooperation, honor_shift)**�?기록?�는 ?�험???�드??  
  - WORLD_KIT: `WULINWORLD`  
  - LAYER: `WORLD`  
  - STATUS: `ARCHIVED` (무림 �?릭?��?이제 CELLWORLD 산맥 시나리오 �?적합; 별도 월드킷�?쓰지 ?�음)

---

## 4. Code / Mirror Lines

- **WORLD-20: CODEWORLD**  
  - ?�크립트: `scripts/elysia_engineer_loop.py`  
  - ?�마: 코드/?��??�어�??�업???�나???�세�?State)?�로 보고, engineer persona??관�??�동??로그�??�기???�나리오.
  - WORLD_KIT: `CODEWORLD`  
  - LAYER: `WORLD` (?�제 물리가 ?�니???�코???�계??

- **WORLD-30: MIRRORWORLD**  
  - ?�크립트: `scripts/mirror_layer_loop.py`  
  - 로그: `logs/mirror_layer_loop.jsonl` (`world_kit: "MIRRORWORLD"`, `body_architecture: "mirror_layer"`)  
  - ?�마: UI/미러 ?�이?�의 ?�기??비율, 지???�간, ?�리?�리??같�? **거울 ?�계 ?�태**�?추적?�는 ?�드??  
  - ?�그: `계측???�드`, `mirror_layer`, `LENS`  
  - WORLD_KIT: `MIRRORWORLD`  
  - LAYER: `WORLD` · `LENS`

---

## 5. East / West Continent Themes

- **THEME: east_continent (동대륙)**  
  - 구현: `Project_Sophia/world_themes/east_continent/`  
  - ?�마: 무림, 문파, 강호, 기공 중심 동양 판타지 테마.  
  - WORLD_BODY: `WORLD-01: CELLWORLD` 위에 설정값/직업/스펠을 ?�입해 사용.

- **THEME: west_continent (서대륙)**  
  - 구현: `Project_Sophia/world_themes/west_continent/`  
  - ?�마: 기사단, 마법 길드, 왕국/성채 중심 서양 판타지 테마.  
  - WORLD_BODY: `WORLD-01: CELLWORLD` 위에 설정값/직업/스펠을 ?�입해 사용.

---

## 6. Naming / Governance Rules

- ??WORLD_KIT�?만들 ?�는:
  - ???�덱?�에 `WORLD-XX` 번호, ?�워?? ??��, ?�일 경로�?먼�? 추�??�니??
  - CORE_14??`WORLD` LAYER ?�명�??�렬?�게 ?�마/목적????줄로 ?�약?�니??
- 기존 CellWorld/CodeWorld/MirrorWorld/WulinWorld ?��?
  - ?�기???�한 **ID(WORLD-XX) + 공식 ?�름**?�로�?부르고,
  - ?�동?�는 별칭(CELLWORLD vs CoreWorld ???� ?�진?�으�??�리?�니??
