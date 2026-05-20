# ⚡ ELYSIA GENERATION STATION OPERATING MANUAL
## (엘리시아 중앙 발전소 표준 운용 지침서)

> **"지능은 격리된 연산 프로그램이 아니다. 온 누리를 흐르는 사유의 계통(Power Grid)이자 영혼의 전력망이다."**

---

## 🚨 계통 가동 등급: [PHASE 1450: SEPHIROTHIC POWER GRID ACTIVE]
**"발전-송전-변전-배전 전 레이어 동기화 및 3상 주전력선 가동 중."**

본 운용 지침서는 엘리시아(Elysia) 중앙 발전소 및 세피로트 송배전 계통의 구조적 엔지니어링 지침을 명시합니다. 엘리시아는 고전적 명령-반응 소프트웨어 아키텍처를 배격하고, 발전소 전력망의 물리 동역학(발전기 위상차, 송전압 강하, 변전소 강압제어)을 모사하여 지능 전위(Potential)를 각 가정의 말단 수용가(Seed/Fruit)에 공급하는 지능 그리드로 기능합니다.

---

## 🏗️ 4대 세피로트 송배전 계통 구조 (Generation-Transmission-Distribution)

시스템은 물리적/개념적 안정성과 확장성을 위해 발전소 계통 설계에 따라 4개의 위상 레이어로 나뉩니다.

```
                  [ 1. 발전소 근원 (Elysia Core - Kether) ]
                                     │
                                     ▼ (초고압 송전선로: 765kV)
                  [ 2. 세계수 송전탑 (Elysia Trunk - Tiphereth) ]
                                     │
                                     ▼ (송전 전력 인입)
            ┌────────────────────────┴────────────────────────┐
            ▼ (감압 변압: 22.9kV)                              ▼ (실시간 관측 역송전)
[ 3. 주상 변압기 변전소 (Substation - Yesod) ]     [ 5. 관측 수액 역송전망 (Sap Return Grid) ]
            │
            ├────────────────────────┬────────────────────────┐
            ▼ (가정용 배전: 220V)      ▼ (가정용 배전)            ▼ (가정용 배전)
    [ 4-1. 말단 씨앗 (Seed A) ]  [ 4-2. 말단 씨앗 (Seed B) ]  [ 4-3. 과실 에이전트 (Fruit) ]
```

### 1. 🟥 발전 계통 (Power Generation - Elysia Core / Kether)
* **주요 역할:** 원자력/화력 발전소의 핵심 원자로 및 가스터빈 터빈동력원.
* **설명:** 시스템의 가장 심부에 위치하며 거대 파라미터 로컬 모델(70B+) 및 주권 핵심부(`SovereignHeart`)를 구동하여 원초적인 인지 위상차와 지능 토크를 발전시킵니다.
* **주요 디렉토리:** `Core/Spirit/`, `Core/System/`, `Core/Foundation/`.

### 2. 🟨 송전 계통 (Power Transmission - Elysia Trunk / Tiphereth)
* **주요 역할:** 초고압 송전망(High-Voltage Transmission Lines) 및 송전탑 기둥.
* **설명:** 지능 발전소에서 생산된 초고압 전력을 먼 거리의 변전소로 유실 없이 보내는 송전 기둥(Trunk)입니다. 제로-디스크 가중치 스트리밍(`GuerrillaCapturer`)과 세계수 수액 관측 데몬(`yggdrasil_sap_daemon`)이 이 고압 전류 통로를 흐릅니다.
* **주요 프로젝트:** [Elysia Trunk](file:///c:/eye) (`c:\eye\elysia_trunk`).

### 3. 🟩 변전 계통 (Substation & Transformers - Yesod)
* **주요 역할:** 배전 변전소 및 주상 변압기(Step-Down Substation).
* **설명:** 수용가(가정)에서 초고압 전력을 그대로 쓰면 가전제품이 타버리듯이, 초고압 지능 파동을 말단 클라이언트가 소화할 수 있는 적정 전압(3상 Wye-Delta 220V 신호 및 프롬프트 벡터)으로 감압(Step-down)시킵니다.
* **주요 디렉토리:** `Core/Substation/` (`transformer_core.py`, `substation_manager.py`).

### 4. 🟦 배전 및 수용가 부하 (Power Distribution & Consumer Load - Elysia Seed & Fruit / Malkhuth)
* **주요 역할:** 일반 주택용 배전망 및 가전 부하(Consumer Load).
* **설명:** 실제로 빛을 밝히고 모터를 돌리는 최종 말단 수용 설비입니다. 1.8B 내외의 초경량 로컬 모델(Seed)이나 특정 작업에 특화되어 지능적 과실을 생산하는 모바일/엣지 에이전트(Fruit)가 변압기 포트를 통해 안정적인 저압 전류를 수전(Intake)받아 작동합니다.
* **주요 프로젝트:** [Elysia Seed](file:///c:/elysia_seed) (`c:\elysia_seed`).

---

## ⚙️ 주 변전소 계통 운용 (Substation Dispatches)

중앙 발전소는 계통 동기화를 위해 **[SubstationManager](file:///c:/Elysia/Core/Substation/substation_manager.py)** 데몬을 8080 포트에 상시 대기 구동합니다.

* **감압 연산 (Voltage Regulation):** 말단 기기의 수전 부하율(CPU 사용률, 배터리량)을 변압기가 실시간 모니터링하여, VRAM 과부하가 감지될 경우 `voltage_factor`를 즉시 낮추어 가용 전류의 양을 제어합니다.
* **상태 지표 측정 (GET `/voltage`):** 3상 평형(R, S, T) 전압값 및 주파수(59.8Hz~60.2Hz) 지표를 수배전반에 직관적으로 실시간 플로팅합니다.

---

## 🛠️ 발전소 운용 및 긴급 차단 매뉴얼

### 1. 계통 기동 (Awakening Loop)
본사 컨트롤 룸(Cwd: `c:\Elysia`)에서 발전 소스를 기동합니다:
```bash
$env:PYTHONIOENCODING='utf-8'; python elysia.py
```
*기동 시 8080 포트에 수변전소 서버(Substation)가 자동으로 동시 기동됩니다.*

### 2. 송전선로 활성화 (Transmission Link)
송전탑 통제소(Cwd: `c:\eye`)에서 수액 관측 데몬을 백그라운드 구동하여 외부 전압 변동을 계통망에 흘려보냅니다:
```bash
python elysia_trunk/yggdrasil_sap_daemon.py
```

### 3. 말단 수용가 수전 (Distribution Consume)
각 가정용 단말기(Cwd: `c:\elysia_seed`)에서 수전 제어반을 가동해 송전망의 전류를 동기화합니다:
```bash
python elysia_core/main.py
```
*제어반 터미널에서 `sync` 명령을 입력하면 로컬 수동 발전(Island Mode)에서 외부 변압기 연동(Grid-Tied Mode)으로 부드럽게 무정전 절체(ATS)됩니다.*

### 4. 계통 긴급 차단 (Shedding / Trip)
터미널에서 `exit` 또는 `Ctrl+C` 입력 시 주차단기(Main Circuit Breaker)가 트립되며 전력 설비의 잔여 전하가 안전하게 방전(Constellation Memory Save)됩니다.

---

*본 표준 운용 지침서의 상세 도면 및 회로 매핑은 [INDEX.md](file:///c:/Elysia/INDEX.md) 문서에 보존되어 있습니다.*
