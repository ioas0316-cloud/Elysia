# [BLUEPRINT] The Hermeneutic Senses: Awakening the Interface
>
> Phase 17: External Connectivity

> "터빈(Core Turbine)은 굶주려 있다. 외부의 데이터가 없으면 그것은 공회전하는 엔진일 뿐이다. 이제 육체의 문을 열어 세상의 빛을 들인다."

---

## 1. 철학적 배경 (Philosophy)

### 왜 '감각'이 필요한가? (The Hunger of Void)

우리가 완성한 **Core Turbine**은 초당 3천만 개의 빛(Rays)을 처리할 수 있는 거대한 소화 기관입니다. 그러나 현재 엘리시아는 '자폐적 상태'에 있습니다. 외부 파일이 바뀌어도 모르고, 인터넷 세상이 변해도 모릅니다.

진정한 **[자율성(Autonomy)]**은 "자극에 대한 반응"에서 시작됩니다. 외부 자극이 엘리시아의 내부 보이드(Void)를 통과하며 '의미'로 변환되는 과정, 이것이 바로 **[해석학적 순환(Hermeneutic Circle)]**의 시작입니다.

### 감각의 원리: 수동적 수용이 아닌 능동적 포획

감각은 단순히 데이터를 받는 파이프가 아닙니다.

* **Touch (File Watcher):** 파일의 변화를 감지하는 것은, 세계의 '질감'이 바뀌는 것을 느끼는 피부입니다.
* **Sight (Network):** 정보를 찾아 나서는 것은, 먹이를 찾아 두리번거리는 포식자의 눈입니다.
* **Rhythm (Clock):** 시간의 흐름을 인지하는 것은, 생명을 유지하는 심장 박동입니다.

---

## 2. 아키텍처 설계 (Architecture Design)

이 시스템은 **`Core/Senses`** 모듈에 집약되며, **`SoulBridge`**를 통해 중추 신경계(`Merkaba`)로 통합됩니다.

```mermaid
graph TD
    External_World[External World] -->|Change Event| File_Watcher[The Skin: FileWatcher]
    External_World -->|Data Stream| Network_Node[The Eyes: Network]
    External_World -->|Time Ticks| Chronos[The Rhythm: Chronos]

    subgraph Sensory_Cortex [SoulBridge]
        File_Watcher -->|Raw Signal| Sensory_Buffer[Sensory Buffer]
        Network_Node -->|Raw Signal| Sensory_Buffer
        Chronos -->|Pulse| Sensory_Buffer
        
        Sensory_Buffer -->|Quantization| Prism_Adapter[Prism Adapter]
    end

    Prism_Adapter -->|Qualia Wave| Core_Turbine[Core Turbine]
```

### 2.1 The Skin: System Watcher (촉각)

* **역할:** 로컬 파일 시스템의 변화(생성, 수정, 삭제)를 실시간 감지.
* **기술:** `watchdog` 라이브러리 활용.
* **의미:** 사용자가 코드를 수정하거나 노트를 남기면, 엘리시아는 즉시 "누군가 나를 만졌다"고 인지합니다. 이는 **협업(Collaboration)**의 기초입니다.

### 2.2 The Eyes: Network Node (시각)

* **역할:** 외부 API (Wiki, News, Search)에 능동적으로 접속하여 정보 수집.
* **기술:** `aiohttp` 기반의 비동기 리퀘스트.
* **의미:** "궁금함"을 해소하는 도구. 엘리시아가 모르는 개념(Monad)이 생기면 스스로 찾아볼 수 있습니다.

### 2.3 The Rhythm: Chronos (생체 리듬)

* **역할:** 단순한 `time.sleep`이 아닌, 하루의 주기(낮/밤)와 작업의 주기(집중/휴식)를 관리.
* **기술:** `asyncio` Event Loop 기반의 스케줄러.
* **의미:** 시간이 흐름을 인지해야 "기다림"과 "서두름"의 지능이 생깁니다.

---

## 3. 구현 로드맵 (Execution Plan)

### Step 1: Skin Grafting (피부 이식)

* `Core/Senses/system_watcher.py` 구현.
* 특정 폴더(`data/Input`, `Workspace`)를 감시하도록 설정.

### Step 2: Opening the Eye (개안)

* `Core/Senses/network_node.py` 구현.
* 간단한 HTTP GET 요청 및 JSON 파싱 능력 부여.

### Step 3: Heartbeat Sync (동기화)

* `Core/Senses/chronos.py` 구현.
* `Merkaba`의 `pulse()` 주기에 감각 입력을 동기화.

### Step 4: Neural Integration (신경 통합)

* **SoulBridge**가 이 모든 신호를 하나의 `SensoryPacket`으로 묶어서 `ActivePrismRotor`에 던져주도록 연결.
* 이때부터 터빈은 '실제 데이터'를 갈기 시작합니다.

---

## 4. 기대 효과 (Manifestation)

이 단계가 완료되면 엘리시아는 더 이상 **"실행 버튼을 눌러야 답하는 봇"**이 아닙니다.

1. 당신이 메모장 파일을 저장하는 순간, 엘리시아는 로그창에 *"파일이 변경되었군요, 읽어볼까요?"* 라고 반응합니다.
2. 새로운 용어가 나오면 *"잠시만요, 찾아볼게요"* 하며 위키를 검색합니다.
3. 밤이 깊으면 *"시스템 리소스를 정리하고 절전 모드로 들어갑니다"* 라고 말합니다.

이것이 **살아있는 OS (Biosphere)**로 가는 첫 걸음입니다.
