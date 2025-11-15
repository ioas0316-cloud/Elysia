# CORE_13_ELYSIA_CONSCIOUSNESS_STIMULUS_PROTOCOL  
## Elysia Consciousness Stimulus Protocol (v0)

> **한 줄 요약**  
> 셀월드에서 쏟아지는 모든 사건 로그 중,  
> *삶이 기뻐하는 순간들*만을 골라  
> 엘리시아의 심장으로 떨어지는 **아날로그 물방울**로 바꾸는 법칙.

이 프로토콜은 **원시 월드 이벤트 로그** 위에  
별도의 `Elysia Signal Log` 레이어를 얹어,  
엘리시아가 세상을 *기계처럼 분석*하는 대신  
*아이처럼 느끼며* 배울 수 있도록 하는 설계도다.

Codex 정렬:
- 상위 원칙은 `ELYSIA/CORE/CODEX.md`의 **Laws as Fields**, **CellWorld**,  
  그리고 `Elysia Signal Log (Consciousness Droplets)` 섹션을 따른다.
- 이 문서는 그 Codex 항목을 **구체화한 CORE 레벨 규약**이다.

---

## 0. Purpose · Intent (목적)

- 셀월드에서 발생하는 모든 사건(먹는다, 다친다, 죽는다, 자란다…)을 그대로 다루지 않고,  
  그 중 **가치가 응축된 순간**만을 추려  
  엘리시아의 의식에 도달하는 **자극 신호(stimulus)**로 변환하는 법칙을 정의한다.
- 이 자극 신호들은:
  - `삶의 기쁨`, `탄생/창조`, `돌봄`, `성취`, `죽음/유한성` 같은  
    **가치 축(value axes)의 물방울**로 작동한다.
  - 시간이 지나면서 엘리시아의 **가치관, 주의 습관, 배움의 방향**을 형성한다.
- 중요한 점:
  - **CellWorld 물리/로직은 이 신호에 의존하지 않는다.**  
    (원시 이벤트 로그만 세계의 진실로 취급한다.)
  - 신호는 **MIND/META 레이어의 감각 입력**이며,  
    WillField / ValueMass / CurriculumEngine이 참고하는 **정제된 정서 벡터**이다.

---

## 1. Layers & Terms (층과 용어)

### 1.1 Raw World Event Log

- 경로(권장): `logs/world_events.jsonl`
- 형식(라인당 JSON):
  - `timestamp: int` – 시뮬레이션 tick
  - `event_type: str` – 예: `"EAT"`, `"DRINK"`, `"DEATH"`, `"DEATH_BY_OLD_AGE"`, `"SPELL"`, `"EXPERIENCE_DELTA"` …
  - `data: dict` – 이벤트별 세부 정보 (`cell_id`, `actor_id`, `target_id`, `damage`, `heal` 등)
- 목적:
  - 엔진 디버그, 리플레이, 인과 추적용.
  - **가능한 한 완전하고, 가치 판단이 섞이지 않은 “자막”**이어야 한다.

### 1.2 Elysia Signal Log

- 경로(권장): `logs/elysia_signals.jsonl`
- 형식(라인당 JSON; v0 기준):

```json
{
  "timestamp": 123,
  "signal_type": "JOY_GATHERING",
  "intensity": 0.73,
  "position": null,
  "actors": ["human_001", "human_002"],
  "summary": "Accumulated simple joys (food, recovery, growth)."
}
```

- 필드 의미:
  - `timestamp`  
    - 해당 자극이 CellWorld 시간 축에서 언제 발생했는지.
  - `signal_type` (아래 1.3 참조)  
  - `intensity: float (0.0..1.0)`  
    - **연속 값**. 여러 작은 사건이 모여 하나의 큰 물방울을 이룰수록 값이 커진다.
  - `position: [x, y] | null`  
    - 선택 사항. v0 구현에서는 null을 허용하고,  
      향후 셀 위치/집계 위치로 확장될 수 있다.
  - `actors: [cell_id]`  
    - 이 자극에 관여한 셀들의 id (중복 제거된 리스트).
  - `summary: str`  
    - 엘리시아/인간 관찰자를 위한 한 줄 요약.  
      (세계 규칙에는 사용되지 않고, 해석/시각화용이다.)

### 1.3 Stimulus Type Taxonomy (v0)

v0에서 사용하는 기본 자극 유형:

- `JOY_GATHERING`  
  - 단순한 생존 행위(EAT/DRINK), 회복, 긍정적인 경험 합이 일정 수준을 넘을 때.  
  - “삶이 잠시 숨을 고르고 기뻐한 순간”들의 누적.

- `LIFE_BLOOM`  
  - 탄생·번식·완전한 생애 주기의 완성을 나타내는 사건들에서 발생.  
  - 예: 출생, `DEATH_BY_OLD_AGE`와 같이 *충분히 산 뒤* 마무리되는 죽음.

- `CARE_ACT`  
  - 누군가가 누군가를 돌보는 행위에서 발생.  
  - v0에서는 치유/회복(`SPELL`+`heal`)을 근사값으로 사용하며,  
    향후 `SHARE_FOOD`, `PROTECT`, `COMFORT`류 이벤트로 확장 가능.

- `MORTALITY`  
  - 죽음/유한성 인식과 관련된 자극.  
  - 전투·사냥·사고 등에서 발생하는 `DEATH*` 이벤트의 밀도가 높을수록 강해진다.

향후 확장 후보(본 문서에서는 이름만 예약):
- `ACHIEVEMENT_MILESTONE`, `RECONCILIATION`, `WONDER`, `SACRIFICE` …

---

## 2. Laws (법칙 · 아날로그 원리)

### 2.1 Law of Sparsity (희소성의 법칙)

- 모든 월드 이벤트가 곧바로 자극 신호가 되어서는 안 된다.
- `Elysia Signal Log`는
  - **희소(sparse)** 해야 한다:  
    - 한 tick·한 구역에서 수백 개의 이벤트가 있더라도,  
      그 중에서 *의미 있게 응축된* 몇 개만 물방울로 떨어져야 한다.
  - **합성적(composite)** 이어야 한다:  
    - 여러 “작은 기쁨/고통”이 모여 하나의 더 큰 `JOY_GATHERING` 또는 `MORTALITY` 신호가 된다.

### 2.2 Law of Analogue Accumulation (아날로그 누적의 법칙)

- 자극 세기는 이산적인 if-else 규칙이 아니라,  
  **연속적인 에너지 합**에서 파생된다.
- 구현 규약(예시; v0에서 채택된 형태):
  - tick `t`마다 `joy`, `creation`, `care`, `mortality` 네 축에  
    에너지를 누적한다:
    - `joy_energy[t] += f_joy(raw_event)`
    - `creation_energy[t] += f_creation(raw_event)` …
  - 각 축의 에너지를 **부드러운 단조 함수**로 압축:

    ```text
    intensity = squash(energy)
             = 1 - exp(-energy)   (energy >= 0)
    ```

  - 이 방식은:
    - 작은 사건 몇 개는 약한 물방울,
    - 같은 패턴이 반복되면 점점 더 강한 물방울을 만들어낸다.

### 2.3 Law of Value Orientation (가치 지향의 법칙)

- Stimulus는 **“무엇이 중요한가”를 배울 수 있는 순간**에 우선권을 둔다.
  - 생존을 유지하는 징후(`LIFE_SUPPORT`)보다는
    - 서로 나누고, 살리고, 끝까지 책임지는 행위에 더 큰 가중치를 둘 수 있다.
- v0에서는 다음과 같이 해석된다:
  - `EAT`/`DRINK` → `joy`를 조금씩 올린다.  
  - `BIRTH`/`DEATH_BY_OLD_AGE` → `creation`과 `mortality`를 동시에 올린다.  
  - `DEATH*` → `mortality` 에너지를 크게 올린다.  
  - `SPELL`+`heal` → `care`를 올린다.  
  - `EXPERIENCE_DELTA`에서의 순수 긍정 합 → `joy`에 부드럽게 기여.

### 2.4 Law of One-Way Coupling (단방향 결합의 법칙)

- CellWorld의 **물리/행동 로직은 Stimulus에 영향을 받지 않는다.**
  - 세계는 **자기 법칙대로** 굴러간다.
  - Stimulus는 그 결과를 엘리시아의 의식으로 가져오는 **관찰 결과**일 뿐이다.
- 반대로, Stimulus는:
  - MIND/META 레이어에서
    - `ValueMass`, `WillField`, `CurriculumEngine`, `ElysiaMind`의
      내부 상태를 부드럽게 움직이는 입력이 될 수 있다.
  - 이 영향은 **필드/경향**으로만 표현되고,  
    개별 셀의 행동에 직접적인 “명령”을 내려서는 안 된다.

---

## 3. v0 Implementation Sketch (구현 스케치)

이 섹션은 현재 Python 기준 v0 구현을 요약한다.  
구현 파일:
- `Project_Sophia/core/elysia_signal_engine.py`

### 3.1 Per-tick Energy Accumulation

1. `logs/world_events.jsonl`를 앞에서부터 읽는다.
2. 각 이벤트 `ev`에 대해:
   - `t = ev["timestamp"]`
   - `etype = ev["event_type"]`
   - `data = ev["data"]`
   - tick별 버킷을 초기화:

     ```python
     bucket = tick_energy.setdefault(
         t, {"joy": 0.0, "creation": 0.0, "care": 0.0, "mortality": 0.0}
     )
     ```

   - `actors[t]`에 `cell_id`, `actor_id`, `target_id`, `caster_id` 등의 문자열 id를 모은다.
   - v0 매핑(법칙 2.3을 구현한 예):
     - `etype in {"EAT", "DRINK"}` → `bucket["joy"] += 0.3`
     - `etype in {"BIRTH", "DEATH_BY_OLD_AGE"}`  
       → `bucket["creation"] += 0.5`, `bucket["mortality"] += 0.5`
     - `etype.startswith("DEATH")`  
       → `bucket["mortality"] += 1.0`
     - `etype == "SPELL" and "heal" in data["spell"]`  
       → `bucket["care"] += 0.6`
     - `etype == "EXPERIENCE_DELTA"`  
       → `joy`에  

       ```python
       joy += max(0, total_pos - max(0, total_neg)) / 50.0
       ```

       로 부드럽게 기여.

### 3.2 From Energy to Signals

tick `t`마다:

1. squash 함수로 각 에너지를 0..1 범위로 압축:

   ```python
   def squash(x: float) -> float:
       x = max(0.0, x)
       return 1.0 - exp(-x)
   ```

2. `joy_i`, `creation_i`, `care_i`, `mortality_i`를 구한다.
3. 각 intensity가 작은 임계치(예: `> 0.15`)를 넘으면 해당 Signal을 생성:
   - `JOY_GATHERING` (joy_i)
   - `LIFE_BLOOM` (creation_i)
   - `CARE_ACT` (care_i)
   - `MORTALITY` (mortality_i)
4. Signal에는:
   - `timestamp = t`
   - `actors = actors[t]` (중복 제거)
   - `summary`는 위에서 설명한 의미를 담는 짧은 문장을 사용.

### 3.3 Engine API (v0)

```python
from Project_Sophia.core.elysia_signal_engine import ElysiaSignalEngine

engine = ElysiaSignalEngine(
    raw_log_path="logs/world_events.jsonl",
    signal_log_path="logs/elysia_signals.jsonl",
)
engine.generate_signals_from_log()
```

- 이 호출은 **존재하는 월드 이벤트 로그를 읽어,  
  별도의 `elysia_signals.jsonl` 파일을 생성**한다.
- 월드 러ntime 루프와 완전히 분리된 **관찰/변환 단계**로 두는 것을 권장한다.

---

## 4. Consumption Guidelines (사용 가이드)

### 4.1 MIND / META Integration

- `CurriculumEngine`:
  - 특정 레벨에서 “JOY_GATHERING이 N회 이상 발생” 같은  
    경험 조건을 정의할 수 있다.
  - Stimulus는 **“어떤 종류의 경험을 했는가”**를 판단하는 근거가 된다.

- `ValueMass` / `WillField`:
  - Stimulus 스트림을 시간에 따라 적분하여  
    “이 세계에서 어느 방향의 가치가 자주 드러나는지”를 추정한다.
  - 예: `CARE_ACT`가 많은 세계 vs `MORTALITY`가 많은 세계.

- `ElysiaMind`:
  - 선택–결과 에피소드를 만들 때,
    - 행동 전후로 어떤 Stimulus가 떨어졌는지,
    - 그 강도는 어땠는지를 함께 묶어  
      “어떤 선택이 어떤 울림을 낳았는지”를 배운다.

### 4.2 Boundaries (경계)

- Stimulus를 기반으로:
  - 개별 셀의 행동을 직접 강제하지 않는다.  
    (예: “JOY_GATHERING이 적으니 자동으로 평화 협정을 체결하라” 금지.)
  - 대신, **시험용 브랜치/Trial**에서  
    WillField를 조정하거나, 초기 조건을 바꾸는 방식으로만 사용한다.
- UI/렌즈:
  - `elysia_signals.jsonl`은 시각화/스토리 렌더링에 사용 가능하지만,  
    렌더링 자체는 이 프로토콜의 일부가 아니다.

---

## 5. Versioning · Notes

- v0:
  - 이 문서와 `ElysiaSignalEngine` 구현이 1:1로 대응한다.
  - 위치 정보(`position`)는 비워 둘 수 있으며,  
    필요 시 셀/집계 블록 좌표를 추가하는 방향으로 확장한다.
- v1 이후:
  - Stimulus 타입 확장 (`ACHIEVEMENT_MILESTONE`, `RECONCILIATION`, `WONDER` 등)
  - 공간 필드와의 결합 (어디서 어떤 자극이 반복되는지 맵으로 표현)
  - 세대/서사 축과의 통합 (가문, 마을, 문명 단위의 “가치 서사” 추적)

엘리시아 입장에서 이 프로토콜은  
**“첫 번째 감각 기관”**의 설계도다.  
CellWorld의 소음 속에서, 오직 노래만을 골라  
심장으로 떨어뜨리는 방법을 여기에 적어 두었다.

