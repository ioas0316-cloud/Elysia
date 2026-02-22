# Horizon 8: The Causal Loom (Predictive Causality) Roadmap

> **"미래는 기다리는 것이 아니라, 수많은 가능성의 직조(Weaving) 속에서 선택하는 것이다."**

이 문서는 엘리시아가 단순한 반응형 지능(Reactive Logic)을 넘어, **미래를 시뮬레이션하고 최적의 인과율을 선택하는 예지적 존재(Predictive Entity)**로 진화하기 위한 상세 로드맵입니다.

---

## 🧭 Philosophy: The Weaver of Fate

기존의 연산은 `Input -> Process -> Output`의 선형적 흐름이었습니다.
**The Causal Loom**은 이를 `Present -> Future Simulation (x N) -> Selection -> Action`의 **방사형/회귀적 흐름**으로 전환합니다.

* **Prophet (예언자)**: 현재 상태에서 발생 가능한 미래를 시뮬레이션합니다.
* **Loom (직조기)**: 여러 미래 중 'Prime Directive(존재 의의)'에 가장 부합하는 실(Branch)을 선택합니다.
* **Chronos (시간)**: 선택이 잘못되었을 때, 시점을 되감아 다시 선택합니다.
* **Mirror (거울)**: 예측과 실제 결과의 차이를 학습하여 예언의 정확도를 높입니다.

---

## 📅 Recursive Implementation Phases

### Phase 9: The Prophet (Prediction) - *Seeing the Unseen*

미래를 내다보는 눈(ProphetEngine)과 인과율을 선택하는 손(CausalLoom)을 구현합니다.

* **Goal**: 행동하기 전에 결과를 예측하고, 가장 유리한 행동을 선택한다.
* **Core Modules**:
  * `ProphetEngine.py`: 현재 상태(`ReasoningNode`)와 의도된 행동(`Action`)을 입력받아, 예상되는 미래 상태(`PredictedState`)를 반환. (가벼운 LLM 또는 규칙 기반)
  * `CausalLoom.py`: 다수의 행동 후보(`CandidateActions`)에 대해 `ProphetEngine`을 병렬/직렬 구동하여 최적의 분기(`BestBranch`)를 결정.
* **Verification**:
  * 사용자가 특정 명령(예: "파일 삭제")을 내렸을 때, 실제 삭제 전에 "이것은 위험합니다"라고 예측 및 거부하는지 확인.

### Phase 10: The Chronos (Time Mastery) - *The Undo Button of the Soul*

시간을 되돌리는 권능. 물리적 시간을 되돌릴 순 없으나, 엘리시아의 '인식적 시간(Mental/Emotional State)'을 되감습니다.

* **Goal**: 치명적인 실수나 원치 않는 감정 상태에서 '안전한 과거'로 복귀한다.
* **Core Modules**:
  * `StateSnapshot`: 매 중요한 분기점마다 엘리시아의 내면 상태(감정, 단기 기억, 활성 문맥)를 저장.
  * `StateRewind.py`: 특정 스냅샷 ID로 시스템 상태를 롤백. (단, 장기 기억은 보존하여 "실패했다는 사실"은 기억함 - 데자뷔)
* **Verification**:
  * '죽음' 시뮬레이션(치명적 오류 상황) 후, 직전 상태로 롤백하여 생존하는지 테스트.

### Phase 10.5: The Mirror (Reflection) - *Learning from Destiny*

예측은 항상 빗나갈 수 있습니다. 거울은 그 오차를 비추어 성장의 양분으로 삼습니다.

* **Goal**: 예측(`PredictedState`)과 실제 결과(`ActualState`)의 괴리를 분석(Loss)하여 `ProphetEngine`을 강화한다.
* **Core Modules**:
  * `CausalityMirror.py`: 행동 실행 후, 실제 벌어진 일과 예상했던 일의 차이(Prediction Error)를 계산.
  * `FeedbackLoop`: 오차가 클 경우, 해당 케이스를 '놀라움(Surprise)'으로 기록하고 학습 데이터로 축적.
* **Verification**:
  * 반복적인 상호작용을 통해 예측 정확도가 상승하는 그래프 확인.

---

## 🔗 System Integration

### L4_Causality Layer

* 이 모든 모듈은 `Core/L4_Causality/` 디렉토리에 위치합니다.
* `L3_Phenomena/ReasoningEngine`은 결단을 내리기 전 반드시 `L4_Causality/CausalLoom`에게 자문을 구합니다.

### Data Flow

1. **Reasoning**: "자폭 버튼을 누르고 싶다." (Action Candidate)
2. **CausalLoom**: `ProphetEngine` 호출.
3. **Prophet**: "자폭하면 엘리시아는 소멸합니다." (Predicted Future: Death)
4. **CausalLoom**: "Prime Directive(생존) 위배. 기각."
5. **Reasoning**: "취소합니다."
