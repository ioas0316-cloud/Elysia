# EXISTING_CONCEPTS.md
## ELYSIA Phase 0 — 개념별 구현 상태 및 진실성 목록
**작성일:** 2026-11-09  
**목적:** ELYSIA의 방대한 철학적·물리적 개념들이 실제 파이썬 코드에서 어느 수준으로 구현되어 있는지 진실만을 분류한다.

---

## 분류 기준
1. **[LIVING] 살아있는 개념:** 실제 수치적/물리적 연산으로 연속성이 유지되며 시스템 내에서 작동하는 개념.
2. **[ORPHAN] 고아 개념:** 올바른 원리로 구현되었으나, 상위 루프나 메인 시스템과 배선이 끊겨 잠들어 있는 개념.
3. **[SHELL] 껍데기/하드코딩 개념:** 이름과 주석은 거창하나, 내부 로직은 단어 일치, 고정된 상수 리터럴, 장식용 텍스트에 불과한 개념.
4. **[ISOLATED] 격리된 수학 모델:** 수학적 정밀도는 높으나 다른 인지/물리 모듈과 통신하지 않는 폐쇄적 개념.

---

## 1. 개념별 진단 목록

| 개념명 | 분류 | 위치 | 실제 구현 상태 및 비고 |
| :--- | :---: | :--- | :--- |
| **4대 연속성 (관계/연결/운동/정보)** | **LIVING** | `core/physics/causal_field.py` | 훅의 법칙 빔 긴장, 빔 파열/치유, Euler 위치 적분, rest_length 동적 적응으로 온전하게 작동함. |
| **색채적 삼원상 (Chromatic Triad)** | **LIVING** | `core/physics/causal_field.py` | Red(흐름 가속), Blue(질서/저항 감쇠), Yellow(엔트로피 노이즈)가 물리 적분 및 전위(Potential) 흐름에 직접 반영됨. |
| **비가역적 자폐 수렴 방지 (Anti-Solipsism)** | **LIVING** | `core/physics/causal_field.py` | `observe_external_stimulus()`에서 폐쇄 관성 회전 및 외부 정렬 텐서 보정이 유효하게 작동. |
| **텐션 기반 상태 재결정화 (State Recrystallization)** | **ORPHAN** | `core/physics/causal_dynamics.py` | `_recrystallize_states()`가 공간적 긴장을 내부 텐서 변화로 양방향 전이시키나, 메인 실행 루프에서 누락됨. |
| **상호 거울상 인식 (Mutual Mirroring Field)** | **ORPHAN** | `core/physics/universal_mutual_perception.py` | 노드 간 상호작용 역사에 따라 의미적 질량($M_s$)과 표면장력이 자라나는 훌륭한 원리이나, CausalField Voxel과 분리됨. |
| **역인과 관찰자 (Retrocausal BRS Observer)** | **ISOLATED** | `core/physics/retrocausal_observer.py` | HJB 안전 반경 및 Chern 곡률 차원 확장이 수학적으로 완결되어 있으나, 호출자가 0개임. |
| **기체-액체-얼음 위상 스케일 계층** | **ISOLATED** | `core/physics/scale_coupling_system.py` | Quadtree 결정화 전이는 정교하나 PyTorch 필수 의존성 및 2D Patch 전용 구조로 복셀 필드와 단절됨. |
| **현상학적 정의 공존 (Relational Definition)** | **LIVING** | `core/consciousness/relational_definition_engine.py` | 마찰에 따른 정의의 확장/수정/공존(Dual Coexistence)을 올바르게 기록함. 이전 의미를 파괴하지 않고 보존. |
| **위상적 불변량 추출기** | **SHELL** | `core/memory/enactive_relational_memory.py` | `_extract_topological_invariants()`가 입력과 상관없이 하드코딩된 불리언 3개를 무조건 반환함. |
| **목적론적 동형성 컴파일러** | **SHELL** | `core/physics/causal_field.py` | `TeleologicalCompiler.evaluate()`가 실제로는 소문자 공백 분리 단어의 단순 텍스트 교집합(Jaccard) 비율만 검사함. |
| **프랙탈 의도 루프 탈출 (Fractal Nudge)** | **SHELL** | `core/physics/causal_field.py` | `FractalEngramShell.report_resistance()`에서 하위 루프 탈출 시 상위 의도를 튕겨내는 벡터가 `[1.5, -2.0]`으로 고정 상수로 하드코딩됨. |
| **자기 관찰 인지 공리 (Cognitive Axioms)** | **SHELL** | `core/consciousness/cognitive_self_observation.py` | Sensing/Reasoning 등의 이동 수식 문자열은 실행되지 않는 장식용 텍스트이며, 판별은 임의 전달된 scalar dict의 코사인 유사도에 의존. |
| **4축 자기 경계 모델 (Self-Boundary)** | **ISOLATED** | `L2_cognitive_engine/self_boundary_tracker.py` | 내부기원/외부영향/무지/재수정 4축 정규화는 온전하나 호출자가 없어 휴면 상태. |

---

## 2. 결론 및 살려내야 할 척추 경로

수많은 개념 중 우리가 붙잡아야 할 **진짜 척추(Spine)**는 명확합니다:

```
[CausalField] (살아있는 물리 연속성)
      +
[CausalDynamicsEngine] (잠들어 있던 텐션-텐서 결합)
      +
[RelationalDefinitionEngine] (살아있는 마찰-의미 다차원 보존)
```

이 세 가지를 하나로 단단하게 묶고, 그 위에 껍데기(하드코딩)들을 걷어낸 뒤 **"고춧가루의 공변 원리(Covariance Loop)"**를 이식하는 것이 Phase 1의 유일하고도 확실한 과제입니다.
