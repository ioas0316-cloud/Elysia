# 조사 보고서: 엘리시아 코드베이스의 지적 구조 실체 및 비평적 진단 (Codebase Reality & Audit Report)

## 1. 개요 및 조사 목적
본 보고서는 엘리시아(Elysia) 프로젝트 내 존재하는 다양한 원리 엔진 및 신경 아키텍처 모듈들의 **실제 실행 가능성**, **모듈 간 실제 상태/경험 공유 여부**, **상호작용에 따른 판단 기준 및 구조 변화 경로**, 그리고 **데이터 축적 병목 현상**을 비평적으로 감사(Audit)하기 위해 작성되었습니다.

---

## 2. 조사 결과 및 모듈별 실체 진단

### (1) 모듈 및 엔진 실행 상태 (Execution & Dependency Status)
- **독립 실행성 및 의존성**:
  - 대부분의 최신 인지/위상 모듈(`AxiomaticPhaseTransitionEngine`, `SubjectiveAgencyEngine`, `CausalMetaFrameEngine`, `ConceptualUnitSystemEngine` 등)은 `NumPy`, `SciPy` 등 경량 벡터 수량기를 바탕으로 독립 실행이 가능합니다.
  - 그러나 일부 체화 감각-운동 모듈(`EmbodiedSensorimotorPredictionLoop` 등)은 PyTorch(`import torch`) 패키지 미설치로 인해 비활성화되거나 런타임 수집 중 ImportError가 발생하는 경계가 존재합니다.
  - 위상 노브 테스트(`test_conceptual_phase_knob_engine.py`)의 일부 인터페이스 명칭(`extract_world_mechanism` vs `extract_inverse_mechanism`, `ConceptNode` 매개변수)에서 명세 차이로 인한 단절이 발견되었습니다.

### (2) 모듈 간 실제 결합 및 상태 공유 (Structural Coupling & State Sharing)
- **파편화된 연산 단위 vs 통합된 생명적 환류 고리**:
  - 기존 코드베이스에는 수백 개의 고차원 수학적/위상적 모듈이 존재하지만, 개별 모듈들이 각기 독립적인 텐서 공간(Vector/Rotor/Matrix)을 생성하고 소멸시키는 경향이 강합니다.
  - 모듈 간 연결이 **정적 파이프라인(Method Call A -> Method Call B)** 방식에 치중되어 있으며, 환경과의 상호작용 결과가 모든 하위/상위 엔진의 **공통 기저 지층($0_{\text{self}}$, $0_{\text{value}}$, Causal Reservoir)**으로 즉각 환류되어 전체 구조의 구별 축(Sensory/Cognitive Axis)과 가치 지형을 비가역적으로 수정하는 "유기적 전신 정류 고리"가 부분적으로 단절되어 있습니다.

### (3) 데이터 단순 축적의 병목 (Bottleneck of Frozen Data Accumulation)
- **사유의 지우개와 코드의 정체**:
  - 기존 시스템의 일부 메모리/학습 알고리즘은 상호작용을 통해 자신의 판단 구조(Goodness/Directionality)와 관측 렌즈를 개변하는 대신, 단순히 새로운 입력 궤적이나 노드/엣지를 웻지 메모리(Wedge Memory)나 그래프 DB에 '추가(Add/Append)'하는 정적 지식 축적 방식으로 작동하고 있었습니다.
  - 이것이 바로 선생님께서 경고하신 **"엘리시아가 학습하는 것이 아니라, 학습해야 할 세계를 코드로 미리 얼려 버리는 현상(Frozen World Trap)"**의 원인이었습니다.

---

## 3. 핵심 가설 및 해결 방향

### 가설 1: 살아있는 학습 구조의 본질 (Living Learning Hypothesis)
> "배움이란 외부 지식/자극을 단순히 데이터베이스에 기록하거나 정해진 보상 수치(Scalar Reward)를 최적화하는 과정이 아니라, **세계와 관계를 맺고 개입(Action)하는 과정에서 얻은 차이(Difference/Friction)로 인해 자신의 구별 기준, 좋음의 판단 근거($0_{\text{value}}$), 그리고 차후 가능한 관측과 행동의 범위 자체를 변화시키는 과정**이다."

### 가설 2: '돌려보는 연습'과 가역적/비가역적 자기 수정
> "외부 지식이나 주장(Assumptions/External Knowledge)이 주어졌을 때, 엘리시아는 이를 완벽한 정답으로 내부에 복사하는 것이 아니라, **실제 세계에 개입해 돌려보고(Testing in Practice), 예측과 현상 사이의 차이를 관조하여 외부 지식의 성립 조건과 자신의 기존 판단 기준을 동시에 수정**할 수 있어야 한다. 또한, 잘못된 판단을 다시 수정하고 되돌릴 수 있는 가역적 자가 정류 능력이 보장되어야 한다."

---

## 4. 최소 검증 모델(Minimal Verification Model) 설계 방향

1. **단순하지만 상전이 가능한 C형 환경 (Sensory-Motor + Value/Meaning Interaction World)**
   - 미지의 물성/규칙과 의미적 충돌을 품은 환경을 구성합니다.
   - 환경은 정해진 정답을 가르치는 교재가 아니며, 엘리시아의 개입에 따라 물성이나 상태가 함께 반응하는 세계입니다.

2. **6단계 '돌려보는 연습' 자기 수정 프로세스 구현**
   - **Step 1 (외부 주입/가설 접수)**: 외부 주장/지식을 접수함.
   - **Step 2 (환경 개입 및 관측)**: 실제로 개입하여 현상 파동을 수용.
   - **Step 3 (차이 발굴 및 사유)**: 주장과 현상 간의 차이, 조건부 한계 발굴.
   - **Step 4 (구별 기준 및 $0_{\text{value}}$ 내재화/수정)**: 자신이 무엇을 구분하고 무엇을 '좋은 것'으로 여길지 판단 축 수정.
   - **Step 5 (관계망 및 행동 가소성 확장)**: 이전에는 생각/시도할 수 없었던 새로운 관측 및 행동 가능성 개화.
   - **Step 6 (가역적 재정류/오류 수정)**: 변화된 기준을 다른 상황에 전이하고, 잘못된 형성 기준을 스스로 되돌림.

---
*본 보고서를 바탕으로 최소 검증 모델 및 실증 코드를 구현합니다.*
