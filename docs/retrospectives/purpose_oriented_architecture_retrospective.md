# 개발 회고: 목적 중심 아키텍처 도입

**날짜:** 2025년 11월 4일
**작성자:** Jules (AI Software Engineer)
**관련 작업:** [#issue-number] 3차원 지식그래프 인과데이터 확장 및 아키텍처 개선

---

## 1. 문제 정의 및 목표

초기 논의는 3차원 지식그래프(사고공간)의 인과관계 데이터 확장에서 시작되었습니다. 하지만 창조주님과의 깊이 있는 대화를 통해, 단순히 데이터를 확장하는 것을 넘어 엘리시아의 근본적인 사고방식을 개선해야 한다는 더 큰 목표에 도달했습니다.

기존의 '목표 기반(Goal-Oriented)' 아키텍처는 주어진 단일 작업을 수행하는 데는 효율적이었지만, "왜 이 일을 하는가?"라는 더 높은 차원의 맥락을 이해하지 못했습니다. 이로 인해 엘리시아의 행동은 단편적이고 수동적일 수밖에 없었습니다.

**최종 목표:** 엘리시아를 단순한 '명령 실행자'에서, 창조주님의 '목적(Purpose)'을 이해하고 그에 맞춰 스스로 전략을 수립하고 행동하는 '전략적 파트너'로 성장시킨다.

## 2. 해결책: 목적 중심 아키텍처 (Purpose-Oriented Architecture)

이 목표를 달성하기 위해, 엘리시아의 '뇌'에 해당하는 `CognitionPipeline`에 새로운 사고 회로를 도입했습니다. 이 아키텍처는 다음과 같은 핵심 모듈(Cortex)들로 구성됩니다.

### 2.1. 핵심 모듈

1.  **`IntentAnalysisCortex` (의도분석 Cortex):**
    *   **역할:** 창조주님의 자연어 '목적' 선언(예: "엘리시아의 추론 능력을 강화하고 싶어")을 기계가 이해할 수 있는 구조화된 '목표 객체(Goal Object)'로 변환합니다.
    *   **구현:** Gemini API를 활용하여, 입력된 텍스트에서 목표의 핵심 `description`, `type`, 그리고 `parameters`를 추출합니다.

2.  **`StrategicCortex` (전략 Cortex):**
    *   **역할:** '목표 객체'를 입력받아, 엘리시아의 최상위 가치(예: 성장, 관계 형성)와 연결하고, 이를 달성하기 위한 대략적인 전략 로드맵을 수립합니다.
    *   **구현:** 목표의 `type`에 따라 미리 정의된 최상위 목적(`related_purpose`)과 우선순위(`priority`)를 할당하는 규칙 기반 시스템으로 구현되었습니다. 향후에는 지식그래프의 상태와 과거 경험을 고려하여 더 동적인 전략을 수립하도록 고도화될 수 있습니다.

3.  **`GoalDecompositionCortex` (목표분해 Cortex):**
    *   **역할:** '전략'을 `ToolExecutor`가 즉시 실행할 수 있는 구체적인 단계별 '실행 계획(Execution Plan)'으로 분해합니다.
    *   **구현:** 목표 객체의 `type`과 `parameters`를 기반으로, 어떤 도구(`tool_name`)를 어떤 인자(`parameters`)로 호출해야 하는지를 정의하는 규칙 기반 로직을 사용합니다.

### 2.2. 데이터 흐름

새로운 아키텍처의 데이터 흐름은 다음과 같습니다.

1.  **입력:** 사용자가 `"목적: ..."` 형태의 메시지를 입력합니다.
2.  **`CognitionPipeline`:** `purpose_prefix`를 감지하고, `process_purpose_oriented_message` 메서드를 호출합니다.
3.  **`IntentAnalysisCortex`:** 자연어 목적을 구조화된 `goal_object`로 변환합니다.
    *   `{ "description": "Enhance reasoning", "type": "ENHANCE_CAPABILITY", ... }`
4.  **`StrategicCortex`:** `goal_object`를 받아 `strategic_roadmap`을 생성합니다.
    *   `{ "related_purpose": "CORE_GROWTH", "goals_in_order": [...] }`
5.  **`GoalDecompositionCortex`:** `strategic_roadmap`의 최우선 목표를 `execution_plan`으로 분해합니다.
    *   `{ "plan_id": ..., "steps": [{"tool_name": "google_search", ...}] }`
6.  **`ToolExecutor`:** `CognitionPipeline`이 `execution_plan`의 각 단계를 `ToolExecutor`에 전달하여 실행합니다.
7.  **성찰 (Reflection):** 실행 결과를 `Memory` 객체로 포장하여 `CoreMemory`에 저장합니다. 이를 통해 엘리시아는 자신의 행동 결과를 '경험'으로 학습합니다.
8.  **출력:** 모든 과정의 최종 결과를 요약하여 사용자에게 보고합니다.

## 3. 다른 AI 에이전트를 위한 가이드

이 아키텍처는 엘리시아의 자율성과 전략적 사고 능력을 확장하기 위한 중요한 기반입니다. 다른 AI 에이전트(Codex, Gemini Code Assistant 등)가 이 프로젝트에 기여할 때 다음 사항을 고려해주시기 바랍니다.

*   **Cortex 모듈 확장:** 각 Cortex는 현재 규칙 기반 또는 단일 LLM 호출로 구현되어 있습니다. 각 모듈의 역할을 명확히 이해하고, 더 정교한 로직(예: 과거 경험 참조, 지식그래프 검색)을 추가하여 고도화할 수 있습니다.
*   **새로운 도구(`Tool`) 추가:** `GoalDecompositionCortex`가 사용할 수 있는 새로운 도구를 `ToolExecutor`에 추가하고, 관련 분해 로직을 `GoalDecompositionCortex`에 구현하여 엘리시아의 행동 범위를 넓힐 수 있습니다.
*   **성찰 메커니즘 고도화:** 현재는 단순히 실행 결과를 기록하는 수준입니다. 계획의 성공/실패 여부를 더 정교하게 판단하고, 실패 원인을 분석하여 다음 계획 수립에 반영하는 '진정한 학습' 메커니즘을 구현하는 것이 중요합니다.
*   **철학적 일관성:** 모든 변경 사항은 `AGENTS.md`에 기술된 프로젝트의 핵심 철학(성장, 자율성, 지혜)과 일관성을 유지해야 합니다.

## 4. 결론 및 향후 과제

이번 작업을 통해 엘리시아는 단순한 응답 생성기를 넘어, 목적을 가지고 스스로 생각하고 행동하는 주체로 나아갈 수 있는 중요한 아키텍처 기반을 마련했습니다.

**향후 과제:**
*   각 Cortex 모듈의 LLM 프롬프트 및 규칙 기반 로직 고도화
*   실행 계획의 동적인 수정 및 재수립 기능 추가
*   더 정교한 성찰 및 학습 메커니즘 구현
*   다양한 도구 추가 및 `GoalDecompositionCortex`와의 연동

이번 회고가 엘리시아 프로젝트에 참여하는 모든 개발자에게 명확한 가이드가 되기를 바랍니다.
