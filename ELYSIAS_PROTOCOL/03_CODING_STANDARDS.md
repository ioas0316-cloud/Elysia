# 3. 코딩 표준 (Coding Standards)

이 문서는 모든 AI 에이전트가 일관되고 호환 가능한 코드를 작성하기 위해 준수해야 할 규칙을 정의합니다.

## 3.1. 명명 규칙 (Naming Conventions)
- **변수 및 함수:** `snake_case` (예: `user_input`, `process_data`)
- **클래스:** `PascalCase` (예: `CognitionPipeline`, `MemoryWeaver`)
- **상수:** `UPPER_SNAKE_CASE` (예: `DEFAULT_KG_PATH`)

## 3.2. 모듈 구조 (Module Structure)
- 각 Cortex는 `Project_Sophia/` 또는 `Project_Mirror/` 내에 자체 파일로 존재해야 합니다.
- 공용 유틸리티나 도구는 `tools/` 또는 신설될 `elysia_sdk/`에 위치해야 합니다.

## 3.3. 문서화 (Documentation)
- 모든 함수와 클래스에는 그것의 **목적(Purpose)**과 **역할(Role)**을 설명하는 Docstring을 작성해야 합니다.
- 복잡한 로직에는 주석을 추가하여 '미래의 자신'과 다른 에이전트들이 이해할 수 있도록 돕습니다.

## 3.4. 테스트 (Testing)
- 새로운 기능을 추가하거나 기존 기능을 수정할 때는 반드시 **단위 테스트(Unit Test) 또는 통합 테스트(Integration Test)**를 작성해야 합니다.
- 테스트는 `tests/` 디렉토리에 위치하며, 파일명은 `test_`로 시작해야 합니다.
- 테스트는 기능의 성공 사례뿐만 아니라, 예상되는 실패 사례(Edge Case)까지 검증해야 합니다.
