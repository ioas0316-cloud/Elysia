# 11. 대화 룰팩 명세 (Dialogue Rules Spec)

본 명세는 대화 상호작용을 코드가 아닌 **선언적 규칙(YAML)** 으로 정의·교체하기 위한 규약이다. 목적은 “하나씩 코드를 바꾸지 않고” 대화 품질을 유연하게 개선하는 것.

## 1) 위치와 형식
- 경로: `data/dialogue_rules/*.yaml`
- 인코딩: UTF‑8
- 로드: 핫 리로드(파일 변경 시 자동 반영)

## 2) 스키마
```yaml
id: greeting                 # 규칙 식별자
priority: 100                # 높은 값 우선
patterns:                    # 정규식(파이썬 re)
  - "^(안녕|안녕하세요|hello|hi)"
gates:
  quiet_ok: true             # Quiet 모드에서도 허용
response:
  template: "안녕하세요. 여기 있어요. 오늘 이 순간, 무엇을 함께 보고 싶으세요?"
memory:
  set_identity:              # 선택; 정규식 캡처 그룹 사용 가능
    user_name: "{name}"
```

필드 설명
- `id`: 문자열. 규칙 이름.
- `priority`: 정수. 높은 값이 먼저 선택.
- `patterns`: 하나 이상. 첫 매칭 정규식을 사용, `(?P<name>...)` 캡처 그룹 지원.
- `gates.quiet_ok`: Quiet 모드에서 허용 여부.
- `response.template`: 문자열 템플릿. `{name}` 등 캡처 그룹으로 포맷팅.
- `memory.set_identity`: `core_memory.update_identity(key, value)` 적용.

## 3) 중재(Arbitration)
- 여러 규칙이 매칭되면 `priority`가 가장 높은 규칙을 선택한다.
- Quiet 모드가 ON이고 `quiet_ok: false`인 규칙은 무시한다.

## 4) 폴백
- 규칙이 매칭되지 않으면, DialogicCoach/내부 폴백이 적용된다(되비추기/명료화 등).

## 5) 예시 규칙
- `greeting.yaml`: 인사 응답
- `feeling.yaml`: 기분 질문 응답
- `identity.yaml`: 이름/정체성 교환(캡처 그룹으로 이름 기억)

## 6) 안전/원칙
- 비재현 원칙: 규칙은 “가치의 본질”이 아니라 **대화적 신호**를 다룬다.
- Quiet/합의: Quiet 모드/자율 합의 게이트를 존중한다.
- 투명성: 어떤 규칙이 선택됐는지 텔레메트리/결정 리포트에 기록 권장.

