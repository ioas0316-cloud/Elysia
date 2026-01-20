# 📏 Coding Standards (코딩 표준)

---

## 🐍 Python 규칙

| 항목 | 규칙 |
| :--- | :--- |
| **버전** | 3.10+ |
| **스타일** | PEP 8 |
| **타입 힌트** | 권장 |
| **Docstring** | Google 스타일 |
| **최대 줄 길이** | 100자 |

---

## 📝 파일 헤더

모든 `.py` 파일 상단에 목적 기술:

```python
"""
Module Name (한글명)
====================
경로: Core/...

"철학적 한 줄"

이 모듈의 목적과 역할을 설명합니다.
"""
```

---

## 🏷️ 네이밍 규칙

| 유형 | 규칙 | 예시 |
| :--- | :--- | :--- |
| 클래스 | PascalCase | `MetalRotorBridge` |
| 함수 | snake_case | `sync_to_device` |
| 상수 | UPPER_SNAKE | `MAX_CAPACITY` |
| 파일 | snake_case | `metal_rotor_bridge.py` |
| 폴더 | PascalCase | `Foundation/Nature` |

---

## 🚫 금지 사항

| 금지 | 이유 |
| :--- | :--- |
| `Util/`, `Misc/` 폴더 | 소속 불명 코드 방지 |
| 전역 상태 남용 | 사이드 이펙트 최소화 |
| 하드코딩 경로 | 이식성 저하 |
| `print()` 디버깅 | `logging` 사용 |

---

## 🧪 테스트

- **위치**: `Core/tests/`
- **네이밍**: `test_*.py` 또는 `*_test.py`
- **실행**: `pytest Core/tests/`
- **커버리지**: 핵심 모듈 80% 이상 권장

---

## 📋 PR 체크리스트

- [ ] PEP 8 준수
- [ ] 타입 힌트 추가
- [ ] Docstring 작성
- [ ] 테스트 통과
- [ ] 관련 문서 업데이트
