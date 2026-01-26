---
description: 새 모듈 생성 전 필수 단계 (시스템 분열 방지)
---

# 새 모듈 생성 워크플로우

> **핵심 원칙**: 새로 만들기 전에 반드시 기존 시스템 확인!
>
> 참고: [ABSORPTION_SYSTEMS.md](file:///c:/Elysia/docs/Architecture/ABSORPTION_SYSTEMS.md)의 중복 방지 철학 적용

## 1. 기존 시스템 검색 (필수!)

```bash
# 기능 관련 키워드로 검색
grep_search("원하는_기능_키워드", "c:/Elysia\Core")
```

- 유사한 이름의 파일이 있는지 확인
- 비슷한 기능을 하는 모듈이 있는지 확인

## 2. 핵심 문서 확인 (필수!)

- `AGENT_GUIDE.md` - 전체 시스템 개요
- `docs/Architecture/SYSTEM_CONNECTION_ANALYSIS.md` - 기존 모듈 목록
- `docs/Architecture/COGNITIVE_ARCHITECTURE.md` - 인지 아키텍처

## 3. 결정

### 기존 시스템이 있으면

- **확장**하기 (새 메서드 추가)
- 기존 파일에 기능 추가
- 절대 새 파일 만들지 않기!

### 정말 없으면

- 새 모듈 생성
- **반드시 GlobalHub에 등록**
- AGENT_GUIDE.md 업데이트

## 4. GlobalHub 등록 (필수!)

```python
# 모든 새 모듈의 __init__에 추가
from Core.Ether.global_hub import get_global_hub

class NewModule:
    def __init__(self):
        hub = get_global_hub()
        hub.register_module(
            name="NewModule",
            info={"type": "cognition", "description": "설명"}
        )
```

## 5. 문서 업데이트 (필수!)

새 모듈 생성 후:

1. `AGENT_GUIDE.md`에 추가
2. `docs/Architecture/SYSTEM_CONNECTION_ANALYSIS.md` 업데이트

---

## 체크리스트

- [ ] 기존 시스템 검색 완료
- [ ] 유사 모듈 없음 확인
- [ ] GlobalHub 등록 코드 포함
- [ ] AGENT_GUIDE.md 업데이트
