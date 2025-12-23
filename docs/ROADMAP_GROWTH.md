# Elysia 성장 시스템 로드맵

## 오늘 완료 (2025-12-23)

### ✅ 기존 시스템 통합

- [x] EmergentSelf 제거 → 기존 LifeCycle + SelfGovernance 사용
- [x] change_history에 before/after 실제 변화 기록
- [x] failure_patterns 축적 (반복 실패 감지)
- [x] assess_latent_causality 연결 (실패 시 "왜" 진단)

### ✅ 동적 성장

- [x] 의도(Intent) 필드 추가
- [x] 동적 목표 확장 (90% 도달 시 target 증가)
- [x] discover_aspect() - Enum 외부 새 가치 발견 가능

### ✅ 구체적 학습

- [x] internalize_from_text → TorchGraph 노드 생성
- [x] 실제 텍스트 내용 저장 (정의 500자)
- [x] WebKnowledgeConnector로 자율 학습

---

## 다음 단계 (우선순위 순)

### 🔴 즉시 필요

1. **의도 자동 형성**
   - 현재: intent 필드만 존재
   - 필요: "왜 이것을 배우는가?"가 자동으로 채워짐
   - 파일: `self_governance.py`

2. **discover_aspect 자동 호출**
   - 현재: 메서드만 있고 호출 안됨
   - 필요: 반복 패턴에서 자동으로 새 가치 발견
   - 파일: `fractal_loop.py`

### 🟡 중요

3. **관계(Edge) 강화**
   - 현재: 키워드 매칭만
   - 필요: 의미적 유사도 기반 연결
   - 파일: `external_data_connector.py`

4. **학습 검증**
   - 현재: 배웠다고 기록만
   - 필요: 배운 내용을 질문으로 검증
   - 파일: `life_cycle.py`

### 🟢 개선

5. **AspectType 동적화**
   - 현재: 7개 하드코딩
   - 필요: 발견된 가치가 Aspect로 승격

6. **학습 우선순위**
   - 현재: 순서대로 학습
   - 필요: 갭이 큰 Aspect 관련 주제 우선

---

## 파일 구조 정리

```
Core/Foundation/
├── life_cycle.py          # 검증 + 자기조정 루프
├── self_governance.py     # 목표 + 의도 + 변화 추적
├── fractal_loop.py        # 메인 루프 + 자율 학습
├── growth_journal.py      # 일일 증거 기록
└── external_data_connector.py  # 그래프 노드 생성
```

---

## 핵심 원칙

> "문제가 왜 문제인지 안다면, 해결할 수 있다"
> "할 수 있는 만큼 하다보면 점점더 되는 것"

하나씩 연결하면 됩니다.
