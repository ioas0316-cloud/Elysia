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

### ✅ 즉시 필요 (완료 - 2025-12-24)

1. **의도 자동 형성** ✅
   - 완료: `_auto_generate_intent()` 메서드 추가
   - 갭 분석 기반으로 "왜 이것을 배우는가?" 자동 생성
   - 파일: `self_governance.py`

2. **discover_aspect 자동 호출** ✅
   - 완료: `_track_pattern_and_discover()` 메서드 추가
   - 3번 반복 시 자동으로 새 가치 발견
   - 파일: `fractal_loop.py`

### ✅ 중요 (완료 - 2025-12-24)

1. **관계(Edge) 강화** ✅
   - 완료: `_find_semantic_neighbors()` + `_cosine_similarity()` 추가
   - TinyBrain embedding 기반 의미적 유사도 연결
   - 파일: `external_data_connector.py`

2. **학습 검증** ✅
   - 완료: `verify_learning()` 메서드 추가
   - 학습 후 질문 생성 → 키워드 매칭 → 이해도 점수
   - 파일: `life_cycle.py`, `fractal_loop.py`

### ✅ 개선 (완료 - 2025-12-24)

1. **AspectType 동적화** ✅
   - 완료: `promote_to_core_aspect()`, `check_promotions()` 추가
   - 발견된 가치가 50% 달성 + 3회 강화 시 핵심 가치로 승격

2. **학습 우선순위** ✅
   - 완료: `_prioritize_learning_queue()` 추가
   - 갭이 큰 Aspect 관련 주제를 우선 학습

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

---

## 🌄 2025-12-24 논의 기록: 동적 지식 지형

### 완료된 구현

1. **동적 지식 지형** ✅
   - `LightUniverse.interfere_with_all()` - 새 지식이 기존과 간섭
   - `absorb_with_terrain()` - 흡수 + 지형 효과 반환
   - `_auto_select_scale()` - 자율적 줌인/줌아웃 (자유의지)
   - 파일: `light_spectrum.py`

2. **문제 해결 학습** ✅
   - `learn_with_purpose(problem, goal)` - 목적 기반 학습
   - 파일: `web_knowledge_connector.py`

### 핵심 통찰 (다음 개발에 적용)

> **"지식이란 유기체의 사고흐름 자체다"**
>
> - 정적 데이터 ❌ → 동적 파동 ✅
> - 저장 ❌ → 지형 변화 ✅
> - 학습 ❌ → 사고 방식 진화 ✅

> **"어텐션 = 패턴 속에서 패턴 선택 = 언어적 사고의 원리"**
>
> - 확장 공간 (LightUniverse) ✅ 있음
> - 선택/집중 (resonate/interfere) ✅ 있음
> - 선형화 → 언어 ← 강화 필요

---

## 🔴 다음 로드맵 (우선순위)

### 1. 복잡한 검증 프레임워크 🔴

- [ ] 시간에 걸친 변화 추적 (단일 호출 X, 연속 학습 후 비교)
- [ ] 예측-검증 루프 (학습 원리로 예측 → 실제와 비교)
- [ ] 메타 지표 (학습 효율이 시간에 따라 향상되는가?)

### 2. 언어적 사고 흐름 강화 🟡

- [ ] 수치 중심 → 서사/은유 중심 표현
- [ ] `CausalNarrativeEngine` 활용 강화
- [ ] resonate → 언어 출력 연결 개선

### 3. 선형화 파이프라인 🟡

- [ ] 확장된 그래프 → 어텐션(선택) → 언어 출력
- [ ] 사고 과정의 언어적 표현

### 4. 진화 방향 (장기) 🟢

- 식물 수준 (반응적) ← 현재
- 동물 수준 (능동적, 예측 기반)
- 인간 수준 (추상적, 메타 인지, 상상)

---

## 참고 문서

- [VISION_LIVING_KNOWLEDGE.md](VISION_LIVING_KNOWLEDGE.md) - 핵심 철학
- [SYSTEM_MAP.md](SYSTEM_MAP.md) - 시스템 구조
