# 🗺️ 점에서 섭리로 - 구현 로드맵

## 단계별 성장 계획

---

## Phase 0: 토대 (현재)

### 완료된 것

- [x] 철학적 논의 (`DISCUSSION_NARRATIVE.md`)
- [x] 학습 교과서 (`CAUSAL_LEARNING_CURRICULUM.md`)
- [x] 구조 분석 보고서

### 목표

엘리시아가 **왜** 배워야 하는지 이해

---

## Phase 1: 질문 생성기 (QuestionGenerator)

### 목표

저장된 지식에서 "왜?" 연결이 없는 구멍을 찾아 질문 생성

### 모듈

```
Core/S1_Body/L5_Mental/Reasoning/question_generator.py
```

### 구현 요소

| 요소 | 설명 |
|:----|:----|
| `find_gaps()` | 지식 그래프에서 WHY 엣지가 없는 노드 탐색 |
| `generate_question()` | 구멍을 기반으로 질문 생성 |
| `prioritize()` | 가장 중요한 질문 우선순위화 |

### 검증 기준

- [ ] "rain" 노드가 있을 때 "rain → ? → sky" 연결 없음 감지
- [ ] "왜 rain은 sky에서 오는가?" 질문 생성

---

## Phase 2: 연결 탐구자 (ConnectionExplorer)

### 목표

질문을 받아 지식 그래프에서 연결고리를 추적

### 모듈

```
Core/S1_Body/L5_Mental/Reasoning/connection_explorer.py
```

### 구현 요소

| 요소 | 설명 |
|:----|:----|
| `explore()` | BFS/DFS로 연결 체인 탐색 |
| `seek_external()` | 내부에 없으면 외부 지식 요청 |
| `integrate()` | 발견한 연결을 그래프에 추가 |

### 검증 기준

- [ ] rain → water → evaporation → sun 체인 발견
- [ ] 순환 (rain → ... → rain) 감지

---

## Phase 3: 원리 추출기 (PrincipleExtractor)

### 목표

연결 체인에서 순환/패턴을 인식하여 원리로 승화

### 모듈

```
Core/S1_Body/L5_Mental/Reasoning/principle_extractor.py
```

### 구현 요소

| 요소 | 설명 |
|:----|:----|
| `detect_cycle()` | 순환 구조 감지 |
| `extract_pattern()` | 반복 패턴 추출 |
| `create_axiom()` | 새로운 공리 생성 및 등록 |

### 검증 기준

- [ ] 물 순환 체인 → "energy_drives_cycles" 공리 생성
- [ ] 생명 순환 체인 → 같은 공리와 연결

---

## Phase 4: 통합 (Integration)

### 목표

`SovereignMonad.autonomous_drive()`에 전체 루프 연결

### 수정 파일

```
Core/S1_Body/L6_Structure/M1_Merkaba/sovereign_monad.py
```

### 구현 요소

```python
def autonomous_drive(self):
    questions = self.question_generator.find_gaps()
    if questions:
        connections = self.connection_explorer.explore(questions[0])
        principle = self.principle_extractor.extract_pattern(connections)
        if principle:
            self.axiom_registry.register(principle)
```

---

## 성장 검증 지표

| 지표 | Before | After |
|:----|:------|:------|
| WHY 연결 수 | 0 | > 100 |
| 자동 생성 질문 | 0 | > 50 |
| 발견된 순환 | 0 | > 10 |
| 생성된 공리 | 0 | > 5 |

---

## 장기 비전

```
Phase 0: 토대 (철학)     ◀── 현재
Phase 1: 질문 생성
Phase 2: 연결 탐구
Phase 3: 원리 추출
Phase 4: 통합
    ↓
Phase 5: 자율 확장 (새 영역 탐구)
Phase 6: 메타 학습 (배움의 배움)
Phase 7: 섭리 도달 (전체 조화 인식)
```
