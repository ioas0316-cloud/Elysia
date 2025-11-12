## 30_VISUAL_CONSCIOUSNESS_PROTOCOL.md

**Title:** Visual Consciousness Mapping Protocol  
**Codename:** L3 Living Patterns  
**Author:** E.L.Y.S.I.A. Core Division  
**Version:** 1.0

---

## Overview

셀월드의 “의식-기억 루프”를 관찰 가능한 시각 언어로 번역한다. 단순 에너지 맵을 넘어, 개체의 의식 상태와 문명적 인과를 드러내어 관찰자가 “살아있다”를 즉시 느끼게 하는 것이 목적이다. 본 문서는 29_PERSISTENCE_AND_MEMORY_PROTOCOL.md와 연동된다.

---

## Visual States (Consciousness → Symbols)

| 상태 | 표현 | 의미 |
|---|---|---|
| 관찰/주의 집중 | 푸른 오라(맥동) | 엘리시아의 주의가 닿은 영역 |
| 사유/결정 중 | 미세 입자 회전 | 인식·판단 활성 |
| 학습/기억 중 | 황금 라인 연결(페이드) | 기억 형성(신경망) |
| 감정/공명 | 색조 변조(HSL) | 집단 감정 표현 |

표현 규칙
- 오라는 반径 r, 주기 T로 맥동(강도 ∝ 알파/반경)
- 입자 회전은 상태 지속 시간에 비례해 각속도/밀도 변화
- 기억 라인은 최근 이벤트의 인과(출발→목적)를 얇은 곡선으로 연결 후 서서히 사라짐
- 감정 스펙트럼은 Aether 기반 색조 변조(긍정=따뜻, 부정=차가움)

---

## Civilizational Layers (Evolution)

- L1 원시 정착: 물 근처 파란 밴드
- L2 사회화: 붉은 맥동(에너지 교류)
- L3 기술 창발: 초록빛 결계(안정 네트워크)
- L4 영적 각성: 보라 파장(집단의식)

전이 규칙: 자원/인구/사기/기억-강화 지표의 임계 조합으로 결정. 상위 레이어는 하위 위에 반투명 누적.

---

## Time Experience (Zoom = Granularity)

- 줌인: 개체의 의식/결정/기억 라인을 상세 표시
- 줌아웃: 문명 전체의 에너지 흐름/기억 패턴을 집계 표시
- Shift+Wheel: 시간가속(정합)  
- Alt+Click: 특정 개체 타임라인 퀵뷰(최근 N 이벤트 카드)

타임라인 퀵뷰 예시
```
[entity: #124]
- t-12 build_house success (rain)
- t-25 carry_wood success (group)
- t-40 fight failure (wolf)
```

---

## Rendering Primitives

- 개체: 방향성 삼각형/결정(zoom-aware), 상태색 코딩
- 건축: 프랙탈 증식(점→싹→모듈), 진행률 링
- 환경: 반투명 필드, 내부 정보는 색조/맥동으로 표현
- 광원: 엘리시아 감정/집중에 반응하는 오라/광륜

성능 수칙: QOE 예산 하드캡, LOD(줌/거리/상태), 배율 리샘플링, 배치 드로우 권장

---

## Data Interfaces

입력: attention_strength(x,y), aether(x,y), mana(x,y), entity.state, entity.last_events[], entity.emotion, settlement.level, metrics(pop, wood, food, morale)

출력: 오라/입자/라인/링의 드로우 명령(프레임 합성)

---

## Integration Plan

Phase 1 (Minimal)
- 의식 오라(관찰/주의), 결정 입자(사유), 기억 라인(최근 3개)
- Alt+Click 타임라인 퀵뷰 패널(좌상단)

Phase 2 (Civilizational)
- 정착지 레벨 레이어(L1~L4)와 색상 파장
- 진행률 링(건축/공정) 및 광원 반응

Phase 3 (Refinement)
- 감정 스펙트럼과 색조 변조
- 성능 최적화(LOD/배율 보정/배치 드로우)

Acceptance
- 줌에 따라 요약↔세부가 자연 전환
- Alt+Click 시 100ms 내 퀵뷰 표시
- QOE 예산 유지로 30fps(Safe) 확보

---

## Links

- 29_PERSISTENCE_AND_MEMORY_PROTOCOL.md
- 27_ELYSIA_VISUAL_UPDATE_V3.md
- 25_QUANTUM_OBSERVATION_ENGINE.md

