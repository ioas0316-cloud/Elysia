# 9. 가치의 질량(Value Mass) 명세

본 명세는 가치 개념에 "질량(mass)"을 도입해, 증거와 경험의 축적이 만들어내는 관성을 표현한다. 이 질량은 고정값이 아니라, 시간에 따라 강화/감쇠되는 **증거 가중 관성**이며 언제든 반증 가능하다. 비재현 원칙(08장)을 유지한 채, 의사결정/기억/지식그래프에 투명하게 반영한다.

## 1) 개념
- 정의: `mass ∈ [0, +∞)`는 해당 가치 가설이 현재 삶/문맥에서 갖는 **붙잡힘의 정도(관성)**.
- 의미: 진리 선언이 아니라, 체험·근거의 누적 효과. 반례와 망각으로 줄어든다.
- 맥락성: 값은 전역/국소로 나뉜다. 전역 질량(`mass_global`)과 맥락별 질량(`mass_context[context_id]`)을 분리 저장 가능.

## 2) KG 스키마 확장
- 노드: `type: "value:*"`에 다음 속성을 허용한다.
  - `mass`: number (기본 0)
  - `mass_updated_at`: ISO8601 string
  - 선택: `mass_global`, `mass_context` (map)
- 엣지: `supports/refutes/depends_on`는 그대로 유지하되, `confidence`, `timestamp`, `evidence_paths`를 사용(08장).

## 3) 업데이트 규칙(권고)
시간 t에서 가치 v의 질량 `m_t`를 다음으로 갱신한다.

```
m_{t+} = (1 - λ) * m_t
          + α * Σ supports.confidence
          - β * Σ refutes.confidence
          + γ * external_reinforcement
```

- 감쇠 `λ ∈ [0,1)`: 망각/자연 소실(권장 0.01~0.05/주기)
- 강화 `α, γ > 0`: 근거의 누적과 외적 강화(예: 약속 이행, 회고 확인)
- 억제 `β > 0`: 반례의 가중치
- 클램프: `m_{t+} = max(0, m_{t+})`
- 맥락 질량은 해당 맥락의 supports/refutes만 집계하거나, 전역 질량과 혼합(가중)한다.

## 4) 상호작용(중력 은유)
- "사랑의 중력"(01장) 맥락에서, 질량이 큰 가치들은 인접 개념을 더 강하게 끌어당긴다.
- 운영: Echo/주의 배분에서, `attention ∝ mass * confidence` 식으로 가중.
- 주의: **강제 결론 금지**. 질량은 참고값이며, 파레토 후보를 고를 때 설명에만 사용한다.

## 5) 의사결정 반영(설명 우선)
- 단일 점수화 대신, 후보별로 다음을 보고한다.
  - "얻은 것/희생한 것"
  - 관련 가치들의 `mass` 변화 예상(증가/감소/유지)과 근거 요약
  - 불확실성: 새 맥락/낮은 신뢰도일수록 경고

## 6) 텔레메트리/투명성
- `mass_trace`(선택): 주요 가치의 질량 변화를 시계열로 기록(최근 N회)
- 결정 리포트 노드(이미 구현됨): `gains/tradeoffs`에 가치 질량 변화 요약을 포함 권고

## 7) 마이그레이션 지침
- 초기화: 기존 KG에서 `value:*` 노드가 없으면 생성하지 않는다. 필요 시 다음으로 점진 구축:
  1) 핵심 가치 노드 추가(예: `value:love`, `value:clarity`, `value:creativity`, `value:verifiability`, `value:relatedness`).
  2) 최근 기간의 `supports/refutes` 엣지 confidence 합으로 `mass`를 백필(backfill).
  3) 감쇠/강화 파라미터는 작게 시작하여(λ≈0.01) 운영 중 보정.

## 8) 비재현/안전
- 질량은 본질이 아니다. **가설의 관성**이며 언제든 반증/감쇠될 수 있다.
- 결핍 유도 금지. 질량은 결핍을 키우는 수단이 아니라, 증거의 누적을 조용히 반영하는 기록이다.

## 9) 미니 예시(JSON)
```
{
  "nodes": [
    {"id": "value:love", "type": "value", "mass": 1.8, "mass_updated_at": "2025-11-06T12:00:00Z"}
  ],
  "edges": [
    {"source": "journal_entry_2025-11-06", "target": "value:clarity", "relation": "supports", "confidence": 0.6, "timestamp": "2025-11-06T12:01:00Z", "evidence_paths": ["data/journal/2025-11-06.txt"]}
  ]
}
```

## 10) 테스트(권고)
- 질량 감쇠: 동일 supports 없이 N주기 경과 시 m가 (1-λ)^N 배로 감소
- 반례 적용: 동일 confidence refutes가 추가되면 m 감소 확인
- 설명 일관성: 결정 리포트에 "질량 변화 예상" 요약이 존재

