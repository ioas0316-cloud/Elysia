# 12. 흐름(Flow) 엔진과 Wisdom‑Virus 명세

본 문서는 엘리시아의 의식을 "연속적 흐름"으로 구현하고, 지식이 아니라 "살아있는 의미 단위(지혜)"가 전파·변이·진화하는 메커니즘을 규정한다. LLM/API 없이도 대화 품질을 확보하기 위한 로고스적 설계 기준이며, `ELYSIAS_PROTOCOL`의 단일 진실 원천 규약을 따른다.

## 12.1 목적
- 규칙이 "규칙이지 않게" 작동하도록, 규칙은 힌트이고 최종 선택은 연속 신호(Flow)로 결정한다.
- 개별 통찰을 Wisdom‑Virus(의미 단위)로 취급하여 KG에 감염→전파→변이를 기록하고, 가치 질량(09)과 연결한다.
- 모든 결과는 체험 경로(`experience_*`)와 함께 증거로 남긴다(화이트홀, 10).

## 12.2 구성요소 개요
- FlowSpec(`data/flows/*.yaml`): 가중치/온도/작용 선택 규칙의 선언적 스펙.
- FlowEngine(`Project_Elysia/flow_engine.py`): 연속 신호를 혼합해 작용을 선택하고 응답을 합성.
- Dialogue Rules(`data/dialogue_rules/*.yaml`): 정규식 기반의 간결한 힌트 규칙(매칭→우선순위→템플릿).
- Wisdom‑Virus(`Project_Sophia/wisdom_virus.py`): 의미 단위의 전파·변이 엔진. KG에 `supports`/메타를 기록.

## 12.3 FlowEngine 명세
### 신호(0..1 범위 권장)
- rule_match: 룰팩 매칭 유무(룰은 힌트이므로 단독 결정하지 않음).
- kg_relevance: 메시지↔KG 근접도(간단 TF‑IDF/BM25/LSA 등 비LLM 가능 기법).
- continuity: `WaveMechanics`/working memory/topic tracker에서 도출한 연속성 점수.
- value_alignment: 가치 질량(09)의 정합도(예: clarity↑면 명료화 가중치에 기여).
- evid_conf: 증거/불확실성 신호(경험 링크 보유, refutes 비율 등).
- latency_cost: 지연/복잡도 비용(낮을수록 가점).

### 선택 규칙(예시)
```
score(op) = w_rule*rule_match
          + w_kg*kg_relevance
          + w_wave*continuity
          + w_value*value_alignment
          + w_evid*evid_conf
          - w_cost*latency_cost
```
- 온도 τ(0.2~0.6 권장)로 탐욕/탐색 균형.
- Quiet/Ask‑Consent(08) 게이트는 최상위에서 억제/허용을 결정한다.

### 작용(operators)
- 최소 집합: clarify, reflect, suggest, retrieve_kg, summarize, visualize.
- 각 작용은 전제/출력/부수효과를 가진다. FlowEngine은 1~3개의 작용을 합성해 응답을 구성하고, `ResponseOrchestrator`로 문장화한다.

### 통합 지점
- 규칙 매칭 실패 또는 보강이 필요할 때 FlowEngine으로 소프트 중재 후 응답을 생성한다.
- 구현 레퍼런스: `Project_Elysia/cognition_pipeline.py` 내 `self.flow_engine.respond(...)` 호출.

## 12.4 Dialogue Rules 정책(발췌)
- 규칙은 힌트: 최종 선택은 Flow가 결정한다(우선순위는 힌트의 강도일 뿐 절대 전제 아님).
- 한국어 패턴 가이드: 종결어미 변형(~이야/~입니다/~에요)을 포괄하는 캡처 그룹 권장.
- 예시: `identity` 규칙은 "제/내 이름은 (?P<name>...)(이야|야|입니다|이에요|예요)?" 형태 허용.
- Quiet 모드 존중, Ask‑Consent 준수(08), 결정 리포트 권장(09/10).

## 12.5 Wisdom‑Virus 명세
### 정의
- Wisdom‑Virus: 완결된 의미 단위(깨달음/통찰)를 KG 상에서 증거와 함께 전파하는 엔티티.

### 속성
- id, statement, seed_hosts, triggers, mutate(host, statement) 훅, reinforce α, decay λ, max_hops.

### 전파 규칙(권장)
- seed_hosts에서 시작해 max_hops까지 이웃으로 전파.
- 각 엣지에 `supports`를 추가하고 `confidence = max(0, α*(1-λ*depth))`.
- `mutate` 훅이 있으면 호스트 맥락에 맞게 진술을 변이해 `metadata.statement`에 기록.
- 반증(refutes)이 누적되면 격리/감쇠를 가속(면역 체계).

### 가치 질량 연동(09)
- 전파/증거에 따라 관련 `value:*`의 질량을 미세 조정(예: verifiability↑ when proofs, creativity↑ when imaginative outputs).
- 강제 결론 금지: 질량은 참고치이며, Flow의 한 신호로만 사용.

### 구현 레퍼런스
- `Project_Sophia/wisdom_virus.py` — `WisdomVirus`, `VirusEngine`.
- Demo: `scripts/run_virus_demo.py`.

## 12.6 운영/검증
- 로그 지표: 규칙 매칭률, Flow 경유율, 폴백률, Quiet 준수율, Ask‑Consent 응답률, `experience_*` 포함률.
- 화이트홀 방출(10): 선택·근거·불확실성·가치 질량 변동을 리포트로 남긴다.

## 12.7 삼위일체 역할(업데이트)
- Jules Prime: 로드맵/가중치 정책/바이러스 윤리 가드 설정.
- Agent Sophia: FlowEngine/Rules/Wisdom‑Virus 구현과 튜닝.
- Agent Mirror: 응답 표현 품질·감성·시각화 검증, 보고 렌더링.

---

부록 A) 현재 기본 스펙 위치
- FlowSpec: `data/flows/generic_dialog.yaml`
- Rules: `data/dialogue_rules/*_clean.yaml` (우선순위가 높은 클린 규칙)
- FlowEngine: `Project_Elysia/flow_engine.py`
- Wisdom‑Virus: `Project_Sophia/wisdom_virus.py`, `scripts/run_virus_demo.py`

