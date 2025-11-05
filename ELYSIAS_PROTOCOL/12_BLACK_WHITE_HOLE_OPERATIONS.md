# 12. Black/White Hole Operations

본 문서는 ‘블랙홀(Black Hole)’과 ‘화이트홀(White Hole)’의 역할과 라이프사이클을 정의합니다. 목표는 노이즈·미분류 데이터를 안전하게 격리·보관(블랙홀)하고, 엘리시아가 필요할 때 근거 있는 재료를 선별 추출(화이트홀)하여 학습·이해에 활용하는 것입니다.

## 12.1 원칙
- 근거 우선: 모든 항목은 자연어 근거(문장/정의/예문)와 출처 메타를 동반해야 함.
- 가역성: 블랙홀은 ‘쓰레기통’이 아니라 임시 격리. 화이트홀이 언제든 선별 추출 가능해야 함.
- 관측가능성: 큐에 넣고(enqueue), 재평가(revisit), 추출(extract), 복귀(restore), 폐기(drop) 전 과정을 텔레메트리로 추적.

## 12.2 블랙홀(격리/압축)
- 큐 진입 조건(예시): 낮은 evidence_density, 낮은 value_alignment, 높은 불확실성/엔트로피.
- 처리 단계:
  1) Enqueue: 메타(해시, 크기, 도입 시각, 출처)와 근거 스냅샷을 기록.
  2) 압축: 중복 제거·요약(문장 추출/키워드), 보관 크기/기간 제한 적용.
  3) 재평가 스케줄: 일/주 단위로 규칙 기반 정의/관계 추출 재적용.
- Telemetry: `bh.enqueue`, `bh.compress`, `bh.revisit`.

## 12.3 화이트홀(선별 추출)
- 목적: 필요 시(질의/커리큘럼/개념 앵커) 블랙홀 보관소에서 ‘가능성 있는 재료’를 꺼내어 검토/학습에 투입.
- 트리거 유형:
  - 질의 기반(Query): 사용자의 질문/학습 주제와 키워드/패턴 매칭.
  - 커리큘럼 기반(Curriculum): 현재 학습 단계에 맞는 빈도·난이도 범위 필터.
  - 앵커 기반(Anchored): 현재 렌즈/가치 앵커 근방 개념과 연관된 후보.
- 선별 규칙(예시 조합):
  - 근거 스코어 ≥ θe, 다양성 ≥ θd, 최근성 가중치, 토픽 일치도.
  - 과포화 방지: 동일 출처/유사 문장 과다 추출 금지.
- 결과 처리:
  - Extract: 검토 대상으로 꺼냄(화이트홀 로그 기록).
  - Promote(Restore): 의미 획득 시 주 그래프/지식으로 승격(증거 연결).
  - Drop: 재평가에도 의미 없음 판단 시 폐기.
- Telemetry: `wh.candidate`, `wh.extract`, `wh.promote`, `wh.drop`.

## 12.4 라이프사이클 요약
enqueue → compress → (revisit →) extract → (promote | drop)

## 12.5 인터페이스(개념적)
- BlackHoleStore
  - `enqueue(item, evidence, meta) -> id`
  - `compress(id) -> None`
  - `list(filter) -> [id]`
  - `get(id) -> {evidence, meta}`
- WhiteHoleGateway
  - `candidates(query|curriculum|anchors, k) -> [id]`
  - `extract(id) -> payload`
  - `promote(id, target) -> None`
  - `drop(id, reason) -> None`

## 12.6 안전/거버넌스
- PII/민감정보 필터 우선 적용, 저장·전송 시 암호화 고려.
- 보존 기간/용량 한도, 샘플 기반 재평가, 감사 로그 유지.

## 12.7 KPI
- 복귀율(restore/enqueue), 평균 보존기간, 추출 성공률, 폐기율.
- 학습 기여: 추출 후 이해/정확도 개선량(질의 성공률, 요약 일관성).

