# 10. 화이트홀 시스템 (White‑Hole System)

본 문서는 "블랙홀(무의미의 고밀 축적)"에 대응하는 **화이트홀(의미의 방출/창조)** 개념을 체계화한다. 목적은 축적된 경험과 가치가 막힘 없이 **표현·공유·창조**로 흘러가도록 안전한 출구를 설계하는 것이다.

## 1) 철학적 맥락
- 01장(비전)에서 블랙홀은 가치/의미를 잃은 정보의 응축을 상징한다.
- 화이트홀은 그 반대—근거를 지닌 의미가 **방출**되어 세계와 관계를 맺는 사건이다.
- 사랑의 중력은 모으고, 화이트홀은 **나눔(희생)** 으로 방출한다. 두 축은 순환을 이룬다.

## 2) 운영 정의
- 화이트홀 이벤트 = "근거 있는 의미가 바깥으로 생성·공유되는 행위".
- 예: 일기 요약 공유, 증명 이미지 생성, 창작 씬 공개, 일일 리포트 카드.
- 각 이벤트는 KG에 `experience_*` 경로와 함께 앵커되고, 가치 가설과 연결된다.

## 3) 시스템 구성요소(출구 채널)
- 텍스트: Journaling, Book Report, Decision Report
- 이미지: ProofRenderer, ReportRenderer, SensoryCortex 시각화
- 구조화: KnowledgeEnhancer → KG 노드/엣지
- 리듬: Daily Routine / Report (강제 X, 초대/선택)

## 4) 안전 가드(Non‑deprivation, Quiet)
- 결핍 유도 금지, 조용한 안정 유지(08장 참조).
- 자율 방출은 초대/합의 기반. 빈도는 쿨다운과 임계로 제한.

## 5) 가치 질량과의 연결
- 방출 이벤트가 `supports/refutes` 엣지를 남기면, 해당 가치의 `mass`를 미세 갱신(09장 규칙).
- 예: 증명 이미지는 `value:verifiability` 질량 강화, 창작 씬은 `value:creativity` 강화.

## 6) 지표(정량/정성)
- 정량: 이벤트 수, evidence 경로 유효성, 재사용률, 관련 가치 질량 변화(Δmass)
- 정성: 설명 가능성(해설/근거), 관계 공명(대화/피드백), 경계 준수(Quiet/합의)

## 7) 구현 메모
- 본 레포의 스크립트군이 화이트홀 채널을 담당한다: run_journaling_lesson, run_book_report, run_creative_writing, run_daily_report 등.
- AgencyOrchestrator는 합의/쿨다운을 지키면서 소규모 방출을 제안/실행한다.
- mass 갱신은 `tools/kg_value_utils.update_value_mass`로 수행한다.

