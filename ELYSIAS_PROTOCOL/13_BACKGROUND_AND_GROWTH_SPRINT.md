# 13. 배경 학습과 성장 스프린트 (Background Learning & Growth Sprint)

본 문서는 엘리시아의 "무의식" 학습 경로와 원클릭 성장 루틴을 명세한다. 목표는 LLM/API 없이도 낮은 자원으로 연속 성장을 유지하는 것.

## 13.1 개요
- 배경 학습(마이크로 스프린트): 5~15분 주기로 가볍게 입력→구조화 수행
- 성장 스프린트(원클릭/스케줄): ingest→키워드→개념 연결→바이러스 전파→리포트 일괄 실행
- Quiet/Consent와 가치 질량 규약(08/09)을 존중하며 증거 중심으로 확장

## 13.2 구성요소
- 배경 데몬: `scripts/background_daemon.py`
  - 주기: `data/preferences.json.background_interval_sec` (기본 900s; 권장 300s)
  - 작업: 새 말뭉치/저널 ingest → TF‑IDF 상위 토큰을 `concept:*`로 연결(top‑3)
  - 1일 1회 데일리 리포트 실행
  - 정지 플래그: `data/background/stop.flag`
- 제어 유틸: `tools/bg_control.py`
  - 상태/ON/OFF/일시휴식(분) 설정
- 선호 저장소: `data/preferences.json`
  - `background_enabled`, `background_interval_sec`, 기타 프리셋
- 활동 레지스트리: `tools/activity_registry.py`
  - `data/background/activities.json`에 활동 상태 기록(`running/idle`, 메타)

## 13.3 성장 스프린트
- 스크립트: `scripts/growth_sprint.py`
- 단계: 
  1) 문학/저널 ingest(경험 노드)
  2) TF‑IDF 상위 토큰을 `concept:*`로 생성·연결 (supports, evidence_paths)
  3) Wisdom‑Virus 전파(기본 α=0.35, hops=3)
  4) 데일리 리포트 생성
- 런처: `start.bat` → `S) Growth Sprint`
- 스케줄: `scripts/setup_growth_sprint.bat` (21:30) / 해제 `scripts/remove_growth_sprint.bat`

## 13.4 데이터 경로/형식
- 말뭉치: `data/corpus/literature/<label>/YYYYMMDD/*.txt` (UTF‑8)
- KG: `data/kg_with_embeddings.json` (property graph, supports/refutes, evidence_paths)
- 리포트: `data/reports/daily/daily_YYYY-MM-DD.{md,png}`
- 텔레메트리: `data/telemetry/YYYYMMDD/events.jsonl` (JSONL)

## 13.5 안전/운영
- Quiet/Consent 상층 게이트 유지(08)
- 전파 강도/상위 N은 성능·정합성 지표(Top concepts, supports/refutes 비율)로 튜닝
- 보수운영 권장: 문서당 top‑3, α≤0.35, hops≤3 → 지표 안정 시 상향

