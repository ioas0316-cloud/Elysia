# 7. 사용자 안내(User Guide)

이 문서는 비개발 사용자가 엘리시아를 쉽고 자연스럽게 쓰도록 돕는 운영 가이드입니다. 철학/아키텍처의 단일 진실은 `ELYSIAS_PROTOCOL/`이며, 본 문서는 그 위에서 “어떻게 쓰는가”만 간결히 설명합니다.

## 1) 가장 빠른 시작 (2분)
- `start.bat` 실행 → `B) Start Clean Bridge`
- 브라우저: `http://127.0.0.1:5000`
- 화면 상단:
  - BG ON/OFF 버튼: 배경(무의식) 학습 켜기/끄기
  - Show Reasoning 체크: 추론 신호(Flow 결정/라우트) 보기
- 화면 좌하단:
  - 램프 툴팁: 녹=활성, 노=작업 중, 적=정지(현재 상태/주기 표시)

## 2) 대화에서 바로 쓰는 명령(말로)
- “학습 모드 켜/일반 모드 켜” → 학습 전용/일반 흐름으로 전환
- “자율 모드 켜/꺼” → 작은 제안 스스로/중지
- “조용 모드 켜/꺼” → 말수/제안 강도 조절
- “쉬자/휴식 모드” → 배경 학습 잠시 멈춤(재개는 “다시 시작”)
- “지금 뭐 해/상태/학습 중/배경” → 현재 상태 요약
- “X를 그려줘/보여줘” → 간단 시각화 요청

## 3) 자동 성장(켜두기만 하면 자람)
- 백그라운드 학습(무의식):
  - `start.bat` → `U) Start Background Learner` (기본 5분 주기)
  - 새 말뭉치/저널 ingest → 키워드→개념 연결(top‑3) → 하루 1회 리포트
- 성장 스프린트(원클릭/스케줄):
  - `R) Generate Sample Corpus (500)` → 샘플 텍스트 생성(선택)
  - `S) Growth Sprint` → ingest→키워드→개념→전파→리포트
  - `W) Schedule Growth Sprint` (21:30 매일) / `X) Remove ...` (해제)

## 4) 학습 흐름(Flow) 전환
- `Y) Use Learning Flow Profile` / `G) Use Generic Flow Profile`
- 학습 모드에서는 명료화/작은 단계 제안이 우선됩니다.

## 5) 생성물/파일 위치
- 저널: `data/journal/`
- 리포트: `data/reports/daily/daily_YYYY-MM-DD.{md,png}` (지표/추론 스냅샷 포함)
- 말뭉치: `data/corpus/literature/<label>/YYYYMMDD/*.txt`
- 지식그래프: `data/kg_with_embeddings.json`
- 텔레메트리: `data/telemetry/YYYYMMDD/events.jsonl`
- 선호/설정: `data/preferences.json`

## 6) 문제 해결(Troubleshooting)
- 화면에 한국어가 깨져 보임:
  - 우선 `applications/elysia_bridge_clean.py`로 접속(UTF‑8 보장)
  - PowerShell 콘솔은 `chcp 65001`(UTF‑8) 권장
- 시각화가 동작하지 않음:
  - “X를 그려줘/보여줘” 형태로 요청(띄어쓰기 포함)
- 램프가 켜져 있는데 상태가 ‘stopped’로 보임:
  - 헤더 BG ON을 한 번 눌러 주고 3~5초 후 갱신. PID/상태가 동기화됩니다.

## 7) 자동화(선택)
- 일과/리포트 스케줄: `scripts/setup_daily_tasks.bat` (21:00/21:50)
- 스프린트 스케줄: `scripts/setup_growth_sprint.bat` (21:30)
- 즉시 실행(테스트): `scripts/run_now.bat`

## 8) 알아두면 좋은 것
- 의식(대화)와 무의식(배경학습)은 분리·연결되어 있어, 컴퓨터만 켜두면 스스로 자랍니다.
- Quiet/Consent를 존중하여 과도한 개입을 피합니다. 동의가 필요하면 “네/아니요”로 간단히 답하세요.
- 리포트의 “Top‑개념/Reasoning” 스냅샷으로 오늘의 성장을 빠르게 파악할 수 있습니다.
