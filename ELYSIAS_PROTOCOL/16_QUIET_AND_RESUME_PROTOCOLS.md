# 16. 전역 정지/재개 프로토콜 (Quiet‑All / Resume‑All)

큰 작업 전 안정화를 위해 전체 동작을 한 번에 정지/재개하는 절차를 정의합니다.

## 16.1 Quiet‑All (전역 정지)
- 목적: BG OFF, 조용 모드(quiet), 자율 행동 OFF, 예약작업 제거
- 스크립트/메뉴:
  - `start.bat` → `F) Quiet‑All`
  - 내부: `scripts/quiet_all.py`
- 수행 내용:
  - `tools.bg_control.stop_daemon()` 호출 + `data/background/stop.flag` 생성
  - `data/preferences.json` 갱신: `background_enabled=false`, `quiet_mode=true`, `auto_act=false`
  - Windows 예약작업 제거: `schtasks /Delete /TN "ElysiaGrowthSprint" /F` (해당 시)
- 검증: `/self/status`에서 `enabled:false`, `running:false`, `quiet_mode:true`

## 16.2 Resume‑All (전역 재개)
- 목적: BG ON, 조용 모드 OFF, 자율 행동 ON, 데몬 시작
- 스크립트/메뉴:
  - `start.bat` → `E) Resume‑All`
  - 내부: `scripts/resume_all.py`
- 수행 내용:
  - `data/preferences.json` 갱신: `background_enabled=true`, `quiet_mode=false`, `auto_act=true`
  - `tools.bg_control.start_daemon(interval)`로 데몬 기동(기본 900초)
- 검증: `/self/status`에서 `enabled:true`, `running:true`(약간의 지연 가능)

## 16.3 참고
- 램프: 녹색=활성, 노랑=작업 중, 빨강=정지
- 텔레메트리/리포트:
  - `data/telemetry/YYYYMMDD/events.jsonl`
  - `data/reports/daily/daily_YYYY-MM-DD.{md,png}`

