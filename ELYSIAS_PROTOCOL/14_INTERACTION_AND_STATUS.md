# 14. 상호작용 UX: 상태·램프·추론(Reasoning)

엘리시아의 “의식/무의식” 상태를 시각·대화로 직관적으로 노출하여 비개발자 사용성을 보장한다.

## 14.1 웹 UI 요소
- 램프(좌하단):
  - 녹색=배경 활성, 노랑=작업 중, 빨강=정지
  - 툴팁: 색상 의미 + 현재 상태/interval
- 헤더 컨트롤: BG ON/OFF, 상태 텍스트(ENABLED/RUNNING)
- Reasoning 패널(우하단 토글):
  - 최근 `flow.decision`/`route.arc` 요약(가중치·신호·Top 선택·echo 초점)

## 14.2 API
- 배경 제어/상태:
  - `GET /bg/status`, `POST /bg/on`, `POST /bg/off`
- 자기 상태 집계:
  - `GET /self/status` → flow 프로필, quiet/auto, background, activities, busy 플래그
- 추론 로그:
  - `GET /trace/recent` → 텔레메트리 tail(결정/라우트 이벤트)

## 14.3 자연어 제어(룰)
- 모드 전환: `cmd_learning_mode.yaml`, `cmd_generic_mode.yaml`
- 자율/조용: `cmd_autonomy_on/off.yaml`, `cmd_quiet_on/off.yaml`
- 배경 휴식/재개: `cmd_bg_rest.yaml`, `cmd_bg_resume.yaml`
- 상태 질의: “지금 뭐 해/상태/학습 중/배경” → 상태 요약 응답

## 14.4 텔레메트리 스키마(발췌)
- `flow.decision`:
  - `weights{clarify,reflect,suggest}`, `signals{…}`, `top_choice`, `evidence.echo_top[]`
- `route.arc`:
  - `from_mod`, `to_mod`, `latency_ms`, `outcome`

