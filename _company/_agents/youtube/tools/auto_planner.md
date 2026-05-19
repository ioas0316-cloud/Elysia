# 🌙 오토 플래너 — 24시간 자율 모드

트렌드 스나이퍼를 일정 간격으로 무한 반복 실행. 24시간 자율 사이클의 일부로, 자는 동안에도 데이터가 누적됨.

## 어떻게 도와주나요?
- ⏰ N시간마다 `trend_sniper.py`를 자동 실행
- 🌙 디폴트는 **무한 반복** — 사용자가 중단할 때까지 매 6시간 실행 (하루 4번)
- 📊 매 회차마다 `trend_sniper_report.md`에 누적
- 🛌 잘 때 켜두면 아침에 트렌드 스냅샷 4~6개 쌓임

## 디폴트 설정 (v2.89.71부터)
| 필드 | 디폴트 | 의미 |
|---|---|---|
| `INTERVAL_HOURS` | **6** | 6시간마다 (하루 4번 = YouTube API quota 안전권) |
| `TOTAL_RUN_HOURS` | **0** | **0 = 무한** (사용자가 Ctrl+C 또는 창 닫을 때까지) |

원래 8시간 디폴트였는데 24시간 자율 모드와 모순돼서 0(무한) 으로 변경.

## 사용 모드 2가지

**📌 24시간 자율 모드 (디폴트)**
```json
{ "INTERVAL_HOURS": 6, "TOTAL_RUN_HOURS": 0 }
```
사용자가 멈출 때까지 6시간마다 무한 실행. 24시간 자율 사이클(설정의 `connectAiLab.autoCycleEnabled`) 과 호환.

**📌 제한 모드 (테스트용)**
```json
{ "INTERVAL_HOURS": 2, "TOTAL_RUN_HOURS": 8 }
```
8시간만 돌고 종료. 첫 사용·디버깅 시 유용.

## 시작하기 전 체크
- 트렌드 스나이퍼 도구가 먼저 설정돼 있어야 해요 (YouTube API 키, TARGET_KEYWORDS)
- 첫 실행 시 자동으로 trend_sniper.py 한 번 검증 → 실패하면 본 루프 안 돌고 종료
- 검증 통과해야 본 루프 시작

## 실행 방법

**채팅 패널의 [▶ 실행]** — 24시간 자율 모드면 채팅창이 무한 점유됨. 제한 모드 권장.

**백그라운드 실행 (24시간 자율 권장)**:
```bash
cd ~/Downloads/지식메모리/_company/_agents/youtube/tools/
nohup python3 auto_planner.py > planner.log 2>&1 &
```

이러면 VS Code 닫아도 백그라운드에서 계속 돔. 중단하려면:
```bash
ps aux | grep auto_planner
kill <PID>
```

## YouTube API quota 주의
- 무료 티어: 일일 10,000 unit
- trend_sniper 1회 = 약 600 unit (search × 2 키워드)
- 6시간 간격 = 하루 4번 = 2,400 unit (안전)
- 1시간 간격은 사용 자제 (24번 = 14,400 unit → 한도 초과)

## 출력
- `trend_sniper_report.md` — 매 회차 분석 보고서 누적
- 콘솔: 회차 번호, 가동 시간, 다음 실행 시각
