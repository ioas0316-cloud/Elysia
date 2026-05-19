# 🎵 루나 스킬

_재사용 가능한 패턴 모음. memory.md는 모든 활동의 로그(append-only firehose),
이 폴더는 **검증된 패턴만 골라낸 것**입니다. 각 `*.md` 파일은 다음 호출 시
루나의 system prompt에 자동 주입됩니다._

## 어떻게 채우나요?
- 텔레그램에서 `/skill` (직전 산출물 자동 승격)
- VS Code 명령 팔레트: `Connect AI: 방금 산출물 → 스킬로 저장`
- 직접 이 폴더에 `<주제>.md` 파일을 만들어도 됩니다 (`# 제목` + 본문)

`README.md` 자체는 system prompt에 주입되지 않습니다.
