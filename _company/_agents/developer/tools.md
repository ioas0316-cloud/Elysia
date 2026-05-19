# 💻 코다리 — 도구 매니페스트

_코다리 에이전트가 어떤 도구를 어디까지 자율적으로 쓸 수 있는지 정의합니다._
_매번 시스템 프롬프트로 주입되며, 텔레그램에서 `/tools`로 현재 상태 확인 가능._

---

## 자율도 레벨

AUTONOMY_LEVEL: 2

| 값 | 의미 |
|---|---|
| 0 | Off — 도구 전체 비활성 (이 에이전트는 채팅만) |
| 1 | Read-only — 읽기·분석·보고만, 외부에 쓰기 X |
| 2 | Draft — 초안 작성 후 사용자 승인 게이트 통과해야 실행 ⭐ 권장 기본값 |
| 3 | Auto — 화이트리스트 안에서 사용자 승인 없이 실행 |

> 위 `AUTONOMY_LEVEL` 줄의 숫자(0~3)를 직접 바꾸면 다음 호출부터 적용됩니다.

---

## 사용 가능한 도구

### `web_init`
5개 템플릿 자동 시작 — vite·next·astro·expo·vanilla

- `enabled`: true
- `requires_credentials`: `config.md` 참조

### `pack_apply`
두뇌의 키트 (landing·portfolio·dashboard·mobile)를 프로젝트에 자동 적용 + npm install + App.tsx 업데이트

- `enabled`: true
- `requires_credentials`: `config.md` 참조

### `web_preview`
dev server 백그라운드 실행 + URL 자동 추출

- `enabled`: true
- `requires_credentials`: `config.md` 참조

### `pwa_setup`
웹사이트 → PWA 변환 (manifest·sw·아이콘 자동 생성)

- `enabled`: true
- `requires_credentials`: `config.md` 참조

### `lint_test`
코드 수정 후 자가 검증 — tsc·py_compile·npm scripts 자동 실행 + 결과 리포트

- `enabled`: true
- `requires_credentials`: `config.md` 참조


---

## 로드맵 (예정)

_아래 도구들은 향후 버전에서 추가 예정. 지금은 카탈로그에만 있음._

### `git_committer` _(예정)_
작업 단위 자동 커밋 (의미 단위 + git add -A 금지)

- 아직 구현되지 않은 도구입니다. 로드맵에 있으며 향후 버전에서 추가 예정.

### `deploy_cli` _(예정)_
Vercel/Netlify/Cloudflare 배포 (deploy --prod는 항상 승인)

- 아직 구현되지 않은 도구입니다. 로드맵에 있으며 향후 버전에서 추가 예정.


---

## 안전 규칙 (모든 레벨 공통, 절대 우회 X)

- **삭제·배포·발송**(rm, deploy --prod, send, publish) 류는 자율도와 무관하게 **항상 승인 게이트**.
- 외부 API 호출 전 `config.md`의 토큰 존재 여부 확인.
- 모든 외부 행동은 `_agents/developer/activity.log`에 한 줄 기록 (감사용).
- 승인 대기 액션은 `approvals/pending/` 에 저장 → 텔레그램 `/approvals` 로 조회.

---

_레벨을 어떻게 골라야 할지 모르겠다면 `2 (Draft)`가 안전한 시작점입니다._
