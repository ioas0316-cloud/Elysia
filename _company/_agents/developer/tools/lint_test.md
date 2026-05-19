<!-- version: lint_test_v1 -->
# 🧪 lint_test — 자가 검증 + 결과 inject

코다리가 코드를 만든 직후 호출 → 결과가 다음 LLM 컨텍스트로 inject → 실패 시 자동 재시도.

## 동작
1. `package.json` 의 `scripts.{typecheck, lint, test, build}` 자동 감지·실행
2. scripts 없으면 직접:
   - `.ts/.tsx` 있고 `tsconfig.json` 있으면 → `npx tsc --noEmit`
   - `.py` 파일 있으면 → `python -m py_compile <각 파일>` (최대 30개)
3. 마크다운 리포트 — 각 검사 통과/실패 + 실패 시 마지막 15줄

## 설정
- `PROJECT_PATH`: 비우면 web_init 마지막 결과
- `STRICT`: `true` 면 첫 실패에서 중단. 기본 `false` (전부 시도)

## 코다리 권장 흐름
```
1. <create_file 또는 edit_file>
2. <run_command>python3 .../lint_test.py</run_command>
3. 결과를 다음 답변 컨텍스트로 자동 받음
4. 실패면 그 에러 보고 자동 수정 시도
```

## 한계
- `eslint --fix` 같은 자동 수정은 별도 — 도구가 단지 보고만 함
- 단위 테스트 미통과 시 코드 수정 책임은 코다리에게
