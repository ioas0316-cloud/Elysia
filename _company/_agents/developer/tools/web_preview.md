<!-- version: web_preview_v1 -->
# 💻 웹 dev server 백그라운드 실행 + URL 안내

`npm run dev` 같은 dev server를 백그라운드로 띄우고 미리보기 URL을 자동 감지·반환.

## 동작
1. PROJECT_PATH의 package.json scripts.dev 자동 감지
2. 백그라운드 실행 (nohup·detached) + PID 파일 저장
3. 첫 8초 동안 로그에서 `localhost:포트` URL 파싱
4. AUTO_OPEN=true 면 브라우저 자동 열기

## 설정
- `PROJECT_PATH`: 비우면 web_init이 마지막에 만든 프로젝트 자동 사용
- `DEV_CMD`: 비우면 자동 감지 (`npm run dev` / `npm start`)
- `AUTO_OPEN`: `true`면 미리보기 URL을 브라우저로 열기

## 종료
- 같은 도구 재실행 → 이전 PID 자동 kill 후 새로 시작
- 수동 종료: `kill <PID>` (PID는 출력에 표시)
- macOS/Linux: `pkill -f "npm run dev"`

## 사용 예시
```
1. web_init으로 프로젝트 셋업 (예: nextjs, my-blog)
2. web_preview 실행 → http://localhost:3000 자동 표시
3. 코드 변경 → HMR로 즉시 반영 (브라우저)
4. 작업 끝나면 kill 또는 도구 재실행
```

## 한계
- 진짜 라이브 미리보기 칩 (사이드바 안의 상태 인디케이터)은 별도 UI 작업 필요. 현재는 출력에 URL만 반환.
