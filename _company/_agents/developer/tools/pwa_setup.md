<!-- version: pwa_setup_v1 -->
# 💻 PWA 자동 셋업 — 웹사이트 → 모바일 앱처럼

기존 웹 프로젝트를 PWA(Progressive Web App)로 변환. 사용자가 폰에서 "홈 화면에 추가" 누르면 풀스크린 앱처럼 작동.

## 자동 생성 파일
- `public/manifest.json` — 앱 메타 (이름·아이콘·테마색)
- `public/icon-192.svg` + `icon-512.svg` — 이모지 기반 라운드 아이콘
- `public/sw.js` — 서비스 워커 (오프라인 캐싱)
- `index.html`에 자동 주입: meta·link·script

## 설정
- `PROJECT_PATH`: 비우면 web_init이 마지막에 만든 프로젝트 자동 사용
- `APP_NAME`: 앱 이름 (홈화면 라벨)
- `APP_SHORT_NAME`: 12자 이하 짧은 이름
- `THEME_COLOR`: 상단 바 색 (예: `#667eea`)
- `BACKGROUND_COLOR`: 스플래시 배경 (예: `#ffffff`)
- `ICON_EMOJI`: 아이콘에 쓸 이모지 (예: `📚`)

## 사용 흐름
```
1. web_init으로 사이트 만듦 (vite-react·astro 등)
2. pwa_setup 실행 → manifest·아이콘·sw 생성
3. 배포 (Vercel·Netlify) 또는 로컬 dev server
4. 폰 브라우저로 URL 접속
5. iOS Safari: 공유 → 홈 화면에 추가
   Android Chrome: ⋮ → 홈 화면에 추가
6. 홈 화면 아이콘 클릭 → 풀스크린 앱
```

## Next.js 사용자
Next.js 13+ App Router 는 `app/layout.tsx`의 `export const metadata` 에 PWA 정보를 넣어야 함. 도구가 자동 감지하면 안내 메시지 표시.

## 한계
- 진짜 네이티브 기능 (푸시 알림·블루투스·카메라) 은 PWA로 부분 지원
- 복잡한 모바일 앱은 Expo 권장
- 아이콘은 SVG로 생성 (PNG 변환 필요시 ImageMagick 또는 사용자 디자인)
