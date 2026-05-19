<!-- version: web_init_v1 -->
# 💻 웹·모바일 프로젝트 자동 시작

5개 템플릿 중 골라서 한 번에 프로젝트 폴더 + 의존성 설치 + 첫 실행 가능한 상태로.

## 템플릿

| 템플릿 | 용도 | 의존성 | 첫 실행 |
|---|---|---|---|
| **vite-react** ⭐ 추천 | SPA·대시보드·SaaS UI | Node·npm | `npm run dev` → :5173 |
| **nextjs** | full-stack·SEO·서버 컴포넌트 | Node·npm | `npm run dev` → :3000 |
| **astro** | 블로그·콘텐츠·랜딩 | Node·npm | `npm run dev` → :4321 |
| **expo** | 진짜 모바일 앱 (iOS/Android) | Node·npm·Expo Go | `npm start` → QR |
| **vanilla** | 단순 HTML/CSS/JS | 없음 | `python3 -m http.server` |

## 사용법

설정 (web_init.json):
- `TEMPLATE`: 위 5개 중 하나
- `PROJECT_NAME`: 영문·숫자·하이픈 (예: `my-blog`)
- `OUTPUT_DIR`: 비우면 `~/connect-ai-projects/`

실행:
```
python3 web_init.py
```

## 어떤 걸 골라야 하나

- **이걸로 시작:** vite-react (SPA·대시보드·내부 도구)
- **블로그·기업 사이트:** astro
- **풀스택 (DB·API):** nextjs
- **모바일 앱:** expo (PWA로 충분하면 vite-react)
- **HTML 한 페이지:** vanilla

## 다음 단계

셋업 후 코다리가:
1. `web_preview` 도구로 dev server 실행
2. 사용자 요구사항대로 컴포넌트 추가
3. `pwa_setup` 으로 PWA 만들기 (모바일 앱처럼)
4. Vercel/Netlify에 배포
