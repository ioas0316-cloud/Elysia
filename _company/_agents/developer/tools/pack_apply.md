<!-- version: pack_apply_v1 -->
# 📋 pack_apply — 키트 한 명령으로 적용

두뇌에 주입된 템플릿 팩을 사용자 프로젝트에 자동 적용. 파일 복사 + 의존성 설치 + App.tsx 자동 업데이트.

## 사용
설정 (pack_apply.json):
- `KIT_NAME`: 'landing-kit' / 'portfolio-kit' / 'dashboard-kit' / 'mobile-kit'
- `PROJECT_PATH`: 적용할 사용자 프로젝트 (비우면 web_init 결과 자동)

실행:
```
python3 pack_apply.py
```

## 동작 (3단계)

1. **파일 복사**: 키트의 `files/` 폴더를 manifest의 `apply.copy_to` 경로로 (예: `src/components/`)
2. **의존성 자동 설치**: manifest의 `apply.post_install` 명령 순차 실행
   - 예: `npm install lucide-react`
   - Expo: `npx expo install @react-navigation/native ...`
3. **App.tsx 자동 업데이트**: manifest의 `apply.app_imports` + `app_body` 로 import + JSX 본문 추가

## 키트별 동작

### landing-kit (vite-react)
- 복사: 6개 컴포넌트 → src/components/
- 설치: lucide-react
- App.tsx: Hero·Features·Pricing·FAQ·CTA·Footer 자동 배치

### portfolio-kit (vite-react)
- 복사: 5개 컴포넌트
- 설치: lucide-react
- App.tsx: Nav·About·Work·Skills·Contact 자동 배치

### dashboard-kit (vite-react)
- 복사: 5개 컴포넌트
- 설치: lucide-react
- App.tsx: `<DashboardLayout />` 한 줄로 풀스크린 대시보드

### mobile-kit (Expo)
- 복사: App.tsx + screens/ 3개
- 설치: @react-navigation/native + bottom-tabs + screens + safe-area-context
- App.tsx: 기존 덮어쓰기 (Bottom Tab Navigator)

## 코다리 사용 예시

```
사용자: "다이어트 SaaS 랜딩 만들어줘"

코다리:
1. web_init (TEMPLATE=vite-react, PROJECT_NAME=diet-saas)
2. pack_apply (KIT_NAME=landing-kit) ← 새 도구
3. edit_file 로 텍스트만 다이어트 SaaS 카피로 교체
4. web_preview (자동 dev server + 브라우저)

→ 5분 안에 완성 + 모바일·데스크탑 반응형 + Tailwind 컨벤션 일관
```

## 안전장치
- `KIT_NAME` 없거나 잘못되면 종료 + 사용 가능 키트 안내
- 두뇌 폴더 자동 탐색 (~/.connect-ai-brain 또는 fallback 경로)
- 파일 복사는 덮어쓰기 (사용자가 수정한 거 있으면 백업 권장)
- 의존성 설치 실패해도 계속 진행 (warn만, 사용자 수동 가능)
- App.tsx 패턴 매칭 실패 시 수동 안내

## 한계
- App.tsx 자동 업데이트는 best-effort (단순 패턴 매칭). 복잡한 기존 App.tsx 는 수동 권장.
- 키트가 React 외 (Vue·Svelte 등)에 적용되면 App.tsx 패턴 안 맞음 — 키트가 진짜 React 인지 manifest.base 로 검증 필요.
