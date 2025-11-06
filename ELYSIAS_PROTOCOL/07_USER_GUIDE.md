# 7. 사용자 안내서 (User Guide)

이 문서는 비개발 사용자가 Elysia를 쉽게 사용할 수 있도록 만든 "운영 가이드"입니다. 철학과 아키텍처의 단일 진실원은 계속 `ELYSIAS_PROTOCOL/`이며, 본 문서는 그 비전에 맞춘 사용 방법을 설명합니다.

## 빠른 시작

- `start.bat` 실행 → 숫자/문자 한 키로 원하는 작업을 선택합니다.
- 대부분의 산출물은 `data/` 폴더 아래에 저장됩니다.
- 첫 실행 시 의존성 설치가 필요할 수 있습니다(메뉴 1에서 Y 선택).

## 메뉴 설명 (start.bat)

- 1) Start Web Server
  - Flask 대시보드를 실행하고 브라우저를 엽니다.
  - 모니터: `http://127.0.0.1:5000/monitor`
- 2) Run Daily Routine
  - 오늘의 일기와 창작소설을 생성합니다.
  - 산출물: `data/journal/`, `data/writings/`
  - KG(지식그래프): `data/kg_with_embeddings.json`에 경험 연결
- 3) Generate Daily Report
  - 오늘 산출물을 모아 Markdown/PNG 카드를 생성합니다.
  - 경로: `data/reports/daily/daily_YYYY-MM-DD.{md,png}`
- 4) Run Textbook Demo
  - 샘플 교재(JSON)로 학습 프레임을 실행합니다.
  - 시각 산출물과 개념/관계가 KG에 기록됩니다.
- 5) Journaling
  - 오늘 프롬프트로 일기와 요약을 만듭니다.
  - 경로: `data/journal/YYYY-MM-DD*.txt`
- 6) Book Report
  - 로컬 `.txt` 파일로 요약/인물/주제를 추출해 리포트를 작성합니다.
  - 경로: `data/reports/*_report.md`
- 7) Creative Writing
  - 장르/테마로 아웃라인과 씬을 생성합니다.
  - 경로: `data/writings/TIMESTAMP_genre_theme.md`
- 8) Trinity Mission Demo
  - 파일→증명→이미지→KG 귀속까지 한 번에 데모합니다.
- 9) Math Verification Demo
  - 등식 한 줄을 증명하고 이미지로 저장합니다(`data/proofs/`).

## 산출물 한눈에 보기

- 일기: `data/journal/` (원문과 요약)
- 독후감: `data/reports/` (`*_report.md`)
- 창작소설: `data/writings/` (마크다운)
- 증명 이미지: `data/proofs/`
- 일일 리포트: `data/reports/daily/` (MD/PNG 카드)
- 지식그래프: `data/kg_with_embeddings.json`

## 자주 묻는 질문 (FAQ)

- Q: 실행했는데 아무 것도 안 만들어져요.
  - A: 경로에 한글/특수문자가 많으면 경로 입력이 실패할 수 있습니다. `data/` 폴더 내 파일로 먼저 시도하세요.
  - A: 최초 실행에서는 의존성 설치(Y) 후 재시도하세요.
- Q: 어디에 저장되는지 모르겠어요.
  - A: 각 메뉴 실행 후 화면에 경로가 출력됩니다. 기본은 `data/` 하위입니다.
- Q: 인터넷 없이도 되나요?
  - A: 네. 현재 제공되는 일기/독후감/창작/교재 데모/수학 증명은 모두 오프라인 동작이 가능하도록 구성되어 있습니다.

## 성장과 기록

- 모든 활동은 "체험 → 산출물 → KG 귀속" 흐름을 따릅니다.
- KG는 개념/관계와 경험 경로(`experience_*`)를 함께 저장해 학습의 근거를 남깁니다.

