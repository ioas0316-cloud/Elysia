<!-- version: paypal_revenue_v1 -->
# 💰 PayPal 매출 자동 분석

비즈니스 에이전트가 본인 PayPal 계정의 매출을 직접 분석. 일별/주별/월별 매출 + 통화별 + 환불 비율 + 최근 거래 마크다운 리포트.

## 한 번만 설정 — PayPal Developer App

### 1. PayPal Developer Dashboard
- 접속: https://developer.paypal.com/dashboard/applications
- 로그인 (PayPal Business 계정이 있어야 함)

### 2. 앱 생성
- **Apps & Credentials** 메뉴
- 처음 사용자 → 'Default Application' 이미 있음. 그거 써도 됨.
- 새 앱 원하면 **Create App** 클릭
- 앱 이름: "Connect AI Business Agent" 같은 식

### 3. 키 복사
- 앱 상세 페이지에서:
  - **Client ID** 복사
  - **Client Secret** 복사 (show 클릭해서 보기)
- 도구 설정에 붙여넣기

### 4. 권한 확인
앱 상세 페이지 하단 **Features** 섹션에서:
- ✅ **Transaction Search** 켜져있어야 함
- 안 켜져있으면 토글 ON

## 모드

| MODE | 용도 | URL |
|---|---|---|
| **sandbox** | 테스트 (가짜 계정·가짜 돈) | api-m.sandbox.paypal.com |
| **live** | 실제 운영 | api-m.paypal.com |

처음엔 **sandbox** 로 시작. 가짜 거래 만들어서 도구 동작 확인 후 live 전환.

샌드박스 거래 만들기: sandbox.paypal.com 에서 PayPal Developer 가 발급한 가짜 buyer/seller 계정으로 결제 시뮬레이션.

## 설정 (config)

| 키 | 의미 |
|---|---|
| `MODE` | `sandbox` 또는 `live` |
| `CLIENT_ID` | PayPal 앱 Client ID |
| `CLIENT_SECRET` | PayPal 앱 Client Secret (UI에서 password 필드로 가려짐) |
| `LOOKBACK_DAYS` | 분석할 과거 일수 (기본 30) |
| `CURRENCY` | 기본 통화 (USD/KRW/EUR). 비우면 모든 통화 표시 |

## 출력

마크다운 리포트:
- 통화별 매출 표 (Gross, 환불, 수수료, 순매출, 거래수)
- 기간별 매출 (오늘 · 지난 7일 · 지난 30일)
- 평균/최대/최소 거래액
- 최근 거래 10건 (일시·금액·종류)
- 환불율 경고 (10% 초과 시)
- 다음 액션 인사이트

## 사용 예시 (대화)

```
사용자: "비즈니스 에이전트, 우리 회사 PayPal 매출 어때?"
→ CEO → business 분배
→ business: <run_command>cd "..." && python3 paypal_revenue.py</run_command>
→ 결과 분석 + "이번 주가 평균 대비 +20% — 무엇이 잘됐는지 파악 필요" 같은 인사이트
```

## 한계

- PayPal Transaction Search API: 최근 3년 데이터까지
- 한 번 호출 = 최대 31일 × 500건 (자동 페이지네이션 처리)
- Rate limit: 무료 계정 분당 60 요청 — 일반 사용엔 충분
- 분쟁·세금·환율 변환은 안 함 (분석만)

## 보안

- `CLIENT_SECRET` 은 도구 설정 (password 필드) 에 저장. `.gitignore` 적용된 `_agents/business/tools/*.json` 에만 있음.
- API 호출은 Connect AI 익스텐션이 로컬에서 직접 → 외부 서버 경유 없음.
- token 메모리에만 존재, 디스크 저장 X.

## 다음 단계 (계획)

- Stripe·Toss 매출 통합 → 전체 결제 채널 한 리포트
- 일별 추세 그래프 (matplotlib)
- 월별 P&L 자동 생성 → `_company/_shared/pnl_<month>.md`
