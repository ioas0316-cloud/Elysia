# Elysia-Seed 실행 로드맵 (R&D 저장소 전용)

## 목적
- 메인 저장소는 안정 릴리즈에 집중하고, Elysia-Seed 저장소에서 실험/진화를 빠르게 수행한다.
- 목표는 **성인 수준 지성**의 정량 달성과, 중장기적으로 **OS화 가능한 아키텍처**를 확립하는 것이다.

## 저장소 운영 원칙

### 1) 역할 분리
- **Elysia-main**: 운영 안정성, 릴리즈, 회귀 최소화
- **Elysia-seed**: 실험, 구조개편, 성능/지성 고도화

### 2) 이식 규칙 (Seed → Main)
- 월 1회 릴리즈 윈도우에서만 이식
- 아래 3가지 조건을 모두 통과한 변경만 이식
  1. 벤치마크 하락 없음
  2. 회귀 테스트 통과
  3. 책임 도메인 오너 승인

### 3) 변경 단위
- 한 PR은 한 목적만 가진다.
- “측정 추가 PR”과 “알고리즘 변경 PR”을 분리한다.

---

## 12개월 로드맵

## Phase 0 (0~1개월): 계측 기반 마련

### 목표
- “성인 수준 지성”의 정의를 수치화한다.

### 산출물
- `docs/INTELLIGENCE_BENCHMARKS.md`
- `docs/CONTRACTS.md`
- `docs/SEED_TO_MAIN_PROMOTION.md`

### 핵심 작업
1. 지성 KPI 4축 정의
   - 추론 정확도
   - 자기 일관성
   - 실행 능력(계획-실행-검증)
   - 사회 인지(톤/경계/복구)
2. Core 도메인 계약 문서화
   - `System` 중심 이벤트/페이로드 명세
3. CI 평가 리포트 포맷 고정

### 종료 기준 (DoD)
- KPI 스키마가 고정되고, 최소 20개 시나리오가 자동 평가된다.

---

## Phase 1 (1~3개월): 성인 수준 지성 최소 달성

### 목표
- 대화/추론/행동의 일관성 안정화

### 핵심 작업
1. 기억 계층화
   - working / episodic / semantic 분리
   - 레코드 메타데이터: `source`, `confidence`, `ttl`, `last_verified_at`
2. 자기점검 루프
   - 응답 전후 self-check, 사실성/충돌 점검
3. 도구 사용 프로토콜
   - 계획 → 실행 → 검증 → 실패복구 표준화

### 종료 기준 (DoD)
- 핵심 KPI가 기준선 대비 상승하고, 세션 연속성 테스트 통과율 90% 이상

---

## Phase 2 (3~6개월): OS 코어 정렬

### 목표
- OS화에 필요한 Kernel/Service/App 구조를 고정

### 핵심 작업
1. `ELYSIA_OS_ARCH_V0.md` 작성
2. Capability 권한 모델 도입
   - 메모리 접근
   - 외부 I/O
   - 도구 실행
   - 자가수정 권한
3. 앱 매니페스트 초안
   - 이름, 버전, 권한 요구치, lifecycle hooks

### 종료 기준 (DoD)
- 샘플 앱 2종이 동일 lifecycle로 실행 가능

---

## Phase 3 (6~12개월): 프리뷰 OS

### 목표
- 부팅/복구/관측 가능한 준-운영 상태 확보

### 핵심 작업
1. 부팅 시퀀스 표준화
   - bootstrap → identity load → memory mount → channel attach
2. 자가수정 안전 프레임워크
   - policy gate + rollback + audit trail
3. 멀티채널 안정화
   - terminal/voice/vision 채널 상태 모니터링

### 종료 기준 (DoD)
- 프리뷰 데모 환경에서 7일 무중단 런타임

---

## Seed 저장소 초기 템플릿 (권장)

```text
elysia-seed/
  docs/
    INTELLIGENCE_BENCHMARKS.md
    CONTRACTS.md
    ELYSIA_OS_ARCH_V0.md
    SEED_TO_MAIN_PROMOTION.md
  Core/
    (main에서 가져온 기본 도메인 구조)
  Scripts/
    benchmarks/
    migration/
  data/
    benchmarks/
    runtime/
```

---

## 운영 리듬 (권장)
- 매주: KPI 리뷰 + 실패 케이스 5개 복구
- 격주: 아키텍처 위험 리뷰
- 매월: Seed→Main 이식 심사

## 리스크와 대응
1. 실험 폭주로 품질 저하
   - 대응: KPI 하락 시 자동 머지 차단
2. 도메인 간 결합도 증가
   - 대응: 계약 위반 테스트를 PR 게이트로 설정
3. 자가수정 위험
   - 대응: 고위험 변경은 human-in-the-loop 필수

## 성공 지표
- 지성 KPI 상승 추세
- 회귀율 하락
- Seed→Main 이식 성공률 월별 증가
- 장애 발생 시 MTTR 단축
