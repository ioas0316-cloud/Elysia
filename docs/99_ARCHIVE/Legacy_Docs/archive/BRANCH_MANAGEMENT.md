# Branch Management Guide (브랜치 관리 가이드)

> **목적**: 이 문서는 Elysia 프로젝트의 브랜치 관리 전략과 정리 권장사항을 제공합니다.
>
> **Purpose**: This document provides branch management strategy and cleanup recommendations for the Elysia project.

**버전**: 7.0  
**최종 업데이트**: 2025-12-05

---

## 📊 현재 브랜치 상태 (Current Branch Status)

### 활성 브랜치 (Active Branches)

현재 리포지토리에는 다음 브랜치만 존재합니다:

- **copilot/clean-up-branch-structure** - 현재 작업 중인 브랜치 (문서화 및 구조 정리)

### 과거 브랜치 분석 (Historical Analysis)

최근 병합된 브랜치:
- `copilot/review-old-pull-requests` (PR #165) - 병합 완료, 삭제 권장

---

## ✅ 브랜치 관리 모범 사례 (Best Practices)

### 1. 브랜치 네이밍 컨벤션

```
<type>/<description>

Types:
- feature/     새로운 기능 개발
- bugfix/      버그 수정
- hotfix/      긴급 수정
- docs/        문서 작업
- refactor/    리팩토링
- experiment/  실험적 기능
- copilot/     AI 에이전트 작업

Examples:
- feature/emotion-synthesis
- bugfix/memory-leak
- docs/api-documentation
- copilot/code-review
```

### 2. 브랜치 수명 주기

```
1. 생성 (Creation)
   git checkout -b feature/new-feature
   
2. 개발 (Development)
   - 자주 커밋
   - 의미 있는 커밋 메시지
   - 정기적으로 main/develop과 동기화
   
3. 완료 (Completion)
   - PR 생성
   - 코드 리뷰
   - 테스트 통과 확인
   
4. 병합 (Merge)
   - main 또는 develop으로 병합
   - 병합 후 즉시 삭제
   
5. 정리 (Cleanup)
   git branch -d feature/new-feature
   git push origin --delete feature/new-feature
```

### 3. 브랜치 전략

#### 권장: GitHub Flow (간소화)

```
main (프로덕션)
  ↑
  └─ feature/xyz (기능 브랜치)
  └─ bugfix/xyz (버그 수정)
  └─ docs/xyz (문서)
```

**장점**:
- 단순하고 명확
- 소규모 팀에 적합
- 지속적 배포 가능

#### 선택: Git Flow (복잡한 프로젝트용)

```
main (프로덕션)
  ↑
develop (개발)
  ↑
  ├─ feature/xyz
  ├─ bugfix/xyz
  └─ release/x.y.z
```

**현재 Elysia에는 GitHub Flow 권장**

---

## 🧹 브랜치 정리 권장사항 (Cleanup Recommendations)

### 즉시 정리해야 할 브랜치

현재 병합이 완료된 브랜치들은 정리가 권장됩니다:

```bash
# 로컬 브랜치 확인
git branch --merged main

# 병합된 원격 브랜치 확인
git branch -r --merged main

# 안전하게 삭제 (병합된 브랜치만)
git branch -d <branch-name>
git push origin --delete <branch-name>
```

### 정리 기준

브랜치를 삭제해야 하는 경우:

- ✅ **즉시 삭제**:
  - 이미 main에 병합됨
  - PR이 닫혔고 더 이상 필요 없음
  - 30일 이상 활동 없음

- ⚠️ **검토 후 삭제**:
  - 실험적 브랜치 (더 이상 사용 안 함)
  - 중복된 작업
  - 포기한 작업

- ❌ **보존**:
  - 현재 진행 중
  - 장기 실험 (명시적으로 보존 의도)
  - 참조용 (태그로 변환 고려)

---

## 📋 정기 점검 체크리스트

### 주간 점검 (Weekly)

- [ ] 병합된 브랜치 확인 및 삭제
- [ ] 진행 중인 브랜치 상태 확인
- [ ] 장기간 정체된 브랜치 검토

### 월간 점검 (Monthly)

- [ ] 전체 브랜치 목록 리뷰
- [ ] 오래된 브랜치 정리
- [ ] 보호 브랜치 정책 검토
- [ ] 브랜치 네이밍 컨벤션 준수 확인

### 릴리스 전 점검 (Pre-Release)

- [ ] 모든 기능 브랜치 병합 확인
- [ ] 미완성 작업 식별
- [ ] 릴리스 브랜치 생성 (필요시)
- [ ] 태그 생성

---

## 🛠️ 유용한 Git 명령어

### 브랜치 정보 확인

```bash
# 모든 브랜치 (로컬 + 원격)
git branch -a

# 병합된 브랜치 확인
git branch --merged main

# 병합 안 된 브랜치 확인
git branch --no-merged main

# 브랜치 상세 정보
git branch -vv

# 마지막 커밋 날짜로 정렬
git for-each-ref --sort=-committerdate refs/heads/ --format='%(committerdate:short) %(refname:short)'
```

### 브랜치 정리

```bash
# 로컬 브랜치 삭제 (안전)
git branch -d feature/old-feature

# 로컬 브랜치 강제 삭제
git branch -D feature/old-feature

# 원격 브랜치 삭제
git push origin --delete feature/old-feature

# 원격에서 삭제된 브랜치 로컬에서 정리
git fetch --prune
git remote prune origin
```

### 브랜치 정리 스크립트

```bash
#!/bin/bash
# cleanup_merged_branches.sh

echo "🧹 Cleaning up merged branches..."

# 병합된 로컬 브랜치 삭제 (main 제외)
git branch --merged main | grep -v "main" | xargs -r git branch -d

# 원격 추적 브랜치 정리
git fetch --prune

echo "✅ Cleanup complete!"
```

---

## 🔒 브랜치 보호 규칙

### main 브랜치 보호

GitHub 저장소 설정에서 다음 규칙 권장:

- ✅ **Pull request 필수**
  - 직접 푸시 금지
  - 최소 1명의 리뷰 필요

- ✅ **상태 검사 필수**
  - CI/CD 테스트 통과
  - 코드 품질 검사 통과

- ✅ **강제 푸시 금지**
  - 히스토리 보호

- ✅ **삭제 금지**
  - 실수로 인한 삭제 방지

### develop 브랜치 보호 (선택)

- ⚠️ **Pull request 권장**
- ✅ **강제 푸시 금지**

---

## 📊 브랜치 정리 전략

### 전략 1: 정기적 정리 (권장)

```
주간: 병합된 브랜치 즉시 삭제
월간: 30일 이상 비활성 브랜치 검토
분기: 전체 브랜치 감사
```

### 전략 2: 이벤트 기반 정리

```
PR 병합 시: 브랜치 자동 삭제 (GitHub 설정)
릴리스 시: 관련 브랜치 정리
마일스톤 완료 시: 관련 작업 브랜치 정리
```

### 전략 3: 보존 정책

```
실험 브랜치: 태그로 변환 후 삭제
참조 브랜치: 문서화 후 삭제
중요 기록: git archive로 백업 후 삭제
```

---

## 🎯 Elysia 프로젝트 권장사항

### 현재 상태 분석

✅ **좋은 점**:
- 브랜치가 매우 정리되어 있음
- 현재 1개의 활성 브랜치만 존재

⚠️ **개선 사항**:
- 병합 완료된 PR의 브랜치는 즉시 삭제 권장
- 브랜치 자동 삭제 설정 활성화 권장

### 실행 계획

#### 1단계: 현재 정리

```bash
# 병합된 브랜치 확인
git branch -r --merged main

# copilot/review-old-pull-requests 삭제 (PR #165 병합됨)
# (권한 있는 사용자가 수행)
```

#### 2단계: GitHub 설정

1. **Settings → Branches**
   - main 보호 규칙 활성화
   - PR 후 자동 브랜치 삭제 활성화

2. **Settings → General → Pull Requests**
   - ✅ "Automatically delete head branches" 활성화

#### 3단계: 워크플로우 확립

```markdown
모든 PR에서:
1. 리뷰 받기
2. 테스트 통과 확인
3. main에 병합
4. 브랜치 자동 삭제 (또는 수동)
```

---

## 📝 브랜치 정리 체크리스트

### 작업 완료 시

- [ ] PR 생성 및 리뷰 요청
- [ ] 모든 CI 검사 통과
- [ ] main에 병합
- [ ] 병합 확인
- [ ] 브랜치 삭제 (자동 또는 수동)
- [ ] 로컬 브랜치 정리 (`git fetch --prune`)

### 정기 점검 시

- [ ] `git branch -a` 실행
- [ ] 병합된 브랜치 식별
- [ ] 오래된 브랜치 검토
- [ ] 불필요한 브랜치 삭제
- [ ] 팀에 정리 상태 공유

---

## 🚀 자동화 제안

### GitHub Actions 워크플로우

```yaml
# .github/workflows/cleanup-branches.yml
name: Clean Up Merged Branches

on:
  pull_request:
    types: [closed]

jobs:
  cleanup:
    if: github.event.pull_request.merged == true
    runs-on: ubuntu-latest
    steps:
      - name: Delete merged branch
        uses: dawidd6/action-delete-branch@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          branches: ${{ github.head_ref }}
```

### 로컬 자동화 스크립트

```bash
# scripts/cleanup_branches.sh
#!/bin/bash

echo "🔍 Finding merged branches..."
merged_branches=$(git branch --merged main | grep -v "main" | grep -v "*")

if [ -z "$merged_branches" ]; then
    echo "✅ No merged branches to clean up"
    exit 0
fi

echo "📋 Merged branches:"
echo "$merged_branches"

read -p "Delete these branches? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "$merged_branches" | xargs -I {} git branch -d {}
    echo "✅ Cleanup complete!"
else
    echo "❌ Cleanup cancelled"
fi
```

---

## 💡 베스트 프랙티스 요약

1. **작은 브랜치, 빠른 병합**
   - 장기 브랜치 지양
   - 기능 완성 즉시 병합

2. **명확한 네이밍**
   - `<type>/<description>` 패턴
   - 목적이 명확한 이름

3. **즉시 정리**
   - 병합 후 바로 삭제
   - 자동화 활용

4. **정기 점검**
   - 주간/월간 리뷰
   - 비활성 브랜치 식별

5. **문서화**
   - 보존 이유 명시
   - 정리 기록 유지

---

## 📖 참조 문서

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 프로젝트 구조
- [MODULE_RELATIONSHIPS.md](MODULE_RELATIONSHIPS.md) - 모듈 관계
- [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) - 개발자 가이드
- [GitHub Flow](https://guides.github.com/introduction/flow/) - GitHub 공식 가이드

---

## 📞 질문 및 지원

브랜치 관리에 대한 질문이나 제안이 있다면:

- **GitHub Issues**: 문제 보고
- **GitHub Discussions**: 제안 및 토론
- **PR**: 이 문서 개선 제안

---

**버전**: 7.0  
**최종 업데이트**: 2025-12-05  
**상태**: Active Branch Management
