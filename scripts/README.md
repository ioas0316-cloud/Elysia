# scripts/ — 실험 & 관찰 스크립트

이 디렉토리는 엘리시아 엔진의 **실험, 관찰, 검증** 목적의 스크립트들을 카테고리별로 분류합니다.
`core/` 의 실제 엔진 코드와 구분되며, 실험적 성격의 코드가 여기에 위치합니다.

---

## 📁 디렉토리 구조

```
scripts/
├── observe/         ← 공명/상태 관찰 스크립트 (observe_*.py, experience_*.py)
├── poc/             ← Proof-of-Concept 실험 스크립트 (apple_resonance, black_box 등)
├── verify/          ← 단위 검증 스크립트 (causal_substitution_verify 등)
├── genesis/         ← 창생 루프 실험 스크립트 (genesis, volition_genesis_loop 등)
│
├── inject_foundations.py   ← 기반 지식 주입 (공용 — 루트 유지)
├── system_sensory_ingestion.py  ← 시스템 감각 데이터 수집 (공용 — 루트 유지)
└── read_log.py             ← 관찰 로그 읽기 유틸 (공용 — 루트 유지)
```

---

## 🗂️ 서브폴더 설명

| 폴더 | 설명 | 주요 파일 |
|------|------|----------|
| `observe/` | 엔진 내부 상태/공명/인과를 관찰하는 스크립트 | `observe_causal_backtracking.py`, `experience_resonance.py` |
| `poc/` | 특정 아이디어를 빠르게 검증하는 POC 스크립트 | `apple_resonance_poc.py`, `black_box_transparency_poc.py` |
| `verify/` | 특정 수학적/물리적 성질을 단위 검증하는 스크립트 | `causal_substitution_verify.py` |
| `genesis/` | 엘리시아 자율 창생 루프 관련 실험 | `genesis.py`, `volition_genesis_loop.py` |

---

> **주의:** 이 폴더의 코드들은 실험 목적입니다. 프로덕션 엔진 코드는 `core/` 에 있습니다.
