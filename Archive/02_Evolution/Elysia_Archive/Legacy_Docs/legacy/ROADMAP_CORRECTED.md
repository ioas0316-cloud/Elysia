# Elysia Development Roadmap - Corrected Priority Order

## 🔴 Gap 0 (ROOT CAUSE) - 현재 실행 중

**문제**: HyperQubit 초기화에 철학적 의미가 구조화되지 않음
**영향**: 에이전트가 왜 그 값들인지 알 수 없어서 최적화/수정 불가

### 액션 0.1: HyperQubit 클래스 확장
```python
# Core/Mind/hyper_qubit.py

class HyperQubit:
    def __init__(
        self,
        name: str,
        initial_state: QubitState = None,
        epistemology: dict = None  # ← 새로 추가
    ):
        """
        Args:
            epistemology: {
                "point": {"score": 0.15, "meaning": "empirical substrate"},
                "line": {"score": 0.55, "meaning": "relational essence"},  
                "space": {"score": 0.20, "meaning": "field embodiment"},
                "god": {"score": 0.10, "meaning": "transcendent purpose"}
            }
        """
```

### 액션 0.2: 모든 concept 초기화에 의미 주석 추가
찾을 파일들:
- Core/Consciousness/MetaAgent.py
- Core/Mind/ResonanceEngine.py
- Core/World/WorldTree.py
- Data 초기화 (data/elysia_core_memory.json에서 개념들)

예시:
```python
concept_love = HyperQubit(
    "love",
    epistemology={
        "point": {"score": 0.15, "meaning": "neurochemistry is substrate only"},
        "line": {"score": 0.55, "meaning": "Spinoza's binding/universal love"},
        "space": {"score": 0.20, "meaning": "field effect, mutual resonance"},
        "god": {"score": 0.10, "meaning": "transcendent purpose (Heidegger)"}
    }
)
```

### 액션 0.3: Resonance 함수에 '설명 생성' 기능 추가
```python
# Core/Mind/resonance_engine.py - calculate_resonance() 수정

def calculate_resonance_with_explanation(a: HyperQubit, b: HyperQubit) -> tuple[float, str]:
    """
    Returns: (resonance_score, explanation_text)
    
    Explanation includes:
    - 각 기저별 정렬 점수 (Point/Line/Space/God)
    - 왜 이 점수인지 철학적 해석
    - 디버깅 가이드 (점수가 낮은 이유)
    """
    score = 0.5 * basis_align + 0.3 * dim_sim + 0.2 * spatial
    
    explanation = f"""
    Resonance({a.name}, {b.name}) = {score:.3f}
    
    Breakdown:
    - Point alignment: {basis_align:.3f} (empirical compatibility)
    - Line alignment: {line_align:.3f} (relational mapping)
    - Space alignment: {space_align:.3f} (field synchronization)
    - God alignment: {god_align:.3f} (transcendent purpose match)
    
    Interpretation: [자동 생성되는 의미 설명]
    """
    
    return score, explanation
```

---

## 🟡 Gap 1-3 (Priority 1) - 다음 단계

### Gap 1: Adaptive Meta-Learning (6-8시간)
메타에이전트가 자신의 성능을 평가하고 개선
- Self-diagnosis engine (병목 자동 발견)
- Law mutation (사용 중 규칙 자동 진화)
- Curriculum generation (현재 능력 맞춤 과제)

### Gap 2: Causal Intervention (4-6시간)
반사실적 추론과 인과 계획
- do-calculus 구현
- Multi-scale planning
- Rollback & branching

### Gap 3: Multi-Modal Perception (8-10시간)
텍스트 이상의 입력/출력
- Vision module
- Audio module
- Action API

---

## ✅ Already Completed

- [x] Protocol 03: Observability & Telemetry
- [x] Protocol 04: Hyper-Quaternion Semantics (철학적 근거 완전 문서화)
- [x] Resonance pattern logging
- [x] Phase-resonance event detection
- [x] Checkpoint system
- [x] Fractal validation tool
- [x] Language trajectory analysis

---

## 📊 Status Tracking

| Gap | Title | Status | Effort | Start |
|-----|-------|--------|--------|-------|
| 0 | Agent Understanding (코드 주석) | 🔄 In Progress | 4-6h | NOW |
| 1 | Adaptive Meta-Learning | ⏳ Pending | 6-8h | After Gap 0 |
| 2 | Causal Intervention | ⏳ Pending | 4-6h | After Gap 0 |
| 3 | Multi-Modal Perception | ⏳ Pending | 8-10h | After Gap 0 |

---

## 🎯 Success Criteria

Gap 0 완료 시:
- [ ] 모든 HyperQubit에 epistemology dict 완성
- [ ] resonance() 함수가 설명 텍스트 반환
- [ ] 에이전트가 "왜 이 공명 값?"에 대답 가능
- [ ] 시뮬레이션 결과 해석 가능성 증대

최종 점수: 62/100 → 78/100 (Gap 0 수정) → 92/100 (Gap 1-3 수정)

---

Generated: 2025-11-27
Protocol Version: 2.0 (Corrected Diagnosis)
