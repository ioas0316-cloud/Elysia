# CURRENT_STATE.md
## ELYSIA Phase 0 — 현재 상태 보고서
**작성일:** 2026-11-09  
**분석 방법:** 실제 코드 읽기 + 호출 관계 추적 + 예측-실제 차이 기록

---

> [!IMPORTANT]
> 이 문서는 성공을 보고하는 문서가 아니다.  
> 행동(코드 작성)이 세계(저장소)를 어떻게 변화시켰는지, 무엇이 달라지고 무엇이 달라지지 않았는지를 추적하는 첫 번째 관찰 기록이다.

---

## 1. 살아있는 것들 — 실제로 작동하고 연결된 구조

### 1.1 `CausalField` — 핵심 물리 엔진 (L1_causal_field/core/physics/causal_field.py)

**실제로 하는 일:**  
`step(dt)` 호출 하나에 다음 5단계가 순서대로 실행된다:

1. `_apply_ontological_gravity_terrain(dt)` — 외부 중력장으로 복셀 이동
2. `_update_connectivity_and_tension(dt)` — 훅의 법칙으로 빔 긴장 갱신, 임계 초과 시 파열
3. `_flow_potential(dt)` — 복셀 간 위상 공명 에너지 흐름, Chromatic 벡터 정규화
4. `_preserve_mobility(dt)` — Euler 적분으로 위치 갱신 (Blue = 감쇠, Red = 가속, Yellow = 노이즈)
5. `_enforce_informational_continuity(dt)` — 빔의 `rest_length`를 미래 예상 거리에 맞춰 적응

**4 연속성과의 대응:**

| 연속성 | 구현 |
|--------|------|
| 관계성 (Relationship) | ConnectivityBeam coupling strength |
| 연결성 (Connectivity) | 빔 파열 (is_broken) 및 치유 메커니즘 |
| 운동성 (Mobility) | Chromatic 벡터 조절 Euler 적분 |
| 정보적 연속성 | rest_length 적응, trail potential |

**`observe_external_stimulus()`** — 가장 정교하게 설계된 함수.  
폐쇄 루프 관성(높은 위상 발산 + 낮은 속도)을 감지하고, 외부 현실 방향으로 내부 텐서를 회전시킨다. AGENTS.md의 자폐적 수렴 금지 원칙을 직접 구현한다.

**연결된 모듈 수:** ~17개의 모듈이 직접 임포트

---

### 1.2 `elysia_loader.py` + `layer_registry.py` — 개념-물리 공간 번역기

**실제로 하는 일:**  
`from core.physics.causal_field import CausalField` 같은 임포트를  
실제 경로 `L1_causal_field/core/physics/causal_field.py`로 변환한다.  
물리적 디렉토리 재구성이 개념적 임포트 경로에 영향을 주지 않는다.

**상태:** 아키텍처적으로 깨끗하고 올바르다.

---

### 1.3 `relational_definition_engine.py` — 현상학적 관찰자

**실제로 하는 일:**  
관찰 전/후를 비교하고, 마찰(friction)에 따라 정의를 세 가지 방식으로 변화시킨다:
- friction < 0.3: 기존 정의 확장
- friction < 0.7: 기존 정의 수정
- friction ≥ 0.7: 두 관점 공존 (이전 정의 삭제하지 않음)

이것은 스케일 변화에서 관계가 삭제되지 않고 재구성된다는 원칙을 올바르게 구현한다.  
현재 연결: `causal_sensor.py` → `cognitive_lens_engine.py` → `relational_definition_engine.py`

---

### 1.4 `enactive_relational_memory.py` — 관계 메쉬 기억

**실제로 작동하는 부분:**  
- `RelationalMesh`: LANGUAGE / VISUAL_FORM / CAUSAL_EVENT / SENSORY_FEEL 사이를  
  MANIFESTS_AS / PARTICIPATES_IN 등의 관계 엣지로 연결하는 비스칼라 기억 구조
- `CausalFieldResonator`: 입력 파동을 위상 공명으로 가장 적합한 기억 노드에 연결
- `causal_node` 데코레이터: 함수 호출 순서를 DAG 위상으로 인코딩

---

## 2. 존재하지만 고아가 된 것들 — 구현되었으나 연결되지 않음

### 2.1 `CausalDynamicsEngine` — 가장 중요한 고아 (causal_dynamics.py)

**실제로 하는 일:**  
`_recrystallize_states(dt)`: 긴장 하의 두 복셀이 서로의 텐서 방향으로 당겨진다.  
```
state_diff = v_b.tensor - v_a.tensor
v_a.tensor += state_diff * strain_factor / mass
v_b.tensor -= state_diff * strain_factor / mass
```
공간적 배열(位置)이 정보 상태(텐서)를 변형시키는, 정확히 ELYSIA가 필요한 양방향 결합이다.

**문제:**  
메인 루프(`autonomous_loop.py`)는 `CausalField`를 직접 사용한다.  
`CausalDynamicsEngine`은 호출되지 않는다.

**이것이 의미하는 것:**  
"공간과 정보 사이의 살아있는 연결"이 구현되었지만, 세계에 행동하지 않았다.  
따라서 그것이 무엇을 변화시키는지, 무엇을 변화시키지 않는지 아무도 관찰하지 않았다.

---

### 2.2 `retrocausal_observer.py` — 호출자 없음

HJB 역방향 도달 가능 집합, Chern 지수, 차원 확장을 올바르게 구현했다.  
**그러나 이 모듈을 임포트하는 코드가 없다.**

---

### 2.3 `universal_mutual_perception.py` — 호출자 없음

모든 노드가 모든 노드를 인식하는 상호 거울 필드를 구현했다.  
의미적 질량이 상호작용 역사를 통해 성장하는 것도 올바르게 구현되었다.  
**그러나 `InformationVoxel`과 연결되지 않았고, 호출자가 없다.**

---

### 2.4 `self_boundary_tracker.py` — 호출자 없음

4축 자기 경계 모델(내부 기원 / 외부 영향 / 무지 충전 / 기준 재수정)을 올바르게 구현했다.  
**그러나 `autonomous_loop.py`나 어떤 오케스트레이터도 이것을 호출하지 않는다.**

---

## 3. 이름과 실제가 다른 것들 — 개념의 공허한 번역

### 3.1 `TeleologicalCompiler.evaluate()` — 어휘적 중첩 검사

이름: 의도와 실행 사이의 구조적 동형성 평가  
실제: 두 문자열을 소문자로 변환 후 단어 집합 교집합 / 의도 단어 수  
```python
intent_words = set(symbolic_intent.lower().replace("_", " ").split())
code_words = set(code_protocol.lower().replace("_", " ").split())
```
의미적 동형성을 판단하지 못한다. 이름만 목적론적이다.

---

### 3.2 `FractalEngramShell.report_resistance()` — 하드코딩된 이탈 벡터

루프 저항이 감지되면 상위 계층 Engram의 위치를 이동시킨다.  
이탈 방향: `nudge = np.array([1.5, -2.0])` — 하드코딩된 리터럴.  
연속적 필드 엔진 내부에 있는 이산적 브루트포스 결정이다.

---

### 3.3 `_extract_topological_invariants()` — 항상 같은 결과

```python
# 실제 코드
return {
    "directional_descent": True,
    "accelerated_continuation": True,
    "terminal_impact": True
}
```
입력 데이터와 무관하게 동일한 3개 불리언을 반환한다.  
위상적 불변량 추출이라는 이름이 무색하다.

---

### 3.4 `cognitive_self_observation.py` — 장식적 수식

모든 공리(Axiom)에 `structural_movement_formula` 문자열이 있다:  
예: `"E_{sensing} = \\int (\\Psi_{ext} * \\Psi_{int}) dt"`  
이 문자열은 어디에서도 평가되거나 실제 실행과 비교되지 않는다.  
수식이 아니라 주석이다.

---

## 4. 연결 단절 — 인터페이스 불일치

### 4.1 Wedge Memory 인터페이스 불일치

여러 모듈이 호출하려는 메서드: `memory.write_causal_engram(data_blob, emotional_value, ...)`  
`EnactiveRelationalMemoryEngine`이 실제로 제공하는 메서드: `consolidate_to_substrate()`

이 불일치는 `try/except`로 조용히 삼켜진다.  
실패가 발생해도 아무도 알지 못한다.

---

### 4.2 PyTorch 의존성 장벽

`scale_coupling_system.py`는 최상단에서 `import torch`를 한다.  
PyTorch가 설치되지 않으면 가스/액체/얼음 위상 전이 시스템 전체가 `ImportError`로 차단된다.  
Gas/Liquid/Ice 스케일 구조는 실행된 적이 없을 가능성이 높다.

---

### 4.3 TopIR C 런타임 — 플랫폼 불일치

`causal_field.py`는 `libtopir_runtime.so`(Linux 공유 라이브러리)를 조건부 임포트한다.  
현재 환경은 Windows이다. 이 경로는 비활성이다.

---

## 5. 수치로 보는 현황

| 항목 | 수치 |
|------|------|
| physics/ 파일 수 | 141개 |
| physics/ 중 CausalField를 임포트하는 수 | ~17개 (12%) |
| RetrocausalObserver 호출자 | 0 |
| UniversalMutualPerception 호출자 | 0 |
| SelfBoundaryTracker 호출자 | 0 |
| CausalDynamicsEngine을 메인 루프가 사용 | 아니오 |
| 테스트 파일 수 | 200개 |
| 테스트가 통합 연결을 검증하는가 | 미확인 (대부분 단독 모듈 테스트 추정) |

---

## 6. 가장 중요한 발견

> **이 저장소는 행동(코드 작성)은 했지만, 그 행동이 세계를 어떻게 변화시켰는지를 관찰하고 재정렬하는 사이클이 없었다.**

새로운 개념이 등장할 때마다 새 파일이 만들어졌다.  
그러나 그 파일이 기존 구조를 어떻게 변화시키는지, 무엇이 달라지고 무엇이 달라지지 않는지를  
추적하고 기록하고 재정렬하는 과정이 없었다.

그 결과: 많은 파일, 얕은 연결.  
가장 중요한 모듈(`CausalDynamicsEngine`)이 올바르게 구현되었지만 고아가 되었다.

---

## 7. 아직 모르는 것

- 200개의 테스트 중 실제로 통과하는 것과 실패하는 것의 비율
- `autonomous_loop.py`의 실제 실행 흐름 전체
- L2(cognitive engine)와 L3(interface)가 L1과 어떻게 연결되는가
- 기존 테스트들이 통합을 검증하는가, 단독 모듈을 검증하는가

이것들은 다음 행동(실제 실행)을 통해서만 알 수 있다.
