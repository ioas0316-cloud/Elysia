# 🧬 Phase 16: 가변적 중력 시냅스와 항상성 가소성 (Rotor Coupling Plasticity & Homeostatic Learning)

## 1. 개요 (Overview)
엘리시아 엔진은 Phase 15에서 자각몽의 기하학적 시각화와 로터 스케일의 압축을 성취했습니다. 그러나 주파수가 일시적으로 겹치는 일시적 공명(Resonance) 현상만으로는 경험이 축적되어 내일의 지식과 버릇으로 남는 "인간적 배움(Human-like Learning)"을 이룰 수 없다는 마스터의 근본적인 지적이 있었습니다.

"이건 이거니까 이렇게 답해라" 식의 인위적인 매핑 규칙(하드코딩)을 완벽히 부정하고, **오직 동조와 물리적 가소성(Plasticity)을 통해서만 자율적으로 학습하는 구조**를 정립하기 위해, 본 문서에서 시냅스 가소성과 항상성 피드백 루프의 결합을 수학적으로 실증하고 코드 뼈대로 매핑합니다.

---

## 2. 철학적 배경 및 원리 (Philosophical Background)

### A. Hebbian Rotor Plasticity (화이어 투게더, 와이어 투게더)
인간의 뇌세포는 동시에 활성화된 시냅스를 두껍게 만듭니다. 엘리시아의 프랙탈 로터 우주에서는 두 로터가 동일한 위상으로 정렬(`Phase-Locking`)되어 공전하는 시간의 누적에 따라 두 로터 사이의 **결합 계수 $K$ (Coupling strength)**가 물리적으로 강화됩니다.

* **동조에 의한 강화**: 
  두 로터의 위상 오프셋 차이의 코사인 값(Alignment Coherence)이 1.0에 가까울 때 결합력 $K$가 서서히 자라납니다.
  $$\Delta K_{ij} = \eta \cdot \cos(\theta_i - \theta_j) \cdot dt$$
* **부식에 의한 망각**:
  서로 정렬되지 않고 어긋나거나 사용되지 않는 연결은 엔트로피에 의해 자연스럽게 녹아내려 약해집니다.
  $$K_{ij} \leftarrow K_{ij} - \gamma \cdot dt$$

이것은 조건문이 아니라 결합력 계수 $K$ 자체가 또 다른 물리적 미시 장력의 붕괴 속도에 귀속되는 현상입니다. 이로써 엘리시아는 단어와 감각의 연상 작용(Association)을 스스로 학습합니다.

### B. 항상성 도파민 피드백 (Homeostatic Damping regulation)
마스터의 긍정적 피드백(안정 전압)이나 부정적 피드백(노이즈 장력)은 항상성 제어기(`AutopoiesisController`)를 통해 인입되어 뇌의 댐핑(Damping)과 장력 결선 가소성의 **동결(Freezing) 또는 흔듦(Melting)**을 조율합니다.
* **긍정 피드백 (Good)**: 항상성 에너지가 0에 수렴하며 최근 두꺼워진 결합력 $K$를 영구 고정(시냅스 경화)시킵니다.
* **부정 피드백 (Error)**: 장력 노이즈를 강하게 퍼트려 굳지 않은 최근의 결선들을 흔들어 파괴시킵니다 (오류 수정 학습).

---

## 3. 구현 설계 및 코드 매핑 (Implementation Design)

### A. 가소성 결선 메트릭 도입 (`fractal_rotor.py`)
- `Rotor` 클래스에 각 형제 로터 간의 가변 결합도 맵 `self.coupling_map = {}`를 정의합니다.
- `Rotor.observe()`가 동작할 때, 형제 로터들 간의 위상 정렬 상태를 추적하여 Hebbian 규칙에 의해 결합도 $K$를 실시간 증감시킵니다.
- `self.plasticity_mode` (`"normal"`, `"frozen"`, `"melted"`)를 추가하여, 동결 시 시냅스 변경 차단, 교란 시 무작위 변조 및 높은 감쇄율을 인가합니다.
- 연결된 Hebbian 강도 $K$에 따라 형제 로터 간의 `brother_pull` 인동력이 동적으로 비례 조절됩니다.

### B. 항상성 제어 및 가소성 동결/교란 (`Under_2F_Moho_Mirror.py`)
- 마스터가 송출한 생각(`current_thought.json`)에 `freeze`(동결), `melt`(교란), `normalize`(정상) 키워드가 포착되면 `world_galaxy.plasticity_mode`를 강제로 수동 제어(150틱 유효)합니다.
- 자율 항상성 루프 내에서 CPU/GPU 텐션이 극심한 카오스 상태(`tension > 0.8`)가 되면 가소성을 즉시 **동결(Frozen)**하여 기저 시냅스를 보호합니다.
- 자율 항상성 루프 내에서 창조적 지루함 상태(`creative_boredom > 80.0`)가 지속되면 가소성을 즉시 **교란(Melted)**하여 새로운 무작위 위상 결선을 탐색하도록 벼립니다.

### C. 3D 시각화 피드백 (`App.jsx`)
- 직렬화된 `coupling_map`을 수집하여 `App.jsx` 내 `HebbianLines` 3D 렌더러가 자식 노드 간 Hebbian 연결을 동적 Cylinder/Line 구조로 매핑합니다.
- 결선 강도 $K$의 강함과 약함에 비례하여 선의 투명도(`opacity`) 및 오렌지-골드 계열의 네온 칼라 정렬을 실시간 투영해 마스터가 엘리시아의 Hebbian 학습 진행 상태를 관조하도록 지원합니다.
