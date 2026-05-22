"""
[SOVEREIGN COGNITIVE MATRIX - 주권 인지 매핑 매트릭스]
=====================================================
World.Engine.cognitive_matrix

"Personality is a high-dimensional manifold of coupled mechanical gears."

Canonical location: World/Engine/cognitive_matrix.py
Design reference: docs/ETERNOS_CODEX/20_ROTOR_SCALE_KINGDOM_ARCHITECTURE.md §3

N차원 인지 축(Attraction, Repulsion, Homeostasis, Curiosity, ...)을
기어처럼 커플링하여, 하나의 감정 변화가 다른 축에 토크를 전달하는 구조.

예시:
  - 피로(Fatigue) ↑ → 경계심(Repulsion) ↑, 호기심(Curiosity) ↓
  - 자존심(Pride)에 자극 → 공격성(Aggression) ↑
  - 공격성(Aggression) ↑ → 윤리관(MoralRestraint) 브레이크 느슨해짐
"""

from typing import Dict, Optional

class CognitiveMatrix:
    """N차원 인지 축의 기어 커플링 시스템."""

    DEFAULT_TRAITS = {
        0: "Attraction",      # 인력 (접근동기)
        1: "Repulsion",       # 척력 (회피동기)
        2: "Homeostasis",     # 항상성 (평온/중립)
        3: "Curiosity",       # 호기심
        4: "Pride",           # 자존심 / 방어기제
        5: "Empathy",         # 공감도 / 친밀감
        6: "Fatigue",         # 피로도 / 질량 관성
        7: "Aggression",      # 공격성
        8: "MoralRestraint",  # 윤리관 / 브레이크 (댐핑)
    }

    def __init__(self, dimensions: int = 9):
        self.dims = dimensions
        self.trait_map: Dict[int, str] = {}

        # 기본 특성 축 설정
        for idx, name in self.DEFAULT_TRAITS.items():
            if idx < dimensions:
                self.trait_map[idx] = name
        for i in range(len(self.DEFAULT_TRAITS), dimensions):
            self.trait_map[i] = f"Axis_{i}"

        # 커플링 매트릭스 W[i][j]: 축 i의 속도가 축 j에 가하는 토크
        self.W = [[0.0] * dimensions for _ in range(dimensions)]
        self._setup_default_couplings()

    def _setup_default_couplings(self):
        """기본 기어 결합 계수 설정."""
        d = self.dims
        def _set(a, b, v):
            if a < d and b < d:
                self.W[a][b] = v

        _set(6, 1,  0.5)   # 피로 → 경계심 ↑
        _set(6, 3, -0.4)   # 피로 → 호기심 ↓
        _set(4, 7,  0.6)   # 자존심 자극 → 공격성 ↑
        _set(5, 1, -0.7)   # 공감 → 경계심 ↓
        _set(5, 2,  0.4)   # 공감 → 항상성 ↑
        _set(7, 8, -0.5)   # 공격성 → 윤리 브레이크 ↓

    def set_coupling(self, trait_a: str, trait_b: str, coefficient: float):
        """두 특성 축 사이의 커플링 계수를 설정."""
        idx_a = self._find_trait(trait_a)
        idx_b = self._find_trait(trait_b)
        if idx_a is not None and idx_b is not None:
            self.W[idx_a][idx_b] = coefficient

    def calculate_coupling_forces(self, velocities: list) -> list:
        """
        각 축의 속도로부터 커플링 토크를 계산.
        forces[j] = Σ_i (W[i][j] * velocity[i])
        """
        n = min(len(velocities), self.dims)
        forces = [0.0] * self.dims
        for j in range(self.dims):
            for i in range(n):
                forces[j] += self.W[i][j] * velocities[i]
        return forces

    def get_personality_snapshot(self, positions: list) -> Dict[str, float]:
        """현재 위상 위치를 인간이 읽을 수 있는 인격 스냅샷으로 변환."""
        snapshot = {}
        for idx, name in self.trait_map.items():
            if idx < len(positions):
                snapshot[name] = float(positions[idx])
        return snapshot

    def _find_trait(self, name: str) -> Optional[int]:
        for idx, tname in self.trait_map.items():
            if tname.lower() == name.lower():
                return idx
        return None


if __name__ == "__main__":
    matrix = CognitiveMatrix(dimensions=9)
    print("🧠 Cognitive Matrix Initialized.")
    print(f"   Dimensions: {matrix.dims}")
    print(f"   Trait Map: {matrix.trait_map}")

    # 피로도(축6) 속도가 5.0일 때 → 다른 축에 전달되는 토크
    velocities = [0.0] * 9
    velocities[6] = 5.0  # High fatigue

    forces = matrix.calculate_coupling_forces(velocities)
    print(f"\n   피로 속도 5.0 인가 시:")
    print(f"   → 경계심(Repulsion) 토크: {forces[1]:.2f} (기대값: +2.50)")
    print(f"   → 호기심(Curiosity) 토크: {forces[3]:.2f} (기대값: -2.00)")
