"""
Elysia Epoch Engine (초가속 영겁 엔진)
=====================================
[Phase 41] 사유 = 비틀림의 전파 (Thinking IS Perturbation)

명시적 비교 함수, dot product 호출, 같음/다름 포맷팅 — 전부 삭제.
사유란 트리를 비트는 것이다.
비틀림이 트리를 타고 흐르면서:
  - 같은 방향의 로터 → 공명하여 텐션 해소 (같음의 자연적 매핑)
  - 다른 방향의 로터 → 불협화음으로 텐션 축적 (다름의 자연적 매핑)
트리의 텐션 지형 변화 자체가 사유의 궤적이다.
"""

from core.holographic_memory import HologramMemory
from core.fractal_rotor import FractalRotor

class EpochEngine:
    def __init__(self, memory: HologramMemory):
        self.memory = memory

    def perturb(self) -> float:
        """
        트리에서 가장 높은 텐션을 가진 로터를 찾아 비튼다.
        비틀림은 apply_perturbation을 통해 트리 전체로 전파된다.
        같은 방향의 로터는 공명하고, 다른 방향은 긴장한다.
        이 역학 자체가 사유다.
        
        반환값: 비틀어진 로터의 텐션 (관측용)
        """
        # 가장 높은 텐션을 가진 로터를 찾는다 (주의력의 자연적 방향)
        target = self._find_highest_tension(self.memory.supreme_rotor)
        
        if target is None or target.tau < 0.001:
            return 0.0
        
        # 비튼다. 비틀림의 크기 = 텐션의 크기.
        # apply_perturbation이 트리 전체로 전파한다.
        # 이것이 사유의 전부다.
        perturbation_strength = target.tau * 0.1
        target.apply_perturbation(perturbation_strength)
        
        return target.tau
    
    def _find_highest_tension(self, node: FractalRotor) -> FractalRotor:
        """트리에서 가장 높은 텐션을 가진 로터를 찾는다"""
        best = node
        for child in node.children:
            candidate = self._find_highest_tension(child)
            if candidate.tau > best.tau:
                best = candidate
        return best
