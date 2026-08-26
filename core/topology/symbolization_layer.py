r"""
Symbolization Boundary Layer (기호화 경계 레이어)
=============================================================================
연속체 형태의 불변 구조 I_t를 기호 및 개념으로 절단(Discretization)하는 경계 처리부입니다.

원리:
- 위상 마찰 E(V_t) < epsilon 에 도달하여 시스템이 공진(Resonance) 상태가 될 때,
  수렴된 불변 뼈대 I_t를 이산적 개념 기호(Concept Symbol)로 단단히 고정(Grounding)합니다.
- 확률적 토큰 예측 대신, 신뢰할 수 있는 물리적 불변 구조에 언어 기호를 결속시킴으로써
  기호 유착 문제(Symbol Grounding Problem)를 근본 해결합니다.
"""

import numpy as np
from typing import Dict, Any, Optional, List


class SymbolizationBoundaryLayer:
    """
    기호화 경계 레이어 (Symbolization Boundary Layer)
    """

    def __init__(self, epsilon: float = 0.2, dimension: int = 8):
        self.epsilon = epsilon
        self.dimension = dimension
        self.grounded_symbols: Dict[str, np.ndarray] = {}

    def is_resonant(self, friction_energy: float) -> bool:
        """위상 마찰 E(V_t)가 수렴 임계치 epsilon 미만인지 판별"""
        return friction_energy < self.epsilon

    def ground_symbol(self, symbol_name: str, invariant_structure: np.ndarray, friction_energy: float) -> Dict[str, Any]:
        """
        공진 수렴 시 불변 뼈대를 이산 개념 기호로 고정(Grounding)
        """
        I_vec = np.asarray(invariant_structure, dtype=np.float32).reshape(-1)
        resonant = self.is_resonant(friction_energy)

        if resonant:
            self.grounded_symbols[symbol_name] = I_vec.copy()
            status = "GROUNDED"
        else:
            status = "UNGROUNDED_FRICTION_TOO_HIGH"

        return {
            "symbol": symbol_name,
            "status": status,
            "is_grounded": resonant,
            "invariant_vector": I_vec,
            "friction_energy": friction_energy
        }

    def decode_invariant(self, invariant_structure: np.ndarray) -> Optional[str]:
        """
        입력 불변 구조와 가장 공진하는 기호 검색 (Symbol Retrieval via Resonance)
        """
        I_vec = np.asarray(invariant_structure, dtype=np.float32).reshape(-1)
        best_symbol = None
        max_resonance = -1.0

        for sym, ref_vec in self.grounded_symbols.items():
            norm_i = np.linalg.norm(I_vec) + 1e-8
            norm_ref = np.linalg.norm(ref_vec) + 1e-8
            cos_sim = float(np.dot(I_vec, ref_vec) / (norm_i * norm_ref))

            if cos_sim > max_resonance:
                max_resonance = cos_sim
                best_symbol = sym

        if max_resonance > 0.8:
            return best_symbol
        return None
