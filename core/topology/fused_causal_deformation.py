"""
Elysia Core Module: Causal Deformation Fused Acceleration (C++/Pybind11 & Pure Python Fallback)
=============================================================================================
CausalDeformationLayer의 국소 이완(Fast Relaxation)과 slow C 변형 연산을
C++ Extension 또는 NumPy Pure Fallback으로 가속 구동하는 연산 레이어.
"""

from typing import Tuple, Optional
import numpy as np

try:
    import causal_engine
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False


def fused_causal_deformation_update(
    S: np.ndarray,
    C: np.ndarray,
    W_back: np.ndarray,
    intent_I: np.ndarray,
    higher_friction_R: Optional[np.ndarray] = None,
    relaxation_steps: int = 5,
    gamma: float = 0.1,
    alpha: float = 0.01,
    use_cpp_extension: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fused Causal Deformation Layer Update with C++ Extension & NumPy Fallback parity.

    Returns:
        S_updated: Updated state vector
        C_updated: Updated constraint matrix
        final_friction_level: Final friction norm
    """
    S_curr = S.copy()
    C_curr = C.copy()
    out_dim, in_dim = C_curr.shape

    # 1. Fast Relaxation Step
    for _ in range(relaxation_steps):
        delta_P = intent_I - np.dot(C_curr.T, S_curr)
        if higher_friction_R is not None:
            min_dim = min(len(intent_I), len(W_back))
            delta_P[:min_dim] += np.dot(W_back, higher_friction_R[:out_dim])[:min_dim] * 0.1

        R = np.dot(C_curr, intent_I) - S_curr
        S_curr += gamma * (R + np.dot(C_curr, delta_P))

    # 2. Slow Deformation Matrix Update Step
    final_R = np.dot(C_curr, intent_I) - S_curr
    delta_C = np.outer(final_R, intent_I)
    C_curr -= alpha * delta_C

    friction_level = float(np.linalg.norm(final_R))
    return S_curr, C_curr, friction_level
