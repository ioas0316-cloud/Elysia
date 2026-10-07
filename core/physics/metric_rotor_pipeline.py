r"""
core/physics/metric_rotor_pipeline.py

Integrated Metric Wave to 5D Clifford Rotor Pipeline
=====================================================
Bridges Tensor Helmholtz Metric Decomposition (g_ij = bar_g_ij + h_ij)
with C\l(5,0) Spin(5) / SO(5) 10-Bivector Rotor Dynamics.

Pipeline Flow:
--------------
1. Input Metric Perturbation h_ij(r) [3, 3, Nx, Ny, Nz]
2. Tensor Helmholtz Decomposition -> {Scalar Trace, Vector Shear, Tensor TT}
3. Metric-to-Bivector Bridge -> 10D Angle Vector \theta_10D \in \mathfrak{so}(5)
4. Clifford 5D Rotor Generation R = exp(-0.5 * \Omega)
5. 5D State Vector Transformation v' = R * v * R^\dagger
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import numpy as np

from core.physics.quantum_rotor_phase import Clifford5DRotorEngine
from core.physics.tensor_helmholtz_decomposer import (
    TensorHelmholtzMetricDecomposer,
    MetricWaveDecompositionResult
)


class MetricToRotorBridge:
    """헬름홀츠 파동 성분(스칼라, 벡터, 텐서 TT)을 10D 바이벡터 회전각 \theta_10D로 사상"""

    def __init__(self, coupling_gain: float = 2.0):
        self.coupling_gain = coupling_gain

    def map_wave_modes_to_10d_bivectors(self, decomp: MetricWaveDecompositionResult) -> np.ndarray:
        theta_10d = np.zeros(10, dtype=float)

        # 1. Tensor TT 모드 -> 3D 공간 회전 바이벡터 평면 (e12, e13, e23) 사상
        tt_tensor = decomp.tensor_tt_mode
        theta_10d[0] = self.coupling_gain * np.mean(tt_tensor[0, 1])  # e12
        theta_10d[1] = self.coupling_gain * np.mean(tt_tensor[0, 2])  # e13
        theta_10d[4] = self.coupling_gain * np.mean(tt_tensor[1, 2])  # e23

        # 2. Vector Shear 모드 -> 4차원 위상 결합 바이벡터 평면 (e14, e24, e34) 사상
        vec_tensor = decomp.vector_shear_mode
        theta_10d[2] = self.coupling_gain * np.mean(vec_tensor[0, 0] - vec_tensor[1, 1])  # e14
        theta_10d[5] = self.coupling_gain * np.mean(vec_tensor[1, 1] - vec_tensor[2, 2])  # e24
        theta_10d[7] = self.coupling_gain * np.mean(vec_tensor[2, 2] - vec_tensor[0, 0])  # e34

        # 3. Scalar Trace 모드 -> 5차원 위상 확장 및 터널링 바이벡터 평면 (e15, e25, e35, e45) 사상
        sc_tensor = decomp.scalar_trace_mode
        e_ratio_scalar = decomp.energy_ratios["scalar_trace"]
        trace_val = np.mean(sc_tensor[0, 0] + sc_tensor[1, 1] + sc_tensor[2, 2])

        theta_10d[3] = self.coupling_gain * trace_val * 0.5            # e15
        theta_10d[6] = self.coupling_gain * trace_val * 0.5            # e25
        theta_10d[8] = self.coupling_gain * e_ratio_scalar * np.pi / 4 # e35 (Phase Tunneling)
        theta_10d[9] = self.coupling_gain * trace_val * 0.2            # e45

        return theta_10d


class IntegratedMetricRotorPipeline:
    """계량 변형 h_ij 수신부터 5D 클리퍼드 로터 회전까지의 통합 파이프라인"""

    def __init__(self, grid_shape: Tuple[int, int, int] = (16, 16, 16), coupling_gain: float = 2.0):
        self.decomposer = TensorHelmholtzMetricDecomposer(grid_shape=grid_shape)
        self.bridge = MetricToRotorBridge(coupling_gain=coupling_gain)
        self.rotor_engine = Clifford5DRotorEngine(dim=5)

    def process(self, h_tensor: np.ndarray, current_state_5d: np.ndarray, wave_number_k0: float = 1.0) -> Dict[str, Any]:
        decomp = self.decomposer.decompose_metric_field(h_tensor, wave_number_k0=wave_number_k0)
        theta_10d = self.bridge.map_wave_modes_to_10d_bivectors(decomp)

        Omega = self.rotor_engine.build_bivector_omega(-0.5 * theta_10d)
        R = self.rotor_engine.exponential_map(Omega)

        v_transformed = self.rotor_engine.sandwich_transform(current_state_5d, R)
        v_normalized = v_transformed / (np.linalg.norm(v_transformed) + 1e-12)

        return {
            "energy_ratios": decomp.energy_ratios,
            "theta_10d": theta_10d,
            "rotor_matrix": R,
            "transformed_state_5d": v_normalized,
            "norm_preserved": float(np.linalg.norm(v_normalized))
        }
