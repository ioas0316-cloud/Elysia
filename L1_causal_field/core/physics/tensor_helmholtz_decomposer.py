"""
core/physics/tensor_helmholtz_decomposer.py

Tensor Helmholtz Operator & Metric Plasticity (g_ij) Wave Decomposition
=======================================================================
Decomposes metric perturbation h_ij(r) under Helmholtz Equation (\nabla^2 + k^2) h_ij = 0
into Scalar (Trace), Vector (Gradient/Solenoidal), and Tensor (Transverse-Traceless) modes.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class MetricWaveDecompositionResult:
    """계량 소성 h_ij 파동 분해 결과 데이터 클래스"""
    original_h: np.ndarray          # [3, 3, N_x, N_y, N_z]
    scalar_trace_mode: np.ndarray   # 스칼라 체적 파동 (Iso-volumetric Expansion)
    vector_shear_mode: np.ndarray   # 벡터 전단 파동 (Solenoidal Gradient)
    tensor_tt_mode: np.ndarray      # 횡파-무자취 텐서 파동 (Transverse-Traceless)
    energy_ratios: Dict[str, float]  # 각 모드별 파동 에너지 비중


class TensorHelmholtzMetricDecomposer:
    """3D/ND 공간 상의 계량 텐서 h_ij 파동 모드 푸리에-헬름홀츠 분해 연산자"""

    def __init__(self, grid_shape: Tuple[int, int, int] = (16, 16, 16), L: float = 2.0 * np.pi):
        self.grid_shape = grid_shape
        self.dim = 3
        self.L = L

        kx = 2.0 * np.pi * np.fft.fftfreq(grid_shape[0], d=L / grid_shape[0])
        ky = 2.0 * np.pi * np.fft.fftfreq(grid_shape[1], d=L / grid_shape[1])
        kz = 2.0 * np.pi * np.fft.fftfreq(grid_shape[2], d=L / grid_shape[2])

        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
        self.K_vec = np.stack([Kx, Ky, Kz], axis=0)  # [3, N_x, N_y, N_z]
        self.K2 = Kx**2 + Ky**2 + Kz**2
        self.K2_safe = np.where(self.K2 == 0, 1e-12, self.K2)

    def decompose_metric_field(self, h_tensor: np.ndarray, wave_number_k0: float = 1.0) -> MetricWaveDecompositionResult:
        """계량 변형 h_ij(r) [3, 3, Nx, Ny, Nz]를 3개 파동 성분으로 정밀 분해"""
        assert h_tensor.shape[:2] == (3, 3), "입력 텐서는 3x3 대칭 계량이어야 합니다."

        h_tilde = np.zeros((3, 3, *self.grid_shape), dtype=complex)
        for i in range(3):
            for j in range(3):
                h_tilde[i, j] = np.fft.fftn(h_tensor[i, j])

        unit_k = self.K_vec / np.sqrt(self.K2_safe)  # [3, Nx, Ny, Nz]
        P_ij = np.zeros((3, 3, *self.grid_shape), dtype=complex)
        for i in range(3):
            for j in range(3):
                delta = 1.0 if i == j else 0.0
                P_ij[i, j] = delta - unit_k[i] * unit_k[j]

        # A. 스칼라 모드 (Trace Mode: h_scalar * delta_ij)
        trace_h = (h_tilde[0, 0] + h_tilde[1, 1] + h_tilde[2, 2]) / 3.0
        h_tilde_scalar = np.zeros_like(h_tilde)
        for i in range(3):
            h_tilde_scalar[i, i] = trace_h

        # B. Transverse-Traceless (TT) 텐서 파동 모드
        P_dot_h = np.einsum('ik...,kl...->il...', P_ij, h_tilde)
        P_dot_h_dot_P = np.einsum('il...,lj...->ij...', P_dot_h, P_ij)
        P_trace = np.einsum('kl...,kl...->...', P_ij, h_tilde)

        h_tilde_tt = np.zeros_like(h_tilde)
        for i in range(3):
            for j in range(3):
                h_tilde_tt[i, j] = P_dot_h_dot_P[i, j] - 0.5 * P_ij[i, j] * P_trace

        # C. 벡터 모드 (Vector Shear Mode) = 전체 - 스칼라 - 텐서_TT
        h_tilde_vector = h_tilde - h_tilde_scalar - h_tilde_tt

        # 헬름홀츠 파동 온-쉘 껍질 억제 필터 (|\mathbf{k}|^2 = k_0^2 필터링)
        helmholtz_filter = np.exp(-0.5 * ((np.sqrt(self.K2) - wave_number_k0) ** 2) / 0.1)
        h_tilde_scalar *= helmholtz_filter
        h_tilde_vector *= helmholtz_filter
        h_tilde_tt *= helmholtz_filter

        h_scalar = np.real(np.stack([np.fft.ifftn(h_tilde_scalar[i, j]) for i in range(3) for j in range(3)]).reshape(3, 3, *self.grid_shape))
        h_vector = np.real(np.stack([np.fft.ifftn(h_tilde_vector[i, j]) for i in range(3) for j in range(3)]).reshape(3, 3, *self.grid_shape))
        h_tt = np.real(np.stack([np.fft.ifftn(h_tilde_tt[i, j]) for i in range(3) for j in range(3)]).reshape(3, 3, *self.grid_shape))

        e_scalar = float(np.sum(h_scalar**2))
        e_vector = float(np.sum(h_vector**2))
        e_tt = float(np.sum(h_tt**2))
        total_e = e_scalar + e_vector + e_tt + 1e-12

        return MetricWaveDecompositionResult(
            original_h=h_tensor,
            scalar_trace_mode=h_scalar,
            vector_shear_mode=h_vector,
            tensor_tt_mode=h_tt,
            energy_ratios={
                "scalar_trace": e_scalar / total_e,
                "vector_shear": e_vector / total_e,
                "tensor_tt": e_tt / total_e
            }
        )
