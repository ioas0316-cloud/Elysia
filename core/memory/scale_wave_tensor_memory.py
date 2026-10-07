"""
core/memory/scale_wave_tensor_memory.py

Scale-Invariant Wave Tensor Memory Architecture Implementation
"""

import numpy as np


class ScaleWaveTensorMemory:
    """단편 스칼라 배열을 스케일 불변 파동 텐서 세트로 사상하고 제어하는 메모리 엔진"""

    def __init__(self, num_scales: int = 3, base_dim: int = 4):
        self.num_scales = num_scales
        self.base_dim = base_dim

        # 각 스케일별 스케일 계수 lambda_l (1.0, 0.5, 0.25...)
        self.scale_factors = np.array([1.0 / (2.0**l) for l in range(num_scales)])

        # 스케일별 삼위일체 파동 메모리 텐서 [Num_Scales, Base_Dim, 3] (3: sin, cos, tan)
        self.tensor_memory = np.zeros((num_scales, base_dim, 3), dtype=float)
        self._initialize_trinity_memory()

    def _initialize_trinity_memory(self):
        """기본 파동 메모리 셀 규격화 초기화"""
        for l in range(self.num_scales):
            lam = self.scale_factors[l]
            for i in range(self.base_dim):
                theta = lam * (i + 1) * np.pi / 4.0
                sin_v = np.sin(theta)
                cos_v = np.cos(theta)
                tan_v = np.tan(theta) if abs(cos_v) > 1e-3 else np.sign(sin_v) * 1e3
                self.tensor_memory[l, i] = [sin_v, cos_v, tan_v]

    def write_wave_data(self, scale_level: int, address_idx: int, phase_shift: float, coupling_gain: float = 1.0):
        """특정 스케일 셀에 위상 충격 인가 및 스케일 결합 전파 (Write Operation)"""
        if scale_level < 0 or scale_level >= self.num_scales:
            raise ValueError("Scale level out of range")

        # 1. 국소 셀 위상 전환
        current_sin = self.tensor_memory[scale_level, address_idx, 0]
        current_cos = self.tensor_memory[scale_level, address_idx, 1]
        current_theta = np.arctan2(current_sin, current_cos) + phase_shift

        self.tensor_memory[scale_level, address_idx, 0] = np.sin(current_theta)
        self.tensor_memory[scale_level, address_idx, 1] = np.cos(current_theta)
        self.tensor_memory[scale_level, address_idx, 2] = np.tan(current_theta)

        # 2. 상위 및 하위 스케일로 파동 전파 (Cross-Scale Coupling)
        for l in range(self.num_scales):
            if l == scale_level:
                continue
            dist = abs(l - scale_level)
            coupling_k = np.exp(-dist) * coupling_gain  # 스케일 거리 비례 감쇄
            for i in range(self.base_dim):
                coupled_theta = current_theta * coupling_k * (self.scale_factors[l] / self.scale_factors[scale_level])
                self.tensor_memory[l, i, 0] += 0.1 * np.sin(coupled_theta)
                self.tensor_memory[l, i, 1] += 0.1 * np.cos(coupled_theta)
                # 규격화
                norm = np.sqrt(self.tensor_memory[l, i, 0]**2 + self.tensor_memory[l, i, 1]**2) + 1e-8
                self.tensor_memory[l, i, 0] /= norm
                self.tensor_memory[l, i, 1] /= norm
                self.tensor_memory[l, i, 2] = self.tensor_memory[l, i, 0] / (self.tensor_memory[l, i, 1] + 1e-8)

    def resonance_read(self, query_wave: np.ndarray) -> np.ndarray:
        """주파수/위상 공명 쿼리를 통한 무손실 내적 읽기 (Read Operation)"""
        # query_wave: [3] (sin, cos, tan)
        resonance_map = np.zeros((self.num_scales, self.base_dim))
        for l in range(self.num_scales):
            for i in range(self.base_dim):
                cell = self.tensor_memory[l, i]
                # 삼위일체 내적 공명도 계산
                resonance_map[l, i] = np.dot(cell[:2], query_wave[:2])  # sin, cos 보강간섭
        return resonance_map
