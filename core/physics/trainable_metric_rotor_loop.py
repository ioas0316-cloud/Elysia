"""
core/physics/trainable_metric_rotor_loop.py

Complete Closed-Loop Architecture:
Tensor Helmholtz Decomposition + 5D Clifford Rotor + Trainable Scale Wave Memory
"""

import torch
import torch.nn as nn
import numpy as np

from core.physics.quantum_rotor_phase import Clifford5DRotorEngine
from core.memory.scale_wave_autograd import ScaleWaveMemoryAutogradFunction


class TrainableMetricRotorPipeline(nn.Module):
    """계량 변형 파동 -> 5D 로터 회전 -> 역전파 파동 메모리 학습 통합 모듈"""

    def __init__(self, num_scales: int = 3, base_dim: int = 8, device: str = "cpu"):
        super().__init__()
        self.num_scales = num_scales
        self.base_dim = base_dim
        self.device = torch.device(device)

        self.rotor_engine = Clifford5DRotorEngine(dim=5)

        scale_factors = [1.0 / (2.0**l) for l in range(num_scales)]
        self.register_buffer("scale_factors", torch.tensor(scale_factors, dtype=torch.float32, device=self.device))
        self.register_buffer("tensor_memory", torch.zeros((num_scales, base_dim, 3), dtype=torch.float32, device=self.device))

        self.phase_shift = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device))
        self.coupling_gain = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device))

    def forward(self, input_theta_10d: np.ndarray, initial_v5d: np.ndarray) -> torch.Tensor:
        # 1. 5D 클리퍼드 로터 연산 집행
        Omega = self.rotor_engine.build_bivector_omega(-0.5 * input_theta_10d)
        R = self.rotor_engine.exponential_map(Omega)
        v_rotated = self.rotor_engine.sandwich_transform(initial_v5d, R)
        v_rotated = v_rotated / (np.linalg.norm(v_rotated) + 1e-12)

        # 2. 로터 회전 결과 v_5d의 4차원, 5차원 위상을 메모리 충격 인자로 변환
        impulse_phase = torch.tensor(float(v_rotated[3] + v_rotated[4]), dtype=torch.float32, device=self.device)

        # 3. 스케일 파동 텐서 메모리로 역전파 가능 충격 전달
        updated_memory = ScaleWaveMemoryAutogradFunction.apply(
            self.tensor_memory,
            self.scale_factors,
            1,  # Target Scale Level 1 (Meso)
            0,  # Target Index 0
            self.phase_shift + impulse_phase,
            self.coupling_gain
        )
        return updated_memory
