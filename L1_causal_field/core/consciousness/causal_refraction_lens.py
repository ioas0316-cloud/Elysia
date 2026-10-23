"""
Causal Refraction Lens Module (인과적 굴절 렌즈 모듈)
===================================================
외부의 무작위 섭동(q_ext, T_munu)을 단순 외력 충돌이 아닌,
원인->결과 상호작용 궤적으로 집속(Refraction)하고, 기존 기억 지형과의
위상적 공명 및 영구적 측경학적 곡률 각인(Crystalline Memory Engram)으로
연동하는 인과적 렌즈 필터링 모델입니다.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class CausalRefractionLens(nn.Module):
    """
    외부 섭동을 인과적 관계성으로 굴절(Refraction)하여 기억 지형에 각인하는 렌즈
    """

    def __init__(self, focal_length: float = 1.0, resonance_threshold: float = 0.3):
        super().__init__()
        self.focal_length = focal_length
        self.resonance_threshold = resonance_threshold

    def forward(
        self,
        j_flux: torch.Tensor,
        g_meta: torch.Tensor,
        q_sys: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            j_flux: [D, H, W, 4] 외부 결합 유입 플럭스 사원수
            g_meta: [D, H, W] 현재 메타-지형 곡률 계량
            q_sys: [D, H, W, 4] 내부 파동장
        Returns:
            dict containing:
            - refracted_trajectory: 렌즈에 의해 원인->결과 초점으로 집속된 인과 궤적
            - phase_locking_resonance: 기존 위상 구조와의 공명 지수
            - memory_geodesic_engram: 메타-지형에 가소적으로 영구 각인되는 기하학적 골짜기 변형량
        """
        # 1. 인과적 굴절 (Causal Refraction)
        # 플럭스의 허수 성분(회전 토크)을 렌즈 초점 거리(focal_length)로 집속
        flux_vector = j_flux[..., 1:]
        refracted_trajectory = flux_vector / (self.focal_length + torch.norm(flux_vector, dim=-1, keepdim=True) + 1e-8)

        # 2. 위상적 연동 및 공명 (Relational Phase-Locking)
        # 굴절된 궤적과 기존 시스템 파동장 q_sys 간의 tensor dot product (공명)
        sys_vector = q_sys[..., 1:]
        dot_product = torch.sum(refracted_trajectory * sys_vector, dim=-1)
        phase_locking_resonance = torch.sigmoid((dot_product - self.resonance_threshold) * 5.0)

        # 3. 메타-지형의 가소적 변형으로서의 '기억화' (Geodesic Memory Basin)
        # 공명된 인과 궤적이 지난 자리에 기하학적 골짜기(곡률 변형)를 영구 각인
        memory_geodesic_engram = phase_locking_resonance * torch.norm(refracted_trajectory, dim=-1)

        return {
            "refracted_trajectory": refracted_trajectory,
            "phase_locking_resonance": phase_locking_resonance,
            "memory_geodesic_engram": memory_geodesic_engram,
        }
