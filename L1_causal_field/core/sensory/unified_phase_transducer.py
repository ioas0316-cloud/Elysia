"""
Unified Phase Transducer Engine (통합 위상 트랜스듀서 엔진)

Converts all modalities (vision, audio, tactile, scientific reasoning, imagination)
into a unified scale wave spectrum and drives 5D Clifford Rotors R in Spin(5) via 10D bivectors.

Modalities:
1. Vision (2D+1T wave phase): theta_vis = arctan2(nabla_y I, nabla_x I)
2. Audio/Language (1D acoustic harmonics): theta_aud = omega_a * t + phi_a
3. Tactile/Spatial (1tan boundary tension): 1tan(theta_tac) = tau_shear / sigma_normal
4. Scientific Reasoning (Energy potential surface): theta_sci = arctan(nabla V_pot)
5. Imagination (Virtual phase projection): theta_imag = theta_base + delta_phi_virt

Output:
- Unified sensory phase vector theta_sensory
- 10D bivector generators for so(5) Lie algebra
- 5D Clifford rotor rotation matrix R in Spin(5)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
from scipy.linalg import expm
import numpy as np


@dataclass
class TransducerOutput:
    theta_sensory: torch.Tensor
    bivector_10d: torch.Tensor
    r_rotor_5d: torch.Tensor
    state_v5d_transformed: torch.Tensor
    closed_loop_resonance: float


class UnifiedPhaseTransducerEngine(nn.Module):
    """
    Directly bridges multi-sensory modalities to 5D Clifford Rotors R in Spin(5).
    """

    def __init__(
        self,
        dimension: int = 64,
        dtype=torch.float32
    ):
        super().__init__()
        self.dimension = dimension
        self.dtype = dtype

        # Projection matrix from dimension -> 10D bivectors
        self.bivector_projection = nn.Linear(dimension, 10, bias=False, dtype=dtype)
        # Initial 5D state vector v_5d
        init_v5 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)
        self.register_buffer("v5d_base", init_v5)

    def transduce_vision(self, vision_field_2d: torch.Tensor) -> torch.Tensor:
        """
        Transduces 2D visual intensity field into phase angles theta_vis.
        """
        # Compute spatial gradients
        grad_y, grad_x = torch.gradient(vision_field_2d, dim=(-2, -1))
        theta_vis = torch.atan2(grad_y, grad_x + 1e-8)
        # Flatten/pool to dimension
        if theta_vis.numel() != self.dimension:
            theta_vis = nn.functional.adaptive_avg_pool1d(theta_vis.view(1, 1, -1), self.dimension).squeeze()
        return theta_vis

    def transduce_audio_language(self, audio_signal_1d: torch.Tensor) -> torch.Tensor:
        """
        Transduces 1D audio acoustic harmonics or text sequence into phase thread.
        """
        phase_aud = torch.atan2(audio_signal_1d, torch.roll(audio_signal_1d, shifts=1) + 1e-8)
        if phase_aud.numel() != self.dimension:
            phase_aud = nn.functional.adaptive_avg_pool1d(phase_aud.view(1, 1, -1), self.dimension).squeeze()
        return phase_aud

    def transduce_tactile_spatial(self, shear_stress: torch.Tensor, normal_stress: torch.Tensor) -> torch.Tensor:
        """
        Transduces tactile pressure and shear stress into 1tan boundary phase angles.
        """
        normal_safe = torch.where(torch.abs(normal_stress) < 1e-5, 1e-5, normal_stress)
        tan_tac = shear_stress / normal_safe
        theta_tac = torch.atan(tan_tac)
        if theta_tac.numel() != self.dimension:
            theta_tac = nn.functional.adaptive_avg_pool1d(theta_tac.view(1, 1, -1), self.dimension).squeeze()
        return theta_tac

    def transduce_scientific_reasoning(self, potential_surface: torch.Tensor) -> torch.Tensor:
        """
        Transduces micro scale energy potential surface into gradient wave theta_sci.
        """
        grad_v = torch.gradient(potential_surface)[0]
        theta_sci = torch.atan(grad_v)
        if theta_sci.numel() != self.dimension:
            theta_sci = nn.functional.adaptive_avg_pool1d(theta_sci.view(1, 1, -1), self.dimension).squeeze()
        return theta_sci

    def transduce_imagination(self, virtual_phase_projection: torch.Tensor) -> torch.Tensor:
        """
        Transduces internal virtual phase projection into theta_imag.
        """
        if virtual_phase_projection.numel() != self.dimension:
            virtual_phase_projection = nn.functional.adaptive_avg_pool1d(
                virtual_phase_projection.view(1, 1, -1), self.dimension
            ).squeeze()
        return virtual_phase_projection

    def construct_so5_generator(self, bivector_10d: torch.Tensor) -> torch.Tensor:
        """
        Constructs 5x5 anti-symmetric Lie algebra matrix Omega in so(5)
        from 10D bivector components.
        """
        omega = torch.zeros((5, 5), dtype=self.dtype, device=bivector_10d.device)
        idx = 0
        for i in range(5):
            for j in range(i + 1, 5):
                omega[i, j] = bivector_10d[idx]
                omega[j, i] = -bivector_10d[idx]
                idx += 1
        return omega

    def compute_clifford_rotor_expm(self, omega_so5: torch.Tensor) -> torch.Tensor:
        """
        Computes 5D Clifford Rotor rotation matrix R = exp(Omega) in Spin(5).
        """
        omega_np = omega_so5.detach().cpu().numpy()
        r_np = expm(omega_np)
        return torch.tensor(r_np, dtype=self.dtype, device=omega_so5.device)

    def forward(
        self,
        vision_input: Optional[torch.Tensor] = None,
        audio_input: Optional[torch.Tensor] = None,
        shear_input: Optional[torch.Tensor] = None,
        normal_input: Optional[torch.Tensor] = None,
        potential_input: Optional[torch.Tensor] = None,
        imagination_input: Optional[torch.Tensor] = None
    ) -> TransducerOutput:
        """
        Combines multi-modal sensory signals into unified scale wave theta_sensory,
        maps to 10D bivectors, and drives 5D Clifford Rotor transformation.
        """
        device = self.bivector_projection.weight.device
        theta_components = []

        if vision_input is not None:
            theta_components.append(self.transduce_vision(vision_input))
        if audio_input is not None:
            theta_components.append(self.transduce_audio_language(audio_input))
        if shear_input is not None and normal_input is not None:
            theta_components.append(self.transduce_tactile_spatial(shear_input, normal_input))
        if potential_input is not None:
            theta_components.append(self.transduce_scientific_reasoning(potential_input))
        if imagination_input is not None:
            theta_components.append(self.transduce_imagination(imagination_input))

        if len(theta_components) == 0:
            # Default zero wave
            theta_sensory = torch.zeros(self.dimension, dtype=self.dtype, device=device)
        else:
            # Unified superposition of phase components
            theta_sensory = torch.stack(theta_components, dim=0).mean(dim=0)

        # Map to 10D bivectors
        bivector_10d = self.bivector_projection(theta_sensory)

        # Build so(5) generator and compute Spin(5) rotor R
        omega_so5 = self.construct_so5_generator(bivector_10d)
        r_rotor = self.compute_clifford_rotor_expm(omega_so5)

        # Transform 5D base state vector
        v5d_transformed = torch.matmul(r_rotor, self.v5d_base)

        # Closed loop resonance score
        resonance_score = float(torch.abs(v5d_transformed[0]).item())

        return TransducerOutput(
            theta_sensory=theta_sensory,
            bivector_10d=bivector_10d,
            r_rotor_5d=r_rotor,
            state_v5d_transformed=v5d_transformed,
            closed_loop_resonance=resonance_score
        )
