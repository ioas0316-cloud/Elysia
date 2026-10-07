"""
Autonomous Phase Rearrangement Engine (자율 위상 재정렬 파동 엔진)

Implements 4-Step Autonomous Wave Mechanism upon Exogenous Causal Shock:
Step 1: Phase Dislocation (위상 결함) & Phase Turbulence (위상 난류 생성)
Step 2: Potential Stress (포텐셜 지형 기울어짐) & 1tan Tangential Tension Accumulation
Step 3: Phase-Slip (위상 도약) & 5D Clifford Rotor Spin(5) Nonlinear Jump
Step 4: Cross-Scale Dissipation (스케일 간 파동 분산) & New Attractor Self-Locking (새로운 끌개 자율 고정)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import math
import torch
import torch.nn as nn
from scipy.linalg import expm
import numpy as np


@dataclass
class RearrangementStepState:
    step_index: int
    step_name: str
    phase_field: torch.Tensor
    turbulence_level: float
    boundary_stress: float
    rotor_state_5d: torch.Tensor
    is_locked: bool


class AutonomousPhaseRearrangementEngine(nn.Module):
    """
    Handles autonomous wave-mechanic self-reorganization across 4 steps when facing shocks.
    """

    def __init__(
        self,
        dimension: int = 64,
        stress_threshold: float = 2.0,
        dtype=torch.float32
    ):
        super().__init__()
        self.dimension = dimension
        self.stress_threshold = stress_threshold
        self.dtype = dtype

        # 5D Clifford Rotor State Vector v_5d in Spin(5)
        init_v5 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)
        self.register_buffer("v5d_rotor_state", init_v5)

        # Baseline internal phase field
        self.phase_field = nn.Parameter(torch.zeros(dimension, dtype=dtype))

    def step1_phase_dislocation(self, shock_pulse: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Step 1: Phase Dislocation & Turbulence Generation.
        Receives shock pulse as destructive interference wave and forms dislocation nodes.
        """
        # Destructive interference creates local phase dislocations
        phase_shock = torch.atan2(shock_pulse, torch.roll(shock_pulse, shifts=1) + 1e-8)
        dislocated_phase = self.phase_field + phase_shock

        # Calculate phase turbulence: div(J_phase)
        sin_p = torch.sin(dislocated_phase)
        cos_p = torch.cos(dislocated_phase)
        grad_cos = torch.gradient(cos_p)[0]
        grad_sin = torch.gradient(sin_p)[0]
        j_phase = sin_p * grad_cos - cos_p * grad_sin
        turbulence = torch.gradient(j_phase)[0]
        max_turb = float(torch.max(torch.abs(turbulence)).item())

        return dislocated_phase, max_turb

    def step2_potential_stress_accumulation(self, dislocated_phase: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Step 2: Potential Stress & 1tan Tangential Boundary Tension Accumulation.
        Distorts potential energy landscape and accumulates shear tension.
        """
        sin_p = torch.sin(dislocated_phase)
        cos_p = torch.cos(dislocated_phase)
        safe_cos = torch.where(torch.abs(cos_p) < 1e-4, 1e-4 * torch.sign(cos_p + 1e-8), cos_p)
        tan_p = torch.clamp(sin_p / safe_cos, -50.0, 50.0)

        boundary_stress = float(torch.mean(torch.abs(tan_p)).item())
        return tan_p, boundary_stress

    def step3_phase_slip_rotor_jump(
        self,
        tan_p: torch.Tensor,
        boundary_stress: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 3: Phase-Slip & 5D Clifford Rotor Nonlinear Jump.
        Triggers nonlinear phase-slip and 5D Clifford rotor rotation when stress > threshold.
        """
        if boundary_stress > self.stress_threshold:
            # Bivector rotation generator omega in so(5) driven by tan stress
            bivector_10d = torch.tanh(tan_p[:10] if tan_p.numel() >= 10 else torch.cat([tan_p, torch.zeros(10 - tan_p.numel(), dtype=self.dtype, device=tan_p.device)]))
            omega_so5 = torch.zeros((5, 5), dtype=self.dtype, device=tan_p.device)
            idx = 0
            for i in range(5):
                for j in range(i + 1, 5):
                    omega_so5[i, j] = bivector_10d[idx]
                    omega_so5[j, i] = -bivector_10d[idx]
                    idx += 1

            r_matrix = torch.tensor(expm(omega_so5.detach().cpu().numpy()), dtype=self.dtype, device=tan_p.device)
            new_v5d = torch.matmul(r_matrix, self.v5d_rotor_state)

            # Phase slip angle adjustment
            phase_slipped = tan_p * 0.2 + math.pi / 4.0
        else:
            new_v5d = self.v5d_rotor_state
            phase_slipped = tan_p * 0.5

        return phase_slipped, new_v5d

    def step4_cross_scale_dissipation_locking(
        self,
        phase_slipped: torch.Tensor,
        v5d_state: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Step 4: Cross-Scale Dissipation & New Constructive Attractor Self-Locking.
        Dissipates shock wave across scale layers and locks into a new constructive attractor.
        """
        # Smooth and stabilize phase field (constructive interference)
        cos_p = torch.cos(phase_slipped)
        locked_phase = torch.atan2(torch.sin(phase_slipped), cos_p + 1e-5)

        # Check phase locking condition
        phase_var = float(torch.var(locked_phase).item())
        is_locked = phase_var < 1.0

        return locked_phase, is_locked

    def execute_4step_rearrangement(
        self,
        shock_pulse: torch.Tensor
    ) -> List[RearrangementStepState]:
        """
        Executes full 4-step autonomous rearrangement pipeline.
        """
        history = []

        # Step 1
        p1, turb1 = self.step1_phase_dislocation(shock_pulse)
        history.append(RearrangementStepState(
            step_index=1, step_name="Phase Dislocation & Turbulence",
            phase_field=p1, turbulence_level=turb1, boundary_stress=0.0,
            rotor_state_5d=self.v5d_rotor_state, is_locked=False
        ))

        # Step 2
        tan2, stress2 = self.step2_potential_stress_accumulation(p1)
        history.append(RearrangementStepState(
            step_index=2, step_name="Potential Stress Accumulation",
            phase_field=tan2, turbulence_level=turb1, boundary_stress=stress2,
            rotor_state_5d=self.v5d_rotor_state, is_locked=False
        ))

        # Step 3
        p3, v5_3 = self.step3_phase_slip_rotor_jump(tan2, stress2)
        history.append(RearrangementStepState(
            step_index=3, step_name="Phase-Slip & 5D Clifford Rotor Jump",
            phase_field=p3, turbulence_level=turb1 * 0.5, boundary_stress=stress2 * 0.5,
            rotor_state_5d=v5_3, is_locked=False
        ))

        # Step 4
        p4, locked4 = self.step4_cross_scale_dissipation_locking(p3, v5_3)
        history.append(RearrangementStepState(
            step_index=4, step_name="Cross-Scale Dissipation & Attractor Self-Locking",
            phase_field=p4, turbulence_level=turb1 * 0.1, boundary_stress=stress2 * 0.1,
            rotor_state_5d=v5_3, is_locked=locked4
        ))

        # Update internal state
        with torch.no_grad():
            self.phase_field.copy_(p4)
            self.v5d_rotor_state.copy_(v5_3)

        return history

    def forward(self, shock_pulse: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass executing 4-step rearrangement.
        """
        history = self.execute_4step_rearrangement(shock_pulse)
        final_step = history[-1]

        return {
            "rearrangement_history": history,
            "final_phase_field": final_step.phase_field,
            "final_rotor_state_5d": final_step.rotor_state_5d,
            "is_self_locked": final_step.is_locked,
            "initial_turbulence": history[0].turbulence_level,
            "peak_boundary_stress": history[1].boundary_stress
        }
