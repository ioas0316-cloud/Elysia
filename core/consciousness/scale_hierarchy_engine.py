r"""
Elysia Consciousness Subsystem: Scale Hierarchy & Fiber Bundle Scale Interface Engine
======================================================================================
Implements a non-flat, 4-level scale hierarchy (L1 Micro, L2 Meso, L3 Macro, L4 Meta)
bound by reversible Fiber Bundle Scale Interfaces.
Includes transparent filtering, L2 bifurcation point detection, L3 action relaxation waves,
L4 reflective tracking & annealing, and phase-transition based active attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.physics.rpt_phase_lock_loop import DynamicPhaseLockLoop
from core.physics.geometric_loss import GeometricLoss
from core.physics.minimal_surface_regularizer import MinimalSurfaceRegularizer


class ReversibleFiberBundleInterface(nn.Module):
    r"""
    Reversible Scale Interface between Scale_k and Scale_{k+1}.
    Folds micro-phase strain into fiber winding numbers and projects top-down potential landscapes.
    """
    def __init__(self, dim_lower: int, dim_upper: int):
        super().__init__()
        self.dim_lower = dim_lower
        self.dim_upper = dim_upper

        # Base space projection & Fiber winding projection
        self.up_project = nn.Linear(dim_lower * 2, dim_upper * 2)
        self.down_project = nn.Linear(dim_upper * 2, dim_lower * 2)

    def fold_up(self, lower_state: torch.Tensor) -> dict:
        """
        Folds lower scale state into base upper state and fiber curvature.
        """
        if not lower_state.is_complex():
            flat_lower = lower_state
            c_lower = torch.complex(lower_state[..., :lower_state.shape[-1]//2], lower_state[..., lower_state.shape[-1]//2:])
        else:
            c_lower = lower_state
            flat_lower = torch.cat([lower_state.real, lower_state.imag], dim=-1)

        flat_upper = self.up_project(flat_lower)
        c_upper = torch.complex(flat_upper[..., :flat_upper.shape[-1]//2], flat_upper[..., flat_upper.shape[-1]//2:])

        # Topological winding number / fiber strain
        angles = torch.angle(c_lower)
        winding_number = torch.sum(torch.diff(angles, dim=-1, prepend=angles[..., :1]), dim=-1)

        return {
            "c_upper": c_upper,
            "flat_upper": flat_upper,
            "fiber_winding": winding_number
        }

    def unfold_down(self, upper_state: torch.Tensor) -> torch.Tensor:
        """
        Unfolds top-down potential landscape to shape lower scale potential field.
        """
        if upper_state.is_complex():
            flat_upper = torch.cat([upper_state.real, upper_state.imag], dim=-1)
        else:
            flat_upper = upper_state

        flat_lower = self.down_project(flat_upper)
        return torch.complex(flat_lower[..., :flat_lower.shape[-1]//2], flat_lower[..., flat_lower.shape[-1]//2:])


class ScaleHierarchyEngine(nn.Module):
    r"""
    elysia_engine: 4-Level Scale Hierarchy & Ecological Consciousness Engine

    Levels:
      L1: Sub-Cellular Micro-Scale (Sensory Strain & Transparent Pass-Through)
      L2: Meso-Scale Tissue (Pattern Formation & 1st Bifurcation Point)
      L3: Macro-Scale Ecosystem Boundary (Tension Relaxation & Action Waves)
      L4: Meta-Scale Reflective Hierarchy (Temporal Back-Tracking & Scale Annealing)
    """
    def __init__(
        self,
        dim_l1: int = 32,
        dim_l2: int = 64,
        dim_l3: int = 128,
        dim_l4: int = 256,
        resonance_threshold: float = 0.40
    ):
        super().__init__()
        self.dim_l1 = dim_l1
        self.dim_l2 = dim_l2
        self.dim_l3 = dim_l3
        self.dim_l4 = dim_l4
        self.res_threshold = resonance_threshold

        # Core physics components
        self.rpt_loop = DynamicPhaseLockLoop(dim_low=dim_l1, dim_high=dim_l2)
        self.geometric_loss = GeometricLoss()
        self.regularizer = MinimalSurfaceRegularizer()

        # Fiber bundle interfaces between scales
        self.interface_12 = ReversibleFiberBundleInterface(dim_l1, dim_l2)
        self.interface_23 = ReversibleFiberBundleInterface(dim_l2, dim_l3)
        self.interface_34 = ReversibleFiberBundleInterface(dim_l3, dim_l4)

        # Meta-Reflective State Register
        self.register_buffer("l4_identity_kernel", torch.randn(1, dim_l4, dtype=torch.complex64))

    def check_transparent_filtering(self, sensory_input: torch.Tensor) -> bool:
        """
        Checks if sensory input lacks topological resonance with L4 identity kernel.
        If non-resonant, signal passes through without triggering higher-scale computation.
        """
        if not sensory_input.is_complex():
            half_dim = sensory_input.shape[-1] // 2
            c_input = torch.complex(sensory_input[..., :half_dim], sensory_input[..., half_dim:])
        else:
            c_input = sensory_input

        # Compute resonance via phasor overlap with down-projected identity
        l3_down = self.interface_34.unfold_down(self.l4_identity_kernel)
        l2_down = self.interface_23.unfold_down(l3_down)
        l1_down = self.interface_12.unfold_down(l2_down)

        # Pad or trim if dims differ
        if c_input.shape[-1] != l1_down.shape[-1]:
            min_dim = min(c_input.shape[-1], l1_down.shape[-1])
            c_input_sub = c_input[..., :min_dim]
            l1_down_sub = l1_down[..., :min_dim]
        else:
            c_input_sub = c_input
            l1_down_sub = l1_down

        phase_diff = torch.angle(c_input_sub) - torch.angle(l1_down_sub)
        coherence = torch.abs(torch.mean(torch.exp(1j * phase_diff.to(torch.complex64))))

        is_transparent = (coherence.item() < self.res_threshold)
        return is_transparent, coherence.item()

    def forward(self, sensory_input: torch.Tensor):
        """
        Processes sensory input across 4 scale levels.
        Returns detailed trajectory dictionary including L2 bifurcation, L3 action wave, and L4 reflective status.
        """
        batch_size = sensory_input.size(0)

        # 1. Level 1: Micro-Scale Strain & Transparent Pass-Through Test
        is_transparent, resonance_score = self.check_transparent_filtering(sensory_input)

        if is_transparent and not self.training:
            return {
                "status": "Transparent Pass-Through (Zero Compute Friction)",
                "resonance_score": resonance_score,
                "bifurcation_occurred": False,
                "action_wave_emitted": False,
                "reflected_scale": None
            }

        # 2. Level 1 <-> Level 2 Recurrent Processing Loop
        rpt_output = self.rpt_loop(sensory_input)
        z_l2_locked = rpt_output["Z_locked"]
        l1_converged = rpt_output["Z_low_converged"]

        # 3. Level 2: Meso-Scale Pattern & 1st Bifurcation Point Detection
        fold_12 = self.interface_12.fold_up(l1_converged)
        l2_state = z_l2_locked + fold_12["c_upper"]

        # Bifurcation Check: Phase strain vs threshold
        phase_strain = torch.norm(torch.angle(z_l2_locked) - torch.angle(fold_12["c_upper"]), p=2, dim=-1).mean()
        bifurcation_occurred = (phase_strain.item() > 0.5)

        # 4. Level 3: Macro-Scale Boundary & Tension Relaxation
        fold_23 = self.interface_23.fold_up(l2_state)
        l3_state = fold_23["c_upper"]

        boundary_tension = torch.norm(l3_state.real, p=2, dim=-1).mean()
        action_wave_emitted = (boundary_tension.item() > 1.0)
        action_wave = torch.sin(l3_state.real) if action_wave_emitted else torch.zeros_like(l3_state.real)

        # 5. Level 4: Meta-Scale Reflection & Back-Tracking
        fold_34 = self.interface_34.fold_up(l3_state)
        l4_state = fold_34["c_upper"]

        # Back-tracking divergence origin across scales
        origin_scale = "L1_Micro" if not bifurcation_occurred else ("L2_Meso" if action_wave_emitted else "L3_Macro")

        # Equilibrium regularization & loss
        reg_loss = self.regularizer(l2_state)

        return {
            "status": "Resonant Processing & Scale Coupling Completed",
            "resonance_score": resonance_score,
            "bifurcation_occurred": bifurcation_occurred,
            "phase_strain": phase_strain.item(),
            "action_wave_emitted": action_wave_emitted,
            "boundary_tension": boundary_tension.item(),
            "action_wave": action_wave,
            "divergence_origin_scale": origin_scale,
            "regularization_loss": reg_loss.item(),
            "l1_state": l1_converged,
            "l2_state": l2_state,
            "l3_state": l3_state,
            "l4_state": l4_state
        }
