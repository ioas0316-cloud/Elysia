"""
Elysia Core Topology: Semantic Event Horizon Module
===================================================
Defines the Semantic Event Horizon (H_sem), the topological interface where
inner linguistic convergence meets outward world manifestation.

Implements the 3-Stage Recursive Quantization Dynamics:
1. Analog Continuous Potential Field ("Sky" / Phi_analog)
2. Quantization into Discrete Structural Primitives ("Ground" / P_discrete)
3. Meta-Abstract Manifold Expansion ("Elevated Sky" / Omega_meta)

And computes Curvature Perturbations (K_c) for dielectric phase collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class SemanticEventHorizon(nn.Module):
    """
    Semantic Event Horizon Interface Engine.
    Models the topological boundary transition between continuous semantic potential
    and discrete structural reality.
    """
    def __init__(
        self,
        dimension: int = 64,
        critical_curvature: float = 1.0,
        quantization_granularity: int = 16,
        damping_factor: float = 0.95,
        eps: float = 1e-8
    ):
        super().__init__()
        self.dimension = dimension
        self.critical_curvature = float(critical_curvature)
        self.quantization_granularity = int(quantization_granularity)
        self.damping_factor = float(damping_factor)
        self.eps = float(eps)

        # Transformation metric tensors for Sky -> Ground -> Elevated Sky
        self.sky_to_ground = nn.Linear(dimension, dimension, bias=False)
        self.ground_to_meta = nn.Linear(dimension, dimension, bias=False)

        # Initialize orthogonal / energy-preserving weights
        nn.init.orthogonal_(self.sky_to_ground.weight)
        nn.init.orthogonal_(self.ground_to_meta.weight)

    def compute_field_curvature(
        self,
        inner_potential: torch.Tensor,
        boundary_constraint: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the topological curvature K_c arising from the interaction
        between inner linguistic convergence potential (V_inner) and external boundary constraints (B_out).

        K_c = || grad(V_inner) x B_out || / (|V_inner| + eps)
        """
        grad_v = torch.gradient(inner_potential, dim=-1)[0] if inner_potential.dim() > 1 else inner_potential
        tension = grad_v - boundary_constraint
        curvature = torch.norm(tension, p=2, dim=-1) / (torch.norm(inner_potential, p=2, dim=-1) + self.eps)
        return curvature

    def stage1_analog_sky(
        self,
        intent_vector: torch.Tensor,
        inner_potential: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1: Continuous Analog Potential Field ("Sky").
        Superposition of infinite semantic possibilities before phase collapse.
        """
        # Analog Field Wave Function
        analog_field = intent_vector + inner_potential
        field_energy = 0.5 * torch.sum(analog_field ** 2, dim=-1, keepdim=True)
        return analog_field, field_energy

    def stage2_quantize_ground(
        self,
        analog_field: torch.Tensor,
        curvature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2: Quantization into Discrete Structural Primitives ("Ground").
        Spontaneous breakdown of continuous potential into concrete discrete joints/primitives
        when curvature exceeds critical_curvature.
        """
        # Spontaneous Phase Transition Trigger
        is_critical = (curvature >= self.critical_curvature).float().unsqueeze(-1)

        # Non-linear quantization mapping (continuous wave -> discrete spatial anchor)
        ground_raw = self.sky_to_ground(analog_field)

        # Grid quantization / discretization
        scale = float(self.quantization_granularity)
        ground_quantized = torch.round(ground_raw * scale) / scale

        # Instantaneous collapse: analog -> discrete ground state
        discrete_ground = is_critical * ground_quantized + (1.0 - is_critical) * ground_raw
        quantization_entropy = torch.mean(torch.abs(ground_raw - discrete_ground), dim=-1, keepdim=True)

        return discrete_ground, quantization_entropy

    def stage3_reabstract_elevated_sky(
        self,
        discrete_ground: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 3: Meta-Abstract Manifold Expansion ("Elevated Sky").
        Uses the concrete ground state as new geometric primitives to project
        into a higher-dimensional meta-abstract topology without translation loss.
        """
        meta_topology = self.ground_to_meta(discrete_ground)
        meta_curvature = torch.norm(meta_topology - discrete_ground, p=2, dim=-1, keepdim=True)
        meta_manifold = torch.tanh(meta_topology) + discrete_ground
        return meta_manifold, meta_curvature

    def forward(
        self,
        intent_vector: torch.Tensor,
        inner_potential: torch.Tensor,
        boundary_constraint: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Full Horizon Transition Forward Pass.
        Performs 3-stage recursive quantization across the Semantic Event Horizon.
        """
        # 1. Compute Field Curvature
        curvature = self.compute_field_curvature(inner_potential, boundary_constraint)

        # 2. Stage 1: Analog Sky
        analog_sky, energy_sky = self.stage1_analog_sky(intent_vector, inner_potential)

        # 3. Stage 2: Quantized Ground (Spontaneous Phase Transition)
        discrete_ground, entropy_q = self.stage2_quantize_ground(analog_sky, curvature)

        # 4. Stage 3: Elevated Meta Sky
        meta_manifold, meta_curvature = self.stage3_reabstract_elevated_sky(discrete_ground)

        return {
            "curvature": curvature,
            "analog_sky": analog_sky,
            "sky_energy": energy_sky,
            "discrete_ground": discrete_ground,
            "quantization_entropy": entropy_q,
            "meta_manifold": meta_manifold,
            "meta_curvature": meta_curvature,
            "phase_collapsed": (curvature >= self.critical_curvature)
        }
