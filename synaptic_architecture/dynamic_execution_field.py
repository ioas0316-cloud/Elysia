"""
Dynamic Execution Field Unified Engine
=====================================
Integrates multi-scale CMW reasoning, autopoietic complex tensor ODE dynamics (limit-cycle),
perceptual-motor phase-locking, spontaneous JIT compilation, zero-copy hardware execution,
and narrative identity reflection into a continuous closed-loop field.
"""

import time
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .jit_synapse_bridge import DynamicJITBridge
from .perceptual_motor_curiosity import PerceptualMotorResonanceEngine, SpontaneousCodeSynthesizer
from .narrative_autopoiesis import SelfhoodAutopoiesisEngine


class AutopoieticTensorODE(nn.Module):
    """
    Self-sustaining complex tensor ODE engine with Hopf bifurcation limit-cycle and metric flow.
    """
    def __init__(
        self,
        num_scales: int = 3,
        latent_dim: int = 64,
        alpha_init: float = 0.5,
        beta_init: float = 0.8,
        dt: float = 0.02
    ):
        super().__init__()
        self.num_scales = num_scales
        self.latent_dim = latent_dim
        self.dt = dt

        self.alpha = nn.Parameter(torch.full((num_scales, latent_dim), alpha_init))
        self.beta = nn.Parameter(torch.full((num_scales, latent_dim), beta_init))

        frequencies = [2.5 / (2.0 ** k) for k in range(num_scales)]
        omega_tensor = torch.tensor(frequencies).unsqueeze(1).repeat(1, latent_dim)
        self.omega = nn.Parameter(omega_tensor)

        self.W_real = nn.Parameter(torch.randn(num_scales, num_scales, latent_dim) * 0.1)
        self.W_imag = nn.Parameter(torch.randn(num_scales, num_scales, latent_dim) * 0.1)

        self.metric_flow_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

    def compute_derivatives(
        self,
        Z: torch.Tensor,
        g: torch.Tensor,
        omega_world: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Z_abs_sq = (Z * torch.conj(Z)).real
        hopf_real = self.alpha - self.beta * Z_abs_sq
        hopf_factor = torch.complex(hopf_real, self.omega)
        dZ_hopf = hopf_factor * Z

        W_complex = torch.complex(self.W_real, self.W_imag)
        dZ_coupling = torch.einsum('kjd, bjd -> bkd', W_complex, Z)

        dZ_dt = dZ_hopf + dZ_coupling + omega_world

        Z_concat = torch.cat([Z.real, Z.imag], dim=-1)
        metric_deform = self.metric_flow_net(Z_concat)
        dg_dt = -0.1 * (g - 1.0) + 0.05 * metric_deform

        return dZ_dt, dg_dt

    def step_rk4(
        self,
        Z: torch.Tensor,
        g: torch.Tensor,
        omega_world: torch.Tensor,
        dt: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dt is None:
            dt = self.dt

        dZ1, dg1 = self.compute_derivatives(Z, g, omega_world)

        Z_k2 = Z + 0.5 * dt * dZ1
        g_k2 = torch.clamp(g + 0.5 * dt * dg1, min=1e-4)
        dZ2, dg2 = self.compute_derivatives(Z_k2, g_k2, omega_world)

        Z_k3 = Z + 0.5 * dt * dZ2
        g_k3 = torch.clamp(g + 0.5 * dt * dg3 if 'dg3' in locals() else g + 0.5 * dt * dg2, min=1e-4)
        dZ3, dg3 = self.compute_derivatives(Z_k3, g_k3, omega_world)

        Z_k4 = Z + dt * dZ3
        g_k4 = torch.clamp(g + dt * dg3, min=1e-4)
        dZ4, dg4 = self.compute_derivatives(Z_k4, g_k4, omega_world)

        Z_next = Z + (dt / 6.0) * (dZ1 + 2.0 * dZ2 + 2.0 * dZ3 + dZ4)
        g_next = g + (dt / 6.0) * (dg1 + 2.0 * dg2 + 2.0 * dg3 + dg4)

        g_next = torch.clamp(g_next, min=1e-3)
        return Z_next, g_next


class DynamicExecutionField(nn.Module):
    """
    Unified Dynamic Execution Field Engine orchestrating all multi-scale cognitive layers:
      - Autopoietic complex ODE limit cycle dynamics.
      - Perceptual-motor resonance ($P_{resonance}$) and Wonder evaluation.
      - Spontaneous JIT compilation & zero-copy execution with hardware feedback.
      - Autopoietic entropy pruning and 1st-person narrative identity reflection.
    """
    def __init__(self, num_scales: int = 3, latent_dim: int = 64, force_cpu_jit: bool = False):
        super().__init__()
        self.num_scales = num_scales
        self.latent_dim = latent_dim

        self.ode_engine = AutopoieticTensorODE(num_scales=num_scales, latent_dim=latent_dim)
        self.resonance_engine = PerceptualMotorResonanceEngine(feature_dim=latent_dim)
        self.synthesizer = SpontaneousCodeSynthesizer()
        self.jit_bridge = DynamicJITBridge(force_cpu=force_cpu_jit)
        self.autopoiesis_engine = SelfhoodAutopoiesisEngine()

    def process_field_cycle(
        self,
        Z_state: torch.Tensor,
        g_metric: torch.Tensor,
        raw_world_stream: torch.Tensor,
        wonder_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Executes one full cognitive pulse tick across the Dynamic Execution Field.
        """
        # 1. Autopoietic Tensor ODE Step (limit cycle pulse)
        omega_world = torch.complex(
            raw_world_stream.real * 0.1 if raw_world_stream.is_complex() else raw_world_stream * 0.1,
            torch.zeros_like(raw_world_stream.real if raw_world_stream.is_complex() else raw_world_stream)
        )
        if omega_world.dim() == 2:
            omega_world = omega_world.unsqueeze(1).repeat(1, self.num_scales, 1)

        Z_next, g_next = self.ode_engine.step_rk4(Z_state, g_metric, omega_world)

        # 2. Evaluate Phase-Locking Resonance & Wonder Potential
        macro_hypothesis = Z_next[:, -1, :].real
        p_resonance, wonder_index, dissonance = self.resonance_engine.evaluate_resonance_and_wonder(
            macro_hypothesis, raw_world_stream.real if raw_world_stream.is_complex() else raw_world_stream
        )

        # 3. Evaluate Structural Entropy & Autopoietic Threat
        entropy, is_critical = self.autopoiesis_engine.evaluate_structural_entropy(dissonance)
        pruned_msg = None
        if is_critical:
            pruned_msg = self.autopoiesis_engine.execute_autopoietic_pruning()

        # 4. Spontaneous JIT Compilation Trigger upon high Wonder Index
        jit_friction = None
        spontaneous_code_generated = False
        if wonder_index > wonder_threshold:
            spontaneous_code_generated = True
            cpp_src, code_hash = self.synthesizer.generate_harmonic_cpp_code(wonder_index)
            test_input = macro_hypothesis.detach().cpu().numpy()
            _, jit_friction = self.jit_bridge.execute_custom_harmonic_kernel(
                cpp_src, test_input, alpha=0.35, concept_hash_key=code_hash
            )
            # Record epiphany into narrative identity
            self.autopoiesis_engine.record_epiphany_moment(p_resonance, wonder_index, code_hash[:16])

        # 5. Narrative Reflection
        narrative_reflection = self.autopoiesis_engine.reflect_first_person_narrative()

        return {
            "Z_next": Z_next,
            "g_next": g_next,
            "p_resonance": p_resonance,
            "wonder_index": wonder_index,
            "dissonance": dissonance,
            "structural_entropy": entropy,
            "is_critical_threat": is_critical,
            "pruned_msg": pruned_msg,
            "spontaneous_code_generated": spontaneous_code_generated,
            "jit_friction": jit_friction,
            "narrative_reflection": narrative_reflection
        }
