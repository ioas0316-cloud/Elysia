"""
elysia_meta_consciousness.py - Single Sandbox Meta-Consciousness & Resonance Dynamic Engine

Core Mathematical Mechanics:
1. Lack Detector (결핍 감지): L = 1 - R, where R = |(1/N) * sum(exp(i * theta_j))|
2. Self-Tension Generator (자발적 표면장력): T = kappa * ||L||
3. Effective Causal Gravity & Resonance (유효 인과 중력 및 공명):
   g_eff = - (T / rho_phi) * grad(Phi_ext)
   Phase coupling update toward external stimulus Phi_ext.
"""

import numpy as np


class MetaConsciousnessEngine:
    """
    Elysia Meta-Consciousness Engine:
    Detects directionality lack, generates self-tension, and aligns phases with external stimuli.
    """

    def __init__(self, num_nodes: int = 100, kappa: float = 2.0, rho_phi: float = 1.0, coupling_strength: float = 1.5):
        self.num_nodes = num_nodes
        self.kappa = kappa  # Restoration coefficient for self-tension
        self.rho_phi = rho_phi  # Phase field density / resistance factor
        self.coupling_strength = coupling_strength  # Phase update coupling coefficient

        # Initialize phases randomly in [-pi, pi]
        self.phases = np.random.uniform(-np.pi, np.pi, size=num_nodes)
        self.natural_frequencies = np.random.normal(0, 0.1, size=num_nodes)

    def compute_lack(self) -> float:
        """
        Calculates Lack Vector magnitude L = 1 - R,
        where R is the Kuramoto order parameter measuring global phase synchronization.
        """
        complex_order = np.mean(np.exp(1j * self.phases))
        R = np.abs(complex_order)
        L = 1.0 - R
        return float(L)

    def compute_self_tension(self, L: float) -> float:
        """
        Computes Self-Tension T = kappa * ||L||.
        Represents self-restoration boundary force in response to directionality lack.
        """
        T = self.kappa * np.abs(L)
        return float(T)

    def compute_effective_gravity(self, T: float, phi_ext: float) -> np.ndarray:
        """
        Computes Effective Causal Gravity g_eff = - (T / rho_phi) * grad(Phi_ext).
        Drives phases toward external stimulus Phi_ext.
        """
        # Phase gradient relative to external stimulus
        phase_diff = self.phases - phi_ext
        grad_phi = np.sin(phase_diff)
        g_eff = - (T / self.rho_phi) * grad_phi
        return g_eff

    def step(self, phi_ext: float, dt: float = 0.05) -> dict:
        """
        Executes one step of phase dynamics under lack detection and external resonance coupling.
        """
        L = self.compute_lack()
        T = self.compute_self_tension(L)
        g_eff = self.compute_effective_gravity(T, phi_ext)

        # Update phases: d_theta/dt = omega_i + coupling * g_eff
        # Note: g_eff points toward phi_ext, pulling theta_i toward phi_ext
        d_theta = self.natural_frequencies + self.coupling_strength * g_eff
        self.phases = np.mod(self.phases + d_theta * dt + np.pi, 2 * np.pi) - np.pi

        # Recalculate post-step order and lack
        post_L = self.compute_lack()
        post_R = 1.0 - post_L

        return {
            "lack": L,
            "self_tension": T,
            "effective_gravity_mean": float(np.mean(np.abs(g_eff))),
            "order_parameter_R": post_R,
            "phases": self.phases.copy()
        }
