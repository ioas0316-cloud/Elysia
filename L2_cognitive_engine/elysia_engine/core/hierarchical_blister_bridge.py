"""
L2_cognitive_engine/elysia_engine/core/hierarchical_blister_bridge.py

Hierarchical Decoupled Blister Bridge (Micro Reflex to Macro Helmholtz & 3x3x3 Fractal Space)
===========================================================================================
Implements the 2-Tier Hierarchical Decoupling Bridge inspired by biological reflex arcs:
1. Micro-Scale (Spinal Reflex): Fast local clumping & blister snapping of OpticalSpectrumBivectorRotors.
2. Causal Gating (Threshold R_gate): Forwards coherent spectrum energy to Macro-Scale only when resonance exceeds threshold.
3. Macro-Scale (Cortical Integration):
   - Tensor Helmholtz Metric Wavefield decomposition at wavenumber k = 2pi / lambda.
   - Phase injection and resonance coupling with 3x3x3 Fractal Cognitive Nodes (27-node lattice).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np

from elysia_engine.core.optical_blister_rotor import (
    OpticalSpectrumBivectorRotor,
    ShadowBlisterTensionKernel,
    MicroReflexBlisterNetwork
)
from core.physics.tensor_helmholtz_decomposer import (
    TensorHelmholtzMetricDecomposer,
    MetricWaveDecompositionResult
)
from elysia_core.causal_cognitive_loop import FractalNode3x3x3


@dataclass
class BridgePropagationReport:
    """Report detailing multi-scale state transitions and causal gating."""
    micro_coherence: float
    is_gated: bool
    num_clumped_clusters: int
    macro_wave_modes: Optional[Dict[str, float]]
    fractal_coherence: float
    effective_wavenumber_k: float


class HierarchicalBlisterBridge:
    """
    Decoupled 2-Tier Architecture connecting Micro-Scale Blister Networks
    with Macro-Scale Helmholtz Wavefields and 3x3x3 Fractal Concept Spaces.
    """

    def __init__(
        self,
        r_gate: float = 0.55,                # Coherence gate threshold for macro cortical forwarding
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        fractal_coupling: float = 0.6
    ):
        self.r_gate = r_gate
        self.grid_shape = grid_shape

        # Micro-scale reflex network
        self.micro_network = MicroReflexBlisterNetwork()

        # Macro-scale components
        self.helmholtz_decomposer = TensorHelmholtzMetricDecomposer(grid_shape=grid_shape)
        self.fractal_3x3x3 = FractalNode3x3x3(coupling_strength=fractal_coupling)

        self.last_report: Optional[BridgePropagationReport] = None

    def register_rotor(self, rotor: OpticalSpectrumBivectorRotor) -> int:
        """Adds an optical bivector rotor to the micro reflex network."""
        return self.micro_network.add_rotor(rotor)

    def step(self, dt: float = 0.02) -> BridgePropagationReport:
        """
        Executes one hierarchical multi-scale cycle:
        1. Micro-scale step: local shadow blister attraction, stiction, auto-clumping.
        2. Gating: inspect order parameter R.
        3. If R >= R_gate, forward coherent spectrum wave to Helmholtz & 3x3x3 Fractal space.
        """
        # 1. Micro-scale reflex step
        micro_res = self.micro_network.step(dt=dt)
        r_coherence = micro_res["coherence"]
        clusters = micro_res["clusters"]

        is_gated = r_coherence >= self.r_gate

        macro_modes = None
        fractal_coherence = self.fractal_3x3x3.coherence
        effective_k = 1.0

        if is_gated and len(self.micro_network.rotors) > 0:
            # 2. Extract dominant optical spectrum from clumped clusters or whole network
            if len(clusters) > 0:
                # Use primary clumped cluster
                macro_rotor = self.micro_network.synthesize_cluster_rotor(clusters[0])
            else:
                macro_rotor = self.micro_network.synthesize_cluster_rotor(list(range(len(self.micro_network.rotors))))

            # Wavenumber k = 2pi / lambda (scaled for grid resolution)
            # lambda in [380, 750] nm -> k in [1.0, 3.0]
            effective_k = float((550.0 / macro_rotor.wavelength_nm) * 1.5)

            # Synthesize synthetic 3x3 metric perturbation field h_ij from rotor state
            h_tensor = self._generate_metric_perturbation(macro_rotor)

            # Decompose into Helmholtz Modes (Scalar, Vector Shear, Tensor TT)
            decomp = self.helmholtz_decomposer.decompose_metric_field(h_tensor, wave_number_k0=effective_k)
            macro_modes = decomp.energy_ratios

            # 3. Inject into 3x3x3 Fractal Concept Space
            # Create stimulus vector from rotor triad and phase
            stimulus = np.array([
                macro_rotor.triad_weights[0] * math.cos(macro_rotor.phase_theta),
                macro_rotor.triad_weights[1] * math.sin(macro_rotor.phase_theta),
                macro_rotor.triad_weights[2]
            ], dtype=np.float32)

            fractal_coherence = self.fractal_3x3x3.step_phase_locking(stimulus, dt=dt)

        report = BridgePropagationReport(
            micro_coherence=r_coherence,
            is_gated=is_gated,
            num_clumped_clusters=len(clusters),
            macro_wave_modes=macro_modes,
            fractal_coherence=fractal_coherence,
            effective_wavenumber_k=effective_k
        )
        self.last_report = report
        return report

    def _generate_metric_perturbation(self, rotor: OpticalSpectrumBivectorRotor) -> np.ndarray:
        """
        Synthesizes a 3D [3, 3, Nx, Ny, Nz] metric perturbation tensor h_ij(r)
        from the optical rotor bivector and spatial center of mass.
        """
        nx, ny, nz = self.grid_shape
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Distance from rotor position
        pos = rotor.position
        R2 = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
        envelope = np.exp(-0.5 * R2 / (rotor.penumbra_radius + 1e-4)**2)

        # Base 3x3 matrix from bivector cross terms (shear) + triad (diagonal trace)
        bv = rotor.get_bivector()
        tr = rotor.triad_weights

        base_mat = np.array([
            [tr[0], bv[2], bv[1]],
            [bv[2], tr[1], bv[0]],
            [bv[1], bv[0], tr[2]]
        ], dtype=np.float32)

        h_tensor = np.zeros((3, 3, nx, ny, nz), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                h_tensor[i, j] = base_mat[i, j] * envelope * math.cos(rotor.phase_theta)

        return h_tensor
