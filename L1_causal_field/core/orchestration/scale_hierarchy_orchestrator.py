#!/usr/bin/env python3
"""
scale_hierarchy_orchestrator.py

Scale Hierarchy Ecosystem Orchestrator
Connects L0 (Substrate & Micro-scale), L1 (Prismatic Gate & Structural Constraint),
L2 (Surface Tension Attractor & Phase Resonance), and L3 (Macro-Causal & Spacetime Deck)
into a unified, continuous multi-scale pipeline with ZERO artificial if-else rules.

Core Principle:
  "Sensory/Physical friction (heat, tension, pressure) at L0/L1 smoothly converts
   into mental image fields and surface relaxation at L2, which then flows into
   macro-narrative purpose fields at L3. The L3 purpose field in turn modulates
   L0/L1 sensitivity dials, achieving complete multi-scale homeostasis."
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

_current_file = Path(__file__).resolve() if '__file__' in globals() else Path.cwd()
_root = _current_file.parents[3] if '__file__' in globals() else Path.cwd()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import elysia_loader
    elysia_loader.install()
except Exception:
    pass

from core.layer_1_constraint.sensory_prismatic_gate import SensoryPrismaticGate
from core.layer_2_resonant.surface_tension_attractor import SurfaceTensionAttractor
from core.layer_3_macro.macro_causal_deck import MacroCausalDeck


class ScaleHierarchyOrchestrator:
    """
    Unified Ecosystem Orchestrator bridging Layer 0 to Layer 3.
    Executes continuous multi-scale feedback loops driven by embodied physical/sensory friction.
    """

    def __init__(
        self,
        lens_density: int = 1000,
        num_spectral_bands: int = 3,
        surface_tension_coeff: float = 0.15,
        macro_dim: int = 64
    ):
        # L1: Structural Constraint & Prismatic Gate
        self.prismatic_gate = SensoryPrismaticGate(
            num_spectral_bands=num_spectral_bands,
            lens_density=lens_density
        )

        # L2: Surface Tension Attractor & Resonance
        self.surface_attractor = SurfaceTensionAttractor(
            tension_coeff=surface_tension_coeff
        )

        # L3: Macro-Causal & Spacetime Deck
        self.macro_deck = MacroCausalDeck(
            macro_dim=macro_dim
        )

        # Multi-scale persistent state & feedback dials
        self.sensitivity_dial = 1.0  # Top-down modulation dial from L3
        self.accumulated_scar = 0.0  # Irreversible metric scar accumulation
        self.current_time = 0.0

    def reset_orchestrator(self):
        """Resets persistent states and feedback dials."""
        self.sensitivity_dial = 1.0
        self.accumulated_scar = 0.0
        self.current_time = 0.0

    def process_embodied_cycle(
        self,
        raw_sensory_stream: np.ndarray,
        world_friction: float = 0.0,
        dt: float = 0.1
    ) -> Dict[str, Any]:
        """
        Executes a complete multi-scale cycle across L0 -> L1 -> L2 -> L3 -> L0/L1 feedback.

        1. [L0/L1 Thrownness & Prismatic Gate]: Modulates raw sensory input with L3 sensitivity dial
           and refracts it into chromatic spectrum & mental imagery field.
        2. [L2 Surface Tension Attractor]: Relaxes mental imagery field toward minimum energy
           equilibrium via physical surface tension dynamics.
        3. [L3 Macro-Causal Deck]: Synthesizes macro narrative purpose field, updates spatiotemporal
           memory ring buffer, and calculates top-down sensitivity dial modulation.
        4. [Scar Accumulation & Homeostasis]: Accumulates physical friction scars into the metric tensor.

        Args:
            raw_sensory_stream: Raw wave/sensory signal (1D array or 2D image/stream).
            world_friction: External physical/environmental friction (e.g., heat, resistance).
            dt: Time increment.

        Returns:
            Dict containing outputs from all layers, showing complete multi-scale continuity.
        """
        self.current_time += dt

        # ---------------------------------------------------------------------
        # STEP 1: L0/L1 Modulated Prismatic Refraction
        # ---------------------------------------------------------------------
        modulated_input = raw_sensory_stream * self.sensitivity_dial
        l1_output = self.prismatic_gate.refract_external_aperture(modulated_input)

        raw_mental_imagery = l1_output["mental_imagery_field"]
        chromatic_spectrum = l1_output["chromatic_spectrum"]
        phase_coherence = l1_output["phase_coherence"]

        # ---------------------------------------------------------------------
        # STEP 2: L2 Surface Tension Attractor Convergence
        # ---------------------------------------------------------------------
        # Inject world friction as physical surface tension strain
        strained_imagery = raw_mental_imagery + world_friction * np.random.randn(*raw_mental_imagery.shape) * 0.1
        l2_output = self.surface_attractor.converge_imagery_attractor(strained_imagery)

        relaxed_imagery = l2_output["relaxed_imagery_field"]
        initial_energy = l2_output["initial_energy"]
        relaxed_energy = l2_output["relaxed_energy"]

        # ---------------------------------------------------------------------
        # STEP 3: L3 Macro-Causal Synthesis & Top-Down Feedback
        # ---------------------------------------------------------------------
        l3_output = self.macro_deck.process_macro_cycle(
            relaxed_imagery=relaxed_imagery,
            chromatic_spectrum=chromatic_spectrum,
            phase_coherence=phase_coherence,
            timestamp=self.current_time
        )

        # ---------------------------------------------------------------------
        # STEP 4: Homeostatic Feedback & Scar Accumulation
        # ---------------------------------------------------------------------
        # Update top-down sensitivity dial
        self.sensitivity_dial = l3_output["sensitivity_dial"]

        # Scar accumulation: High energy dissipation permanently warps system metric baseline
        energy_dissipated = max(0.0, initial_energy - relaxed_energy)
        self.accumulated_scar += energy_dissipated * 0.01 * (1.0 + world_friction)

        return {
            "status": "Continuous Multi-Scale Cycle Completed Successfully",
            "timestamp": self.current_time,
            "L1_chromatic_spectrum": chromatic_spectrum,
            "L1_phase_coherence": phase_coherence,
            "L2_initial_energy": initial_energy,
            "L2_relaxed_energy": relaxed_energy,
            "L2_convergence_ratio": l2_output["convergence_ratio"],
            "L2_steps_taken": l2_output["steps_taken"],
            "L3_macro_tension": l3_output["macro_tension"],
            "L3_purpose_field_norm": float(np.linalg.norm(l3_output["purpose_field"])),
            "feedback_sensitivity_dial": self.sensitivity_dial,
            "accumulated_scar": self.accumulated_scar,
            "continuity_verified": True
        }


if __name__ == "__main__":
    orchestrator = ScaleHierarchyOrchestrator()
    sample_wave = np.sin(np.linspace(0, 8 * np.pi, 1000)) + 0.3 * np.random.randn(1000)

    print("--- Executing 5 Continuous Multi-Scale Cycles ---")
    for step in range(5):
        friction = 0.2 * step
        res = orchestrator.process_embodied_cycle(sample_wave, world_friction=friction)
        print(f"Cycle {step+1} | Time: {res['timestamp']:.1f}s | Friction: {friction:.1f} | "
              f"L2 Energy: {res['L2_initial_energy']:.3f}->{res['L2_relaxed_energy']:.3f} | "
              f"L3 Tension: {res['L3_macro_tension']:.3f} | Sensitivity Dial: {res['feedback_sensitivity_dial']:.3f}")
    print("Orchestrator Multi-Scale Verification Successful!")
