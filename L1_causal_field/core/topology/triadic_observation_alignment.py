"""
Trinity (Self-Other-Meta World) Observation Lens Diagnostic System (2nd Priority Core)
-----------------------------------------------------------------------
Implements the triadic observation lens alignment dynamics:
  1. Self Phase (Subjective Lens): Internal 64-bit causal potential node state.
  2. Other Phase (Objective / Other Lens): External phenomenon node state.
  3. Meta World Space (Trinity Coordinate): Higher meta-space evaluating causal refraction (Bias)
     and delta (Δ = |Self - Other|) to prevent closed-world delusional state and maintain
     a 'Dispassionate Mirror' condition.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from core.physics.causal_engine import CausalNodePy, PHASE_GAS, PHASE_FLUID, PHASE_CRYSTAL


class TrinityDiagnosticLens:
    """
    Evaluates real-time causal refraction bias and state delta across
    Self (Subject), Other (Object), and Meta-World perspectives.
    """

    def __init__(self, refraction_decay_rate: float = 0.05, max_bias_tolerance: float = 500.0):
        self.refraction_decay_rate = refraction_decay_rate
        self.max_bias_tolerance = max_bias_tolerance

        # Historical bias trajectory (Refraction Memory)
        self.bias_history: list[float] = []

    def observe_and_diagnose(self, self_node: CausalNodePy, other_node: CausalNodePy) -> Dict[str, float]:
        """
        Calculates causal refraction (Bias) and Delta (Δ) in Meta-World space.
        """
        self_pot = self_node.potential
        other_pot = other_node.potential

        # Causal Delta (Δ = |Self - Other|)
        delta_potential = abs(float(self_pot) - float(other_pot))

        # Causal Refraction Index (Bias): Phase misalignment and offset tension
        phase_discrepancy = 1.0 if self_node.phase_state != other_node.phase_state else 0.0
        topo_discrepancy = abs(self_node.topo_offset - other_node.topo_offset)

        bias = delta_potential + (phase_discrepancy * 200.0) + (topo_discrepancy * 10.0)
        self.bias_history.append(bias)

        # Mirror Purity Index (1.0 = Pure Dispassionate Mirror, 0.0 = Closed World Delusional Fall)
        mirror_purity = float(np.exp(-bias / max(1.0, self.max_bias_tolerance)))

        return {
            "delta_potential": delta_potential,
            "bias_refraction": bias,
            "phase_discrepancy": phase_discrepancy,
            "mirror_purity": mirror_purity,
            "is_closed_world_risk": float(bias > self.max_bias_tolerance)
        }

    def calibrate_self_lens(self, self_node: CausalNodePy, other_node: CausalNodePy, diagnostic: Dict[str, float]) -> CausalNodePy:
        """
        Applies self-calibration feedback loop to eliminate observer bias (Self-Refraction),
        re-aligning the Self CausalNode to mirror the true causal mechanics of the Other node.
        """
        if diagnostic["mirror_purity"] >= 0.95:
            return self_node

        calibrated = CausalNodePy(self_node.raw)

        # 1. Potential alignment gradient towards real external resistance
        delta_pot = float(other_node.potential) - float(self_node.potential)
        adjusted_pot = int(self_node.potential + delta_pot * self.refraction_decay_rate * 5.0)
        calibrated.potential = max(0, min(0xFFFFFF, adjusted_pot))

        # 2. Topology offset healing towards objective grounding
        if self_node.topo_offset != other_node.topo_offset:
            shift = 1 if other_node.topo_offset > self_node.topo_offset else -1
            calibrated.topo_offset = self_node.topo_offset + shift

        # 3. Phase state relaxation if bias is critically high
        if diagnostic["is_closed_world_risk"] > 0.5:
            calibrated.phase_state = PHASE_FLUID

        return calibrated
