import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class MembraneLayer:
    """
    Represents a specific layer of the multi-layered causal membrane (onion structure).
    - C_macro: Outer layer (real world, human language/intent, physical constraints)
    - C_meso: Middle layer (digital twin / sandbox environment, temporal dynamics)
    - C_micro: Inner layer (micro potential field, agent's internal state & self-preservation)
    """
    name: str
    precision: float = 1.0  # Active Inference precision weighting (Π)
    resistance_coefficient: float = 0.5
    potential_offset: np.ndarray = field(default_factory=lambda: np.zeros(8))

class CausalMembrane:
    """
    Multi-Layered Causal Membrane (Markov Blanket Boundary Dynamics).
    Represents the dynamic interface where top-down intention vector meets
    bottom-up sensory friction & environmental resistance.
    """

    def __init__(self, vector_dim: int = 8):
        self.vector_dim = vector_dim
        self.macro_layer = MembraneLayer(name="C_macro", precision=1.2, resistance_coefficient=0.8, potential_offset=np.zeros(vector_dim))
        self.meso_layer = MembraneLayer(name="C_meso", precision=1.0, resistance_coefficient=0.5, potential_offset=np.zeros(vector_dim))
        self.micro_layer = MembraneLayer(name="C_micro", precision=0.8, resistance_coefficient=0.2, potential_offset=np.zeros(vector_dim))

        # Internal potential baseline (Intent state)
        self.intent_potential = np.zeros(self.vector_dim)
        # Membrane boundary tension / friction energy
        self.boundary_tension = 0.0

    def project_top_down_intent(self, intent_vector: np.ndarray) -> np.ndarray:
        """
        Projects top-down intent through the nested membrane layers.
        Shape: (vector_dim,)
        """
        self.intent_potential = intent_vector.copy()
        # Modulate intent through precision weighting across layers
        projected = (
            intent_vector * self.macro_layer.precision +
            intent_vector * self.meso_layer.precision +
            intent_vector * self.micro_layer.precision
        ) / 3.0
        return projected

    def interact_bottom_up_resistance(
        self,
        raw_sensory_signal: np.ndarray,
        layer_name: str = "C_meso"
    ) -> Tuple[np.ndarray, float, float]:
        """
        Calculates bottom-up physical friction & top-down/bottom-up phase cancellation.
        Returns:
            - Filtered Signal (Signal after noise attenuation)
            - Resonance Score (0.0 ~ 1.0)
            - Friction Tension Energy
        """
        if raw_sensory_signal.shape[0] != self.vector_dim:
            # Handle dimension alignment if needed
            padded = np.zeros(self.vector_dim)
            min_dim = min(self.vector_dim, raw_sensory_signal.shape[0])
            padded[:min_dim] = raw_sensory_signal[:min_dim]
            raw_sensory_signal = padded

        layer = {
            "C_macro": self.macro_layer,
            "C_meso": self.meso_layer,
            "C_micro": self.micro_layer
        }.get(layer_name, self.meso_layer)

        # 1. Active Inference Phase Filter / Resonance calculation
        # Normalized dot product with current intent vector
        intent_norm = np.linalg.norm(self.intent_potential) + 1e-8
        sensory_norm = np.linalg.norm(raw_sensory_signal) + 1e-8

        dot_product = np.dot(self.intent_potential, raw_sensory_signal)
        resonance_score = max(0.0, float(dot_product / (intent_norm * sensory_norm)))

        # 2. Phase Cancellation of Irrelevant Noise (Top-down Attenuation)
        # Noise component perpendicular to intent vector is attenuated according to layer precision
        intent_unit = self.intent_potential / intent_norm
        parallel_component = np.dot(raw_sensory_signal, intent_unit) * intent_unit
        perpendicular_component = raw_sensory_signal - parallel_component

        # Attenuate noise component by precision weighting Π
        # High precision on intent suppresses perpendicular noise
        attenuated_noise = perpendicular_component / (1.0 + layer.precision * 5.0)
        filtered_signal = parallel_component + attenuated_noise

        # 3. Calculate Membrane Friction Tension
        # Difference between projected intent and environmental reaction
        friction = np.linalg.norm(self.intent_potential - raw_sensory_signal) * layer.resistance_coefficient
        self.boundary_tension = float(0.8 * self.boundary_tension + 0.2 * friction)

        # Dynamic offset adjustment on higher layer if friction is high (Reciprocal feedback)
        if friction > 1.5:
            layer.potential_offset += 0.1 * (raw_sensory_signal - self.intent_potential)

        return filtered_signal, resonance_score, float(friction)
