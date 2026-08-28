import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from synaptic_architecture.causal_membrane import CausalMembrane

@dataclass
class EngramSymbol:
    """
    Extracted minimal-action symbol/engram resulting from converged potential friction.
    """
    symbol_id: str
    engram_vector: np.ndarray
    causal_why_reason: str
    mass: float = 1.0
    invariant_type: str = "spatial"  # 'spatial' (static image-like) or 'spatiotemporal' (video/flow-like)

class EmbodiedCausalSubstrate:
    """
    Geometric Sensory Substrate & Structural Data Lens.
    - Treats incoming data not as passive materials, but as active refracting lenses (Data Lenses).
    - Incorporates Embodied Bounded Substrate: physical/energy constraints on processing.
    - Self-Structuring: Distinguishes spatial invariants (static image) vs spatiotemporal trajectories (video)
      without pre-defined human labels.
    - Minimal Action Symbol Extraction: Transduces friction convergence into EngramSymbols.
    """

    def __init__(self, vector_dim: int = 8, energy_limit: float = 100.0):
        self.vector_dim = vector_dim
        self.energy_limit = energy_limit
        self.current_energy = energy_limit
        self.membrane = CausalMembrane(vector_dim=vector_dim)

        # Engram memory store
        self.engram_bank: List[EngramSymbol] = []

    def set_intent(self, intent_vector: np.ndarray) -> np.ndarray:
        """Sets top-down intent through membrane."""
        return self.membrane.project_top_down_intent(intent_vector)

    def process_data_as_lens(
        self,
        data_stream: List[np.ndarray],
        time_steps: Optional[List[float]] = None,
        layer_name: str = "C_meso"
    ) -> Dict[str, Any]:
        """
        Processes sensory data stream through Structural Data Lens.
        Analyzes topological invariance across time to self-discover if data is spatial (image) or spatiotemporal (video).

        Returns:
            Dict containing:
                - invariant_type: 'spatial' or 'spatiotemporal'
                - noise_reduction_ratio: float (0.0 to 1.0)
                - converged_engram: Optional[EngramSymbol]
                - total_friction: float
        """
        if not data_stream:
            raise ValueError("Data stream cannot be empty")

        # 1. Energy consumption (Self-Sacrifice / Embodied Constraint)
        energy_cost = len(data_stream) * 0.5
        self.current_energy = max(0.0, self.current_energy - energy_cost)

        # 2. Filter data stream through Membrane & calculate Phase Attenuation
        filtered_stream = []
        friction_list = []
        raw_noise_energies = []
        filtered_noise_energies = []

        for frame in data_stream:
            filtered, resonance, friction = self.membrane.interact_bottom_up_resistance(frame, layer_name=layer_name)
            filtered_stream.append(filtered)
            friction_list.append(friction)

            # Measure noise energy reduction (perpendicular component suppression)
            raw_noise = np.linalg.norm(frame - self.membrane.intent_potential)
            filtered_noise = np.linalg.norm(filtered - self.membrane.intent_potential)
            raw_noise_energies.append(raw_noise)
            filtered_noise_energies.append(filtered_noise)

        avg_raw_noise = np.mean(raw_noise_energies) + 1e-8
        avg_filtered_noise = np.mean(filtered_noise_energies)
        noise_reduction_ratio = max(0.0, 1.0 - (avg_filtered_noise / avg_raw_noise))

        # 3. Self-Discovery & Structuring: Spatial (Image) vs Spatiotemporal (Video)
        # Calculate variance of filtered state across time steps
        if len(data_stream) > 1:
            stream_matrix = np.array(filtered_stream)
            temporal_variance = np.var(stream_matrix, axis=0)
            mean_temp_variance = float(np.mean(temporal_variance))
        else:
            mean_temp_variance = 0.0

        # Threshold to discern static structural invariant vs trajectory
        if mean_temp_variance < 0.05:
            invariant_type = "spatial"  # Static invariant lens (e.g. image snapshot)
            why_desc = "Static spatial boundary invariance detected under intent lens."
        else:
            invariant_type = "spatiotemporal"  # Dynamic state trajectory (e.g. video/motion flow)
            why_desc = f"Continuous spatiotemporal state trajectory variance ({mean_temp_variance:.4f}) detected."

        # 4. Spontaneous Minimal Action Symbol Transduction
        avg_friction = float(np.mean(friction_list))
        converged_engram = None

        if avg_friction < 2.0:  # Suitably converged minimal action path
            compact_vector = np.mean(filtered_stream, axis=0)
            engram_id = f"engram_{len(self.engram_bank) + 1}_{invariant_type}"
            converged_engram = EngramSymbol(
                symbol_id=engram_id,
                engram_vector=compact_vector,
                causal_why_reason=why_desc,
                mass=float(1.0 / (avg_friction + 1e-5)),
                invariant_type=invariant_type
            )
            self.engram_bank.append(converged_engram)

        return {
            "invariant_type": invariant_type,
            "temporal_variance": mean_temp_variance,
            "noise_reduction_ratio": float(noise_reduction_ratio),
            "converged_engram": converged_engram,
            "total_friction": avg_friction,
            "remaining_energy": self.current_energy
        }
