import numpy as np
from typing import Tuple
from .field import MemristiveField
from .vortex import VortexConvergence

class SynapticOrganism:
    """
    [Synaptic Architecture] The Digital Organism
    Integrates Field, Vortex, and Self-Wiring logic.
    """
    def __init__(self, resolution: int = 256):
        self.field = MemristiveField(resolution)
        self.vortex = VortexConvergence(self.field)
        self.tension_threshold = 0.8

    def induce_synapse(self, data_a: np.ndarray, data_b: np.ndarray):
        """
        [Self-Wiring]
        Two information pieces attract each other if they have resonance/complementarity.
        If tension > threshold, they 'slide' together in the field.
        """
        # 1. Calculate resonance (Attraction force)
        # Use the resonance_map from vortex to check for field-wide attraction
        res_map_a = self.vortex.resonance_map(data_a)
        res_map_b = self.vortex.resonance_map(data_b)

        # Simple global max resonance for these patterns
        res = np.max(res_map_a * res_map_b)

        if res > 0.1: # Threshold lowered for demo/sliding visibility
            # Attraction happens!
            # Find where they are (vortices)
            pos_a = self.vortex.converge_to_vortex(data_a)
            pos_b = self.vortex.converge_to_vortex(data_b)

            print(f"[Self-Wiring] Resonance ({res:.4f}) detected between {pos_a} and {pos_b}")

            # Move them closer in the field (Sliding pointers)
            target_pos = (pos_a + pos_b) / 2.0

            # Leave a heavy conductance trace in the bridge (the 'synapse')
            self._bridge_trace(pos_a, pos_b)

            # Re-deposit data closer to each other (Physical alignment)
            self.field.deposit_engram(target_pos, (data_a + data_b) / 2.0)
            print(f"  > Synaptic bridge formed at {target_pos}")

    def _bridge_trace(self, pos_a: np.ndarray, pos_b: np.ndarray):
        """Create a path of high conductance between two points."""
        # Simple linear bridge
        steps = 10
        for i in range(steps + 1):
            p = pos_a + (pos_b - pos_a) * (i / steps)
            self.field.propagate_signal(p, 2.0)

if __name__ == "__main__":
    organism = SynapticOrganism()

    # Create two resonating patterns
    p1 = np.random.randn(64)
    p2 = p1 + np.random.normal(0, 0.1, 64) # High resonance

    organism.field.deposit_engram(np.array([10, 10]), p1)
    organism.field.deposit_engram(np.array([100, 100]), p2)

    organism.induce_synapse(p1, p2)
