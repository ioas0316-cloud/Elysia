"""
Creative Axiom Cell
===================
Specialized in Creativity, Flux, and Emergence.
"""

import jax.numpy as jnp
from Core.Cognition.resonant_cell import ResonantCell
from Core.System.universal_constants import AXIOM_CREATIVITY

class CreativeAxiom(ResonantCell):
    """
    The catalyst of the system.
    Attracts Void (potential) and unpredictable sparks.
    Prevents the system from settling into a stagnant equilibrium.
    """

    def __init__(self):
        super().__init__("CreativeAxiom")
        # Focus on: Emotion (1), Intuition (2), Resonance (4), Spirit/Mystery (6)
        self.will_mask = jnp.array([0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0])
        print(f"Cell {self.cell_id}: Initialized.")

    def pulse(self, global_intent: jnp.ndarray):
        """
        The Creative Axiom listens to the Body/Drive strand (dimensions 0-6).
        """
        # Extract Body sector
        body_resonance = global_intent[0:7]
        
        # Apply resonance with focus on 'Mystery' and 'Sparks'
        self._apply_resonance(body_resonance, influence=0.5)
        
        # Emergence is now an organic outcome of the resonance pulse.
             
        print(f"CreativeAxiom Pulse: Resonance {jnp.sum(self.space_7d):.2f}")
