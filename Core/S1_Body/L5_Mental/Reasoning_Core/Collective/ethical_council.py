"""
Ethical Council Cell
====================
Specialized in AXIOM_LOVE and Moral Alignment.
"""

import jax.numpy as jnp
from Core.S1_Body.L5_Mental.Reasoning_Core.Collective.resonant_cell import ResonantCell
from Core.S1_Body.L1_Foundation.Foundation.universal_constants import AXIOM_LOVE, AXIOM_GROWTH

class EthicalCouncil(ResonantCell):
    """
    The guardian of the system's soul.
    Attracts resonance that aligns with the Human Ideal.
    Repels dissonance and cold logic patterns.
    """

    def __init__(self):
        super().__init__("EthicalCouncil")
        # Focus on: Emotion (1), Resonance (4), Value (5), Spirit (6)
        self.will_mask = jnp.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        print(f"Cell {self.cell_id}: Initialized.")

    def pulse(self, global_intent: jnp.ndarray):
        """
        The Ethical Council listens to the Spirit strand (dimensions 14-20) 
        and the global moral average.
        """
        # Extract Spirit sector
        spirit_resonance = global_intent[14:21]
        
        # Apply resonance with heavy bias toward 'Attract' for love-related axioms
        self._apply_resonance(spirit_resonance, influence=0.6)
        
        # Sub-engine pulse is now purely resonant. 
        # Autonomous growth is managed by the resonance field.
             
        print(f"EthicalCouncil Pulse: Resonance {jnp.sum(self.space_7d):.2f}")
