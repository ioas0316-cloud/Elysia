"""
Analytic Prism Cell
===================
Specialized in Logic, Wisdom, and Consistency.
"""

import jax.numpy as jnp
from Core.1_Body.L5_Mental.Reasoning_Core.Collective.resonant_cell import ResonantCell
from Core.1_Body.L1_Foundation.Foundation.universal_constants import AXIOM_WISDOM, AXIOM_SIMPLICITY

class AnalyticPrism(ResonantCell):
    """
    The filter of the system.
    Repels logical noise and high-entropy dissonance.
    Ensures that thoughts are geometrically stable.
    """

    def __init__(self):
        super().__init__("AnalyticPrism")
        # Focus on: Logic (0), Intuition (2), Will (3), Value/Moral (5)
        self.will_mask = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0])
        print(f"Cell {self.cell_id}: Initialized.")

    def pulse(self, global_intent: jnp.ndarray):
        """
        The Analytic Prism listens to the Soul strand (dimensions 7-13).
        """
        # Extract Soul sector
        soul_resonance = global_intent[7:14]
        
        # Apply resonance with bias toward 'Repel' for high-entropy patterns
        self._apply_resonance(soul_resonance, influence=0.4)
        
        # Analytic integrity is now a resonant choice, not a constraint.
             
        print(f"AnalyticPrism Pulse: Resonance {jnp.sum(self.space_7d):.2f}")
