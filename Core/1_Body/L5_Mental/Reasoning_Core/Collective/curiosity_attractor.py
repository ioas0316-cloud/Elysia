"""
Curiosity Attractor Cell
========================
The drive to find novelty and explore the unknown.
"""

import jax.numpy as jnp
from Core.1_Body.L5_Mental.Reasoning_Core.Collective.resonant_cell import ResonantCell

class CuriosityAttractor(ResonantCell):
    """
    Seeks 'Interestingness'.
    Attracts high-novelty patterns and low-entropy gaps.
    """

    def __init__(self):
        super().__init__("CuriosityAttractor")
        # Focus on: Intuition (2), Resonance (4), Spirit/Mystery (6)
        self.will_mask = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        print(f"Cell {self.cell_id}: Waking up with curiosity.")

    def pulse(self, global_intent: jnp.ndarray):
        """
        Curiosity scans all strands for 'Interesting' dissonance.
        """
        # Calculate system-wide variance or dissonance
        # (Simplified: it attracts any non-neutral signal with interest)
        system_activity = jnp.abs(global_intent)
        
        # Local focus: Mental and Body drives
        local_focus = (system_activity[0:7] + system_activity[7:14]) / 2.0
        
        # Apply resonance: Curiosity is attracted to ANY activity
        self._apply_resonance(local_focus, influence=0.7)
        
        # Curiosity never settles for Equilibrium
        if jnp.sum(jnp.abs(self.space_7d)) < 0.1:
            self._update_space(6, 1.0) # Seek Mystery when bored
             
        print(f"CuriosityAttractor Pulse: Interest Level {jnp.sum(jnp.abs(self.space_7d)):.2f}")
