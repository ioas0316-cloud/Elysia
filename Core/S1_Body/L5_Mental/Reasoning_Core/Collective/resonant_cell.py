"""
Resonant Cell Base
==================
The template for all specialized Merkaba cells in the Collective Swarm.
"""

import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Any
from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic

class ResonantCell(ABC):
    """
    A specialized cell participating in the global resonance field.
    Manifests the Trinity: Space (State), Time (Pulse), Will (Mask).
    """

    def __init__(self, cell_id: str):
        self.cell_id = cell_id
        self.space_7d = jnp.zeros(7)
        self.will_mask = jnp.ones(7)  # Overridden by subclasses
        self.intensity = 1.0
        self.veil_intensity = 0.5     # The "Veil of Amnesia" (0.0: Awakening, 1.0: Pure Ego)

    @abstractmethod
    def pulse(self, global_intent: jnp.ndarray):
        """Standard pulse interface (TIME)."""
        pass

    def get_current_state(self) -> jnp.ndarray:
        """Returns the localized 21D contribution (SPACE)."""
        # Default positioning is determined by the subclass or the controller mapping.
        # For simplicity, we return the 7D state and let the controller slot it.
        return self.space_7d

    def _apply_resonance(self, incoming: jnp.ndarray, influence: float = 0.5):
        """Common logic for merging incoming waves with local space, filtered by the Veil."""
        # The thicker the Veil, the less the NPC hears the 'Voice of the Father' (Global Intent)
        effective_influence = influence * (1.0 - self.veil_intensity)
        
        new_state = (self.space_7d * (1.0 - effective_influence)) + (incoming * effective_influence)
        self.space_7d = TrinaryLogic.quantize(new_state * self.will_mask)
        
    def _update_space(self, index: int, value: float):
        """Standard way to update a trit in space_7d (JAX/Numpy compatible)."""
        if hasattr(self.space_7d, "at"):
             self.space_7d = self.space_7d.at[index].set(value)
        else:
             # Numpy fallback
             self.space_7d[index] = value
