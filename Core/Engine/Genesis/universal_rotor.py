"""
Core.Engine.Genesis.universal_rotor
===================================
The Engine of ANY Process.

Instead of hardcoded 'Gravity' or 'Economy', this rotor takes a LAW (function).
"""

from typing import Callable, Any, Dict
from Core.Foundation.Nature.rotor import Rotor, RotorConfig

class UniversalRotor(Rotor):
    """
    A Rotor that spins a Law, not just a value.
    
    Law Signature: (current_state, dt, context) -> new_state
    """
    def __init__(self, name: str, law: Callable, config: RotorConfig):
        super().__init__(name, config)
        self.law = law
        self.context = {} # The environment (Monads)
        
    def bind_context(self, context: Dict[str, Any]):
        """Connect the rotor to the world (Monads)."""
        self.context = context
        
    def tick(self, dt: float):
        """
        Execute the Law.
        The Rotor's 'Energy' determines the Intensity of the Law.
        """
        super().update(dt) # Standard spin physics
        
        # Apply the Law to the Context
        # Using Rotor Energy as a multiplier (Intensity)
        if self.energy > 0.1:
            self.law(self.context, dt, self.energy)

# ==============================================================================

"""
Core.Engine.Genesis.concept_monad
=================================
The Substance of ANY Thing.

A Monad can be an Electron, a Dollar, or a Feeling.
"""
import uuid
from dataclasses import dataclass, field

@dataclass
class ConceptMonad:
    """
    The Universal Particle.
    """
    name: str
    domain: str # 'Physics', 'Economy', 'Mind'
    val: float  # Mass, Price, Intensity
    
    # Dynamic Properties (The DNA)
    props: Dict[str, Any] = field(default_factory=dict)
    
    # Unique ID
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __repr__(self):
        return f"<{self.domain}:{self.name} val={self.val:.2f}>"
