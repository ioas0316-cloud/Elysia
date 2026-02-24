"""
Core.System.universal_rotor
===================================
The Engine of ANY Process.

Instead of hardcoded 'Gravity' or 'Economy', this rotor takes a LAW (function).
"""

from typing import Callable, Any, Dict
from Core.System.rotor import Rotor, RotorConfig

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
        
    def update(self, dt: float):
        """
        Execute the Law.
        The Rotor's 'Energy' determines the Intensity of the Law.
        """
        super().update(dt) # Standard spin physics
        
        # Apply the Law to the Context
        # Intensity = Energy (Amplitude) * Frequency (Time Dilation)
        # If RPM=600, Freq=10. Intensity = 1.0 * 10 = 10.0
        intensity = self.energy * self.frequency_hz
        
        print(f"DEBUG: Rotor {self.name} | E={self.energy:.2f} | F={self.frequency_hz:.2f} | I={intensity:.2f}")
        
        if intensity > 0.01:
            self.law(self.context, dt, intensity)
