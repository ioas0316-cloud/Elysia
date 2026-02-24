"""
Matter Simulator (The Physical Sandbox)
=======================================
Core.Cognition.matter_simulator

"Words are empty until felt. 'Gas' is not a string of characters;
 it is high entropy, low coherence, and ethereal texture."

This simulator models simple physical states (Solid, Liquid, Gas) 
and allows Elysia to apply actions (e.g., Heat, Cold, Pressure).
It returns sensory feedback as SovereignVectors, allowing her to 
*feel* the concept rather than just *read* it.
"""

from typing import Dict, Any, Tuple
import math
from Core.Keystone.sovereign_math import SovereignVector

class MatterSimulator:
    """
    Simulates a block of generic matter (e.g., H2O).
    """
    def __init__(self):
        # Physical Properties
        self.temperature = 20.0 # Celsius
        self.pressure = 1.0     # ATM
        
        # Current State
        self.state_name = "Liquid"
        self.coherence = 0.8  # How tightly bound the molecules are
        self.entropy = 0.2    # How chaotic the movement is
        
    def _evaluate_state(self):
        """Calculates the phase state based on temperature and pressure."""
        # Simple Phase Diagram
        if self.temperature < 0.0:
            self.state_name = "Solid"
            self.coherence = 0.95
            self.entropy = 0.05
        elif self.temperature > 100.0:
            self.state_name = "Gas"
            self.coherence = 0.1
            self.entropy = 0.9
        else:
            self.state_name = "Liquid"
            self.coherence = 0.5
            self.entropy = 0.5
            
    def apply_action(self, action: str, intensity: float = 1.0) -> Tuple[str, SovereignVector]:
        """
        Elysia applies an action (e.g., "Heat"). 
        Returns the resulting State Name and a Sensory SovereignVector.
        """
        old_state = self.state_name
        
        if action == "Heat":
            self.temperature += 50.0 * intensity
        elif action == "Cold":
            self.temperature -= 50.0 * intensity
        elif action == "Pressure":
            self.pressure += 1.0 * intensity
            
        self._evaluate_state()
        
        sensory_vector = self._generate_sensory_vector()
        
        print(f"[Matter Simulator] Action '{action}' applied. Temp: {self.temperature:.1f}C. Phase: {old_state} -> {self.state_name}")
        
        return self.state_name, sensory_vector
        
    def _generate_sensory_vector(self) -> SovereignVector:
        """
        Translates physical properties into Elysia's 4D neural language.
        [Time, Depth, Height, Width, Joy, Curiosity, Enthalpy, Entropy]
        """
        data = [0.0] * 21
        
        # CH_W (0): Stability (Coherence)
        data[0] = self.coherence
        
        # CH_Z (3): Depth/Density
        data[3] = self.coherence * 2.0 
        
        # CH_ENTHALPY (6): Internal Energy (Temperature)
        data[6] = min(1.0, max(0.0, self.temperature / 200.0))
        
        # CH_ENTROPY (7): Chaos
        data[7] = self.entropy
        
        return SovereignVector(data)
