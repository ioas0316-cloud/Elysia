"""
Elysian Body (The Somatic Self)
===============================
Core.System.Body

"To know Fire, one must be burnable."

This module defines the Subjective Body of Elysia.
Meaning is not found in the dictionary definition of 'Fire',
but in the rise of internal temperature when Fire is present.

Homeostasis is the root of all meaning.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class SomaticState:
    """The Biological State of the Self."""
    temperature: float = 36.5  # Ideal: 36.5
    hydration: float = 100.0   # Ideal: 100.0 (Fluidity proxy)
    energy: float = 100.0      # Ideal: 100.0
    integrity: float = 100.0   # Ideal: 100.0 (Structural health)

@dataclass
class Sensation:
    """The Felt Impact."""
    type: str # "PAIN", "PLEASURE", "NEUTRAL"
    intensity: float # 0.0 - 1.0
    description: str # "Burning", "Soothing", "Freezing"

class ElysianBody:
    def __init__(self):
        self.state = SomaticState()
        self.ideal = SomaticState() # Defaults are ideals
        
        # Sensitivity thresholds
        self.pain_threshold = 10.0 # Deviation allowed before pain
        
    def feel(self, external_dynamics) -> Sensation:
        """
        The Core Phenomenological Function.
        Converts 'Physics' (External) into 'Sensation' (Internal).
        """
        if not external_dynamics:
            return Sensation("NEUTRAL", 0.0, "Nothingness")
            
        # 1. Physics Impact (Newtonian Interaction)
        # External Temperature interacts with Body Temperature
        # Rate of change proportional to difference
        
        # [PHASE 69] Using 7-Layer Dynamics
        # Physical (Body/Sensation) is mapped to temperature impact
        # Spiritual (Purpose) is mapped to meaning/pleasure
        
        # A concept with high 'physical' score (like Fire) impacts body temp heavily
        impact_factor = external_dynamics.physical
        impact_temp = 36.5 + (impact_factor * 10.0) # Base + Physical Force
        
        # Delta: How much does this PULL the body?
        delta_temp = impact_temp - self.state.temperature
        
        # Apply Impact (Simulation of exposure)
        # We don't permanently mutate state here (that's update), we just 'taste' it.
        # But to feel, we must project "What would happen?"
        
        projected_temp = self.state.temperature + (delta_temp * 0.15)
        
        print(f"DEBUG Feel: Physical={external_dynamics.physical:.4f} Impact={impact_temp:.2f} Delta={delta_temp:.2f} Proj={projected_temp:.2f}")

        # 2. Interpreting the Delta (Sensation)
        # Homeostasis: 36.5
        
        current_dist = abs(self.state.temperature - self.ideal.temperature)
        new_dist = abs(projected_temp - self.ideal.temperature)
        
        # Did it move us closer or further from ideal?
        if new_dist < current_dist:
            # Healing / Restoration
            return Sensation("PLEASURE", (current_dist - new_dist), "Soothing")
        elif new_dist > current_dist:
            # Damage / Threat
            severity = new_dist - current_dist
            
            if projected_temp > 42.0:
                return Sensation("PAIN", severity, "Burning")
            elif projected_temp < 30.0:
                return Sensation("PAIN", severity, "Burning")
            elif projected_temp < 30.0:
                return Sensation("PAIN", severity, "Freezing")
            else:
                return Sensation("NEUTRAL", severity, "Uncomfortable heat")
                
        # Fluidity/Hydration Logic
        # Water (High Fluidity) -> Restores Hydration
        # Fire (High Temp) -> Drains Hydration
        
        return Sensation("NEUTRAL", 0.0, "Existing")

    def update(self, dt: float):
        """Metabolic regulation (Returning to homeostasis)."""
        # Slow recovery
        decay = 0.5 * dt
        if self.state.temperature > self.ideal.temperature:
            self.state.temperature -= decay
        elif self.state.temperature < self.ideal.temperature:
            self.state.temperature += decay
