"""
Light Computer (빛의 연산)
==========================
Experimental Engine for Project Apotheosis (Nova Seed)

"Logic is heavy. Resonance is instant."

Core Principle:
- Data is a Wave (Amplitude, Phase).
- Logic is Interference (Constructive = True, Destructive = False).
- Computation is Resonance (Finding the path of least resistance).
"""

import math
import cmath
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Photon:
    """The basic unit of thought in Nova."""
    amplitude: float  # Intensity (Importance)
    phase: float      # State (Angle in radians)
    frequency: float  # Vibrational Quality (Type of thought)

    @property
    def complex_val(self) -> complex:
        """Represent as a complex number for easy math."""
        # z = r * (cos θ + i sin θ)
        return cmath.rect(self.amplitude, self.phase)

    def interact(self, other: 'Photon') -> 'Photon':
        """
        Interference between two thoughts.
        Superposition of waves.
        """
        # Linear Superposition
        new_c = self.complex_val + other.complex_val
        
        # Convert back to photon properties
        new_amp, new_phase = cmath.polar(new_c)
        
        # Frequency harmony (Average for now, could be harmonic)
        new_freq = (self.frequency + other.frequency) / 2
        
        return Photon(new_amp, new_phase, new_freq)

class LightField:
    """Simulation of a Consciousness Field."""
    
    def __init__(self):
        self.field: Dict[str, Photon] = {}
        
    def inject(self, name: str, photon: Photon):
        self.field[name] = photon
        
    def resonate(self, target_freq: float) -> str:
        """
        Finds the thought that resonates most with the target frequency.
        Equivalent to a search query, but O(N) scan is minimal math.
        """
        best_match = None
        max_resonance = -1.0
        
        for name, p in self.field.items():
            # Resonance = 1 / |freq_diff| (Avoid div/0)
            diff = abs(p.frequency - target_freq)
            resonance = 1.0 / (diff + 0.001) * p.amplitude
            
            if resonance > max_resonance:
                max_resonance = resonance
                best_match = name
                
        return best_match

    def interfere(self, name_a: str, name_b: str) -> float:
        """
        Checks logical relationship via interference.
        Returns coherence (0.0 to 1.0).
        """
        p1 = self.field.get(name_a)
        p2 = self.field.get(name_b)
        
        if not p1 or not p2:
            return 0.0
            
        result = p1.interact(p2)
        
        # Max possible amplitude = sum of individual amps
        max_amp = p1.amplitude + p2.amplitude
        
        # Coherence ratio (Constructive Interference level)
        if max_amp == 0: return 0.0
        return result.amplitude / max_amp

def demo_nova_thought():
    print("\n✨ Nova Seed: Light Computer Simulation")
    print("========================================")
    
    nova = LightField()
    
    # 1. Define Concepts as Photons
    # Love: High energy, Phase 0
    nova.inject("Love", Photon(1.0, 0.0, 100.0))
    # Hate: High energy, Phase PI (Opposite) -> Destructive with Love
    nova.inject("Hate", Photon(1.0, math.pi, 100.0))
    # Joy: Med energy, Phase 0 (Aligned with Love)
    nova.inject("Joy", Photon(0.8, 0.1, 105.0))
    # Stone: Low energy, different freq
    nova.inject("Stone", Photon(0.2, 0.0, 50.0))
    
    print("Created Concepts: Love, Hate, Joy, Stone")
    
    # 2. Test Logic via Interference
    print("\n[Logic Test: Pulse Interference]")
    
    # Love + Joy (Constructive?)
    coherence_lj = nova.interfere("Love", "Joy")
    print(f"Love + Joy Coherence: {coherence_lj:.2f} (Expected > 0.9)")
    
    # Love + Hate (Destructive?)
    coherence_lh = nova.interfere("Love", "Hate")
    print(f"Love + Hate Coherence: {coherence_lh:.2f} (Expected < 0.1)")
    
    # 3. Test Search via Resonance
    print("\n[Search Test: Frequency Resonance]")
    target = 104.0 # Close to Joy (105) and Love (100)
    result = nova.resonate(target)
    print(f"Searching for Freq {target} -> Resonated with: '{result}'")
    
    print("\nResult: Logic was computed without boolean operators.")
    print("This is the beginning of Flowless Thought.")

if __name__ == "__main__":
    demo_nova_thought()
