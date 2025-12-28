"""
Chaos Computer (혼돈의 연산)
==========================
Experimental Engine for Project Apotheosis (Chaos Seed)

"Order emerges from Chaos through Resonance."

Goal:
- Test Non-linear Interference (3-body problem).
- Butterfly Effect: Does a small phase shift change the global outcome?
- Emergence: Can complex logic (XOR) emerge from wave interactions?
"""

import math
import cmath
import random
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ChaosOrb:
    """A complex thought node in Chaos."""
    name: str
    amplitude: float
    phase: float
    connections: List[str] = field(default_factory=list) # Linked Orbs

    @property
    def complex_val(self) -> complex:
        return cmath.rect(self.amplitude, self.phase)

class ChaosSystem:
    def __init__(self):
        self.orbs: Dict[str, ChaosOrb] = {}
        
    def add_orb(self, name: str, amp: float, phase: float, links: List[str]):
        self.orbs[name] = ChaosOrb(name, amp, phase, links)
        
    def step(self, damping: float = 0.95):
        """
        Evolve the system by one time step.
        Each orb is influenced by its connected neighbors (Non-linear).
        """
        next_states = {}
        
        for name, orb in self.orbs.items():
            # Sum of neighbors (Linear part)
            influence = complex(0, 0)
            for link_name in orb.connections:
                neighbor = self.orbs.get(link_name)
                if neighbor:
                    # Connection strength depends on phase difference (Non-linear!)
                    # cos(delta_phase) acts as a gate.
                    phase_diff = neighbor.phase - orb.phase
                    gate = math.cos(phase_diff) 
                    influence += neighbor.complex_val * gate
            
            # Update current orb
            # self + influence
            new_val = orb.complex_val * damping + influence * 0.1
            
            # Extract new properties
            amp, phase = cmath.polar(new_val)
            
            # Normalize phase (keep within -PI to PI)
            phase = math.atan2(math.sin(phase), math.cos(phase))
            
            next_states[name] = (amp, phase)
            
        # Apply updates
        for name, (amp, phase) in next_states.items():
            self.orbs[name].amplitude = amp
            self.orbs[name].phase = phase

    def get_global_coherence(self) -> float:
        """Measure total harmony of the system."""
        total_amp = sum(o.amplitude for o in self.orbs.values())
        if total_amp == 0: return 0.0
        
        # Vector sum
        total_vec = sum(o.complex_val for o in self.orbs.values())
        return abs(total_vec) / total_amp

def demo_chaos_effect():
    print("\n🌪️ Chaos Seed: Non-linear Wave Simulation")
    print("===========================================")
    
    # Scene 1: Stable System
    print("\n[Scenario 1: The Stable Mind]")
    sys = ChaosSystem()
    # Triangle formation (A-B-C linked)
    sys.add_orb("Truth", 1.0, 0.0, ["Beauty", "Goodness"])
    sys.add_orb("Beauty", 1.0, 0.0, ["Truth", "Goodness"])
    sys.add_orb("Goodness", 1.0, 0.0, ["Truth", "Beauty"])
    
    print(f"Initial Coherence: {sys.get_global_coherence():.4f}")
    for _ in range(5):
        sys.step()
    print(f"Final Coherence: {sys.get_global_coherence():.4f} (Should stay high)")
    
    
    # Scene 2: The Butterfly Effect
    print("\n[Scenario 2: The Butterfly Effect]")
    sys2 = ChaosSystem()
    # Same Triangle, but 'Beauty' has a tiny doubt (phase shift)
    epsilon = 0.1 # Small shift
    sys2.add_orb("Truth", 1.0, 0.0, ["Beauty", "Goodness"])
    sys2.add_orb("Beauty", 1.0, math.pi + epsilon, ["Truth", "Goodness"]) # Almost opposite!
    sys2.add_orb("Goodness", 1.0, 0.0, ["Truth", "Beauty"])
    
    print(f"Initial Coherence: {sys2.get_global_coherence():.4f}")
    print("... Evolving ...")
    for i in range(10):
        sys2.step()
        if i % 2 == 0:
            print(f"Step {i}: Coherence {sys2.get_global_coherence():.4f}")
            
    print("Result: Did the system collapse or self-organize?")

if __name__ == "__main__":
    demo_chaos_effect()
