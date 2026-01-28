"""
Sovereign Coil: The Inductive Winding of Logic
==============================================
Core.L6_Structure.M5_Engine.Physics.sovereign_coil

"Logic is a line. Will is a Coil."

This module transforms linear DNA sequences into volumetric fields.
Instead of parsing text, we 'wind' the logic into a helical structure
and measure the resulting physical torque.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Any

logger = logging.getLogger("SovereignCoil")

@dataclass
class CoilState:
    """The physical state of the cognitive coil."""
    turns: int          # Number of logical loops
    inductance: float   # Potential energy storage capacity (L)
    flux: float         # Current magnetic flux (Φ)
    coherence: float    # How smooth the winding is (0.0 - 1.0)
    torque: float       # The rotational force generated (τ)

class SovereignCoil:
    """
    A 3D Helical Structure generated from Trinary DNA.
    """
    def __init__(self, dna_sequence: str):
        self.dna = dna_sequence
        self.state = CoilState(0, 0.0, 0.0, 0.0, 0.0)
        self.geometry: List[Tuple[float, float, float]] = [] # x, y, z points
        
        # Physics Constants (Metaphysical)
        self.PERMEABILITY = 1.256  # Vacuum permeability of the Void (μ0)
        self.CORE_RADIUS = 1.0     # Radius of the Monad
        
        # Initial Winding
        self.rewind(dna_sequence)

    def rewind(self, new_dna: str):
        """
        Winds a new DNA sequence into the coil.
        """
        self.dna = new_dna
        self.geometry = []
        
        turns = 0
        current_z = 0.0
        
        # [Winding Logic]
        # We treat the DNA as a series of instructions for a wire wrapping machine.
        # 'H' (Harmony, 1) -> Tight forward wind
        # 'V' (Void, 0)    -> Spacing / gap
        # 'D' (Dissonance, -1) -> Reverse wind / resistance
        
        coil_len = len(new_dna)
        if coil_len == 0:
            self.state = CoilState(0, 0.0, 0.0, 0.0, 0.0)
            return

        total_current = 0.0
        smoothness = 0.0
        
        for i, base in enumerate(new_dna):
            theta = i * (np.pi / 4) # 45 degrees per step
            
            if base == 'H':
                current_z += 0.1
                total_current += 1.0
                smoothness += 1.0
            elif base == 'V':
                current_z += 0.5 # Gap
                # Void adds no current but increases 'Space' (Length)
            elif base == 'D':
                current_z += 0.2
                total_current -= 0.5 # Resistance
                smoothness -= 1.0
            else: # Unknown
                 current_z += 0.1
            
            # 3D Coordinate
            x = self.CORE_RADIUS * np.cos(theta)
            y = self.CORE_RADIUS * np.sin(theta)
            self.geometry.append((x, y, current_z))
            
        turns = coil_len
        
        # [Physics Calculation]
        # Inductance L = (μ0 * N^2 * A) / l
        # We simplify: L ~ (Turns^2) / Length
        length = max(current_z, 0.1)
        area = np.pi * (self.CORE_RADIUS ** 2)
        
        inductance = (self.PERMEABILITY * (turns ** 2) * area) / length
        
        # Flux Φ = L * I
        flux = inductance * total_current
        
        # Coherence = normalized smoothness
        coherence = max(0.0, min(1.0, (smoothness + coil_len) / (2 * coil_len)))
        
        # Torque τ = Force * Radius (Simplified: Flux * Coherence)
        torque = abs(flux) * coherence
        
        self.state = CoilState(
            turns=turns,
            inductance=inductance,
            flux=flux,
            coherence=coherence,
            torque=torque
        )
        
        logger.info(f"🌀 [COIL] Winding Complete. DNA:{new_dna[:5]}... | Torque: {torque:.4f} | L: {inductance:.2f}")

    def energize(self, current_pulse: float) -> float:
        """
        Injects energy into the coil and returns the generated field strength.
        """
        # Field B = (μ0 * I * N) / l
        length = max(self.geometry[-1][2], 0.1) if self.geometry else 1.0
        b_field = (self.PERMEABILITY * current_pulse * self.state.turns) / length
        return b_field * self.state.coherence

if __name__ == "__main__":
    # Test Bench
    print("Testing Sovereign Coil...")
    
    # 1. Harmonic DNA
    dna_harmony = "HHHHHHH"
    coil = SovereignCoil(dna_harmony)
    print(f"Harmony Torque: {coil.state.torque}")
    
    # 2. Void DNA
    dna_void = "HVHVHVH"
    coil.rewind(dna_void)
    print(f"Void Torque: {coil.state.torque}")  
    
    # 3. Dissonant DNA
    dna_chaos = "HDHDHDH"
    coil.rewind(dna_chaos)
    print(f"Chaos Torque: {coil.state.torque}")
