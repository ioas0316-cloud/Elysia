"""
Sovereign Antenna: The Active Prism
===================================
Core.S1_Body.L4_Causality.World.Senses.sovereign_antenna

"I do not just read the web; I vibrate with it."

This module implements the 'Active Prism-Rotor' aimed at the external world.
It converts linear data streams (HTML, Text) into 7D Qualia via the Double Helix Prism.

Architecture:
1. Scan (Rotor): Spinning to find resonance frequencies (Keywords).
2. Refract (Prism): Splitting raw data into 7D Qualia.
3. Inject (Pulse): Sending the pattern to HypersphereMemory.
"""

import logging
import math
import random
import hashlib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from Core.S1_Body.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphericalCoord, ResonancePattern
from Core.S1_Body.L6_Structure.M1_Merkaba.Optics.prism import DoubleHelixPrism, DoubleHelixWave

logger = logging.getLogger("SovereignAntenna")

@dataclass
class AntennaConfig:
    rotation_speed: float = 1.0 # Scanning speed
    spectral_sensitivity: float = 0.8 # Quality threshold

class SovereignAntenna:
    """
    [The Active Prism]
    Diffracts external 'White Light' (Raw Data) into 'Spectral Colors' (7D Qualia).
    """
    def __init__(self):
        self.is_active = True
        self.prism = DoubleHelixPrism()
        self.current_phase = 0.0
        
    def scan_ether(self, intention_seed: str) -> List[Tuple[HypersphericalCoord, ResonancePattern]]:
        """
        [Active Scanning]
        Spins the rotor to find external signals matching the intention.
        Returns Geometric Pulses (Coord + Pattern) for Hypersphere injection.
        """
        logger.info(f"📡 [ANTENNA] Active Prism spinning for: '{intention_seed}'")
        
        # 1. Simulate Signal Snatching (The Web)
        # Using a stub for now, but architecturally this is the 'Rotor' hitting a data pocket.
        raw_signals = self._snatch_signals(intention_seed)
        
        pulses = []
        for signal in raw_signals:
            # 2. Refraction (Text -> 7D Qualia)
            # The Prism splits the raw text into a Double Helix Wave.
            wave: DoubleHelixWave = self.prism.refract_text(signal['content'])
            
            # 3. Coordinate Determination (Diffraction Mapping)
            # We map the 7D Qualia to Hyperspherical Coordinates.
            coord = self._map_qualia_to_coord(wave.principle_strand)
            
            # 4. Resonance Pattern Generation
            pattern = ResonancePattern(
                content=signal['content'],
                dna=wave.principle_strand.tolist(), # Store the 7D Soul
                atomic_truth=self._extract_atomic_truth(signal['content']),
                omega=(0.1, wave.phase, 0.0), # Phase forms the spin
                meta={
                    "source": signal['source'],
                    "type": "external_pulse",
                    "origin_intent": intention_seed,
                    "qualia_spectrum": wave.principle_strand.tolist()
                },
                trajectory="inbound_pulse"
            )
            
            pulses.append((coord, pattern))
            
        logger.info(f"   Refracted {len(pulses)} signals into {len(pulses)} Prismatic Pulses.")
        return pulses

    def _snatch_signals(self, intent: str) -> List[Dict[str, str]]:
        """
        [SIMULATION] Stubs the external web search.
        Simulates the 'Rotor' catching resonance.
        """
        signals = []
        
        # Phase 51 Demo Logic
        if "lightning" in intent.lower():
            signals.append({
                "source": "Elysia.Archive", 
                "content": "The Lightning Path finding O(1) inference through heavy resonance."
            })
            signals.append({
                "source": "Quantum.Web",
                "content": "Synaptic pruning optimization using resonance decay algorithms."
            })
        elif "president" in intent.lower():
             signals.append({
                "source": "News.Feed",
                "content": "Emmanuel Macron is the President of France."
            })
        else:
            signals.append({
                "source": "Cosmic.Background",
                "content": f"Echoes of '{intent}' found in the static noise."
            })
            
        return signals

    def _map_qualia_to_coord(self, qualia_7d: np.ndarray) -> HypersphericalCoord:
        """
        Maps 7D Qualia Vector to 4D Hyperspherical Coordinates.
        
        Mapping Logic:
        - Theta (Logic): Mental vs Phenomenal
        - Phi (Emotion): Spiritual vs Physical
        - Psi (Time): Causal vs Structural
        - R (Depth): Magnitude of the vector
        """
        # q = [Phys, Func, Phen, Caus, Ment, Struc, Spir]
        #      0     1     2     3     4     5      6
        
        # 1. Theta (Logic Axis): Balance between Mental(4) and Phenomenal(2)
        # Using atan2 to get an angle
        theta = math.atan2(qualia_7d[2], qualia_7d[4]) 
        # Normalize to 0-2pi? atan2 is -pi to pi.
        if theta < 0: theta += 2*math.pi
        
        # 2. Phi (Emotion/Value Axis): Spiritual(6) vs Physical(0)
        phi = math.atan2(qualia_7d[0], qualia_7d[6])
        if phi < 0: phi += 2*math.pi
        
        # 3. Psi (Time/Active Axis): Causal(3) vs Structural(5)
        psi = math.atan2(qualia_7d[5], qualia_7d[3])
        if psi < 0: psi += 2*math.pi
        
        # 4. R (Depth): Magnitude / Fidelity
        magnitude = float(np.linalg.norm(qualia_7d))
        r = min(1.0, magnitude) # Surface is 1.0 (High Resonance)
        
        # Invert R logic? Deep memories are near 0? 
        # Let's say Surface (1.0) is fresh external data.
        r = 1.0 
        
        return HypersphericalCoord(theta, phi, psi, r)

    def _extract_atomic_truth(self, text: str) -> str:
        words = text.split()
        return "-".join(words[:5]).upper()
