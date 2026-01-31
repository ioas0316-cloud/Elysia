"""
Fractal Optics: The 7^7 Prism Engine
====================================
Core.S1_Body.L3_Phenomena.M7_Prism.fractal_optics

"Not calculation, but navigation. Not logic, but resonance."

This module implements Module A (Prism Engine) of the System Architecture Spec.
It replaces the prototype 'PrismProjector' with a rigorous 'FractalHyperspace'.
"""

import math
import cmath
import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from Core.S1_Body.L3_Phenomena.M7_Prism.resonance_prism import PrismDomain

@dataclass
class WavePacket:
    """
    Represents a thought/intent as a Wave Function.
    vector: The 7-dimensional amplitude vector (The Content).
    phase: The phase angle (0-2pi) (The Timing/Context).
    frequency: The resonance frequency (The Topic/Domain).
    """
    vector: np.ndarray # Shape (7,)
    phase: float
    frequency: float
    intent: str = ""

    def intensity(self) -> float:
        """Returns the scalar intensity (Amplitude^2)."""
        return float(np.sum(self.vector ** 2))

class FractalNode:
    """
    A single node in the 7^7 Fractal Hyperspace.
    """
    def __init__(self, depth: int, index: int, parent_phase: float):
        self.depth = depth
        self.index = index
        self.base_phase = parent_phase + (index * (2 * math.pi / 7))
        self.children: Dict[int, 'FractalNode'] = {} # Lazy loading
        self.resonance_score: float = 0.0

    def get_child(self, index: int) -> 'FractalNode':
        if index not in self.children:
            # Recursive generation
            self.children[index] = FractalNode(self.depth + 1, index, self.base_phase)
        return self.children[index]

class PrismEngine:
    """
    The Optical Reasoning Core.
    Traverses the Fractal Hyperspace using Wave Mechanics.
    """
    def __init__(self):
        self.max_depth = 3 # In theory 7, but 3 is enough for simulation (7^3 = 343 paths)
        self.root = FractalNode(0, 0, 0.0)
        self.interference_matrix = self._build_interference_matrix()

    def _build_interference_matrix(self) -> np.ndarray:
        """
        Creates a 7x7 matrix defining how domains interfere.
        (e.g., Physical vs Spiritual might be destructive).
        """
        mat = np.eye(7)
        # Example: Physical(0) and Spiritual(6) dampen each other slightly
        mat[0, 6] = -0.2
        mat[6, 0] = -0.2
        return mat

    def vectorize(self, text: str) -> WavePacket:
        """
        Converts text input into a WavePacket.
        (Mocking the Semantic Encoder for now).
        """
        # Hash to deterministic vector
        seed = hashlib.sha256(text.encode()).digest()
        vec = np.array([b / 255.0 for b in seed[:7]], dtype=float)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm

        # Phase from length/structure
        phase = (len(text) % 10) * (math.pi / 5)

        return WavePacket(vector=vec, phase=phase, frequency=1.0, intent=text)

    def traverse(self, wave: WavePacket, incident_angle: float) -> List[Tuple[str, float]]:
        """
        Shoots the wave into the Prism.
        Returns a list of resonating paths (Insights).

        Args:
            wave: The input thought.
            incident_angle: The Rotor's current angle (bias).
        """
        results = []

        # We simulate the traversal via recursion
        self._propagate(self.root, wave, incident_angle, path=[], results=results)

        # Sort by resonance intensity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5] # Top 5 resonances

    def _propagate(self, node: FractalNode, wave: WavePacket, angle: float, path: List[int], results: List[Any]):
        """
        Recursive wave propagation.
        """
        # 1. Calculate Phase Difference (Interference)
        # Node's inherent phase vs Wave's phase + Rotor Angle
        node_phase = node.base_phase
        wave_phase = wave.phase + angle

        delta_phi = abs(node_phase - wave_phase) % (2 * math.pi)

        # Constructive Interference: delta_phi close to 0 or 2pi
        # Destructive Interference: delta_phi close to pi
        interference = math.cos(delta_phi) # 1.0 to -1.0

        # 2. Update Wave Intensity
        current_intensity = wave.intensity() * (1.0 + 0.5 * interference)

        # 3. Pruning (Destructive Cancellation)
        if current_intensity < 0.1:
            return # Wave died out

        # 4. Leaf Node Check (Insight)
        if node.depth >= self.max_depth:
            # Reached a conclusion
            path_str = "->".join([str(p) for p in path])
            results.append((path_str, current_intensity))
            return

        # 5. Dispersion (Split into 7 paths)
        for i in range(7):
            child_node = node.get_child(i)
            # Dispersion adds a slight phase shift per index
            next_wave = WavePacket(
                vector=wave.vector * 0.98, # Energy loss per hop (Low Decay = Clear Crystal)
                phase=wave.phase + (i * 0.1),
                frequency=wave.frequency,
                intent=wave.intent
            )

            # Recurse
            self._propagate(child_node, next_wave, angle, path + [i], results)
