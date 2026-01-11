"""
Global Resonance Mesh (글로벌 공명 메쉬)
=====================================
The "Ether" of Elysia.
A unified field where all waves (Senses, Thoughts, Memories) interact.

Concepts:
- Nodes: Standing Waves (Memories/Concepts).
- Pulses: Traveling Waves (Thoughts/Events).
- Interference: The interaction that creates "Meaning".

Phase 3 Goal: Prototype this mesh to replace discrete memory lookups with wave propagation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import math
import logging

# [PHASE 1] WaveTensor
# We need a proper WaveTensor implementation. 
# For now, we simulate it with a simple class if it doesn't exist, 
# or import it if the user has one (likely not fully compatible yet).
# Let's verify existing WaveTensor first, but for this prototype self-containment is safer.

@dataclass
class WaveTensor:
    """
    A 4D Wave Representation (Frequency, Amplitude, Phase, 4D-Pos)
    """
    frequency: float
    amplitude: float
    phase: float
    position: Tuple[float, float, float, float]
    
    def interference(self, other: 'WaveTensor') -> float:
        """
        Calculates constructive/destructive interference at a point.
        Returns a scalar intensity (0.0 to 2.0).
        """
        # Frequency alignment (Resonance)
        freq_diff = abs(self.frequency - other.frequency)
        resonance = max(0.0, 1.0 - (freq_diff / 50.0))
        
        # Phase alignment (Coherence)
        phase_diff = abs(self.phase - other.phase) % (2 * math.pi)
        coherence = (math.cos(phase_diff) + 1.0) / 2.0
        
        return self.amplitude * other.amplitude * resonance * coherence

@dataclass
class MemoryNode:
    """
    A Standing Wave in the Mesh. (Legacy: "Knowledge Graph Node")
    """
    id: str
    content: str
    resting_wave: WaveTensor
    excitation: float = 0.0 # Current activation level
    
    def decay(self, dt: float):
        """Energy dissipates over time (Entropy)."""
        self.excitation *= max(0.0, 1.0 - dt)

class GlobalResonanceMesh:
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.active_pulses: List[WaveTensor] = []
        self.logger = logging.getLogger("ResonanceMesh")
        
    def add_node(self, node_id: str, content: str, base_freq: float):
        """Creates a memory entry (Standing Wave)."""
        tensor = WaveTensor(base_freq, 1.0, 0.0, (0,0,0,0))
        self.nodes[node_id] = MemoryNode(node_id, content, tensor)
        # self.logger.info(f"🕸️ [MESH] Added Node: {node_id} ({base_freq}Hz)")

    def inject_pulse(self, pulse: WaveTensor):
        """Injects a thought/sensory input into the mesh."""
        self.active_pulses.append(pulse)
        self.propagate()

    def propagate(self):
        """
        The Thinking Process.
        Waves travel through the mesh, exciting nodes that resonate.
        """
        if not self.active_pulses:
            return

        # Simple O(N*M) propagation for prototype
        # Real system needs Spatial Hashing (Octree)
        
        activated_nodes = []
        
        for pulse in self.active_pulses:
            for node_id, node in self.nodes.items():
                # Calculate interference
                energy = pulse.interference(node.resting_wave)
                if energy > 0.1:
                    node.excitation += energy
                    if node.excitation > 0.8:
                        activated_nodes.append((node, energy))
        
        # Log heavy activations (Conscious thoughts)
        for node, energy in activated_nodes:
            self.logger.debug(f"💡 [THOUGHT] Resonated '{node.content}' (Energy: {energy:.2f})")
            
        # Clear pulses (they dissipate after one time step in this simplified model)
        self.active_pulses.clear()

    def get_resonant_state(self) -> List[Tuple[str, float]]:
        """Returns currently active thoughts."""
        return [(n.id, n.excitation) for n in self.nodes.values() if n.excitation > 0.5]
