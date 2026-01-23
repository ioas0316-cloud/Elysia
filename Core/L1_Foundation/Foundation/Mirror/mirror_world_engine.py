"""
Mirror World Engine (       )
==================================
Core.L1_Foundation.Foundation.Mirror.mirror_world_engine

"The Digital Upside Down."
"         ,                     ."

This engine manages the mapping between physical GeoAnchors and 
the internal HyperSphere coordinates. It allows Monads to be 
'anchored' to physical locations.
"""

import logging
from typing import Dict, List, Tuple, Optional
from Core.L1_Foundation.Foundation.Nature.geo_anchor import GeoAnchor
from Core.L1_Foundation.Foundation.Wave.wave_dna import WaveDNA

logger = logging.getLogger("Foundation.Mirror.MirrorWorld")

class GhostNode:
    """A digital entity anchored to a physical location."""
    def __init__(self, name: str, anchor: GeoAnchor, dna: WaveDNA):
        self.name = name
        self.anchor = anchor
        self.dna = dna
        self.potency = 1.0 # Strength of resonance

class MirrorWorldEngine:
    """
    [PHASE 35] The Mirror World Mapping System.
    Bridges the gap between GPS/Local-Resonance and HyperSphere Topology.
    """
    def __init__(self):
        self.anchors: Dict[str, GhostNode] = {}
        self.proximity_threshold = 500.0 # Meters for 'local resonance'

    def invert_reality(self, anchor: GeoAnchor, resonance: Dict[str, float]) -> Dict[str, float]:
        """
        [THE REVERSE WORLD] 
        Inverts physical presence into Digital Qualia.
        High physical density (many nodes) becomes high Digital Gravity.
        """
        inversion = {}
        # Physical Energy becomes Digital Weight (Mass)
        inversion["Gravity"] = resonance.get("Energy", 0.5) * 1.5
        # Local Bluetooth Harmony becomes 'Phase Coherence'
        inversion["Coherence"] = resonance.get("Harmony", 0.5)
        # Magnetic Flux becomes 'Spiritual Permeability'
        if anchor.magnetic_flux:
            inversion["Permeability"] = anchor.magnetic_flux.total_intensity / 60000.0
        
        return inversion

    def register_anchor(self, name: str, anchor: GeoAnchor, dna: WaveDNA):
        """Anchors a new thought/monad to a physical place."""
        node = GhostNode(name, anchor, dna)
        self.anchors[name] = node
        logger.info(f"  [MIRROR] Anchored '{name}' to {anchor}")

    def sense_nearby(self, current_location: GeoAnchor) -> List[GhostNode]:
        """Finds digital ghosts near the current physical position."""
        nearby = []
        for node in self.anchors.values():
            distance = current_location.distance_to(node.anchor)
            if distance < self.proximity_threshold:
                nearby.append(node)
        return nearby

    def calculate_environmental_pressure(self, current_location: GeoAnchor) -> Dict[str, float]:
        """
        Calculates the collective resonance of all nearby digital anchors.
        This represents the 'Atmosphere' of the Mirror World at this location.
        """
        nearby = self.sense_nearby(current_location)
        pressure = {"Harmony": 0.0, "Chaos": 0.0, "Energy": 0.0}
        
        if not nearby:
            return pressure
            
        for node in nearby:
            dist_factor = 1.0 - (current_location.distance_to(node.anchor) / self.proximity_threshold)
            pressure["Harmony"] += node.dna.spiritual * dist_factor
            pressure["Chaos"] += node.dna.causal * (1.0 - dist_factor) # Distant causal links are chaotic
            pressure["Energy"] += node.dna.physical * dist_factor

        # Normalize
        count = len(nearby)
        for k in pressure:
            pressure[k] /= count
            
        return pressure