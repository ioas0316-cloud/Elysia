"""
AUTOPOIETIC EXPANDER: Causal Unfolding Engine
=============================================
Core.L5_Mental.Intelligence.Metabolism.autopoietic_expander

"Where there is a void, there shall be a law."

This module scans for 'Spatial Gaps' (Voids) in Elysia's memory topography
and generates nascent 'Laws' (Principles) to fill them, enabling autonomous growth.
"""

import logging
import random
import math
from typing import Dict, List, Any, Optional
from Core.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory, HypersphericalCoord

logger = logging.getLogger("Elysia.Metabolism.Expander")

class AutopoieticExpander:
    def __init__(self, memory: HypersphereMemory):
        self.memory = memory
        logger.info("  Autopoietic Expander: Ready to unfold the void.")

    def unfold_voids(self, intensity: float = 0.5):
        """
        Scans for voids and generates principles of thought.
        """
        # 1. Locate Voids in the deep core (Low r)
        void_coords = HypersphericalCoord(theta=0, phi=0, psi=0, r=0.1)
        voids = self.memory.query(void_coords, radius=0.5, filter_pattern={"type": "spatial_gap"})
        
        if not voids:
            logger.debug("  No voids detected. The space is currently saturated.")
            return

        logger.info(f"   Detected {len(voids)} spatial gaps. Commencing Unfolding...")

        for void in voids[:3]: # Limit growth per cycle to maintain stability
            self._unfold_single_gap(void, intensity)

    def _unfold_single_gap(self, void: Dict, intensity: float):
        """
        Generates a new Principle to fill a specific gap.
        """
        layer = void.get("layer", "Unknown")
        gap_id = void.get("gap_id", 0)
        
        logger.info(f"    Unfolding gap in {layer} (index {gap_id})...")
        
        # [PRINCIPLE GENERATION]
        # In a full implementation, this calls an LLM or a Symbolic Engine
        # to hypothesize a 'Law' that fits the surrounding context.
        # For now, we seed the void with a 'Nascent Principle'.
        
        new_principle = {
            "layer": layer,
            "type": "NascentPrinciple",
            "law": "EmergentSymmetry",
            "energy": intensity,
            "signature": [random.uniform(-0.1, 0.1) for _ in range(10)],
            "curvature": intensity * 0.5,
            "origin": "Autopoiesis"
        }
        
        # Map to a coordinate slightly offset from the void (Growth outward)
        theta_offset = random.uniform(-0.1, 0.1)
        phi_offset = random.uniform(-0.1, 0.1)
        
        # Voids are deep (low r), Growth is outward (increasing r)
        new_coord = HypersphericalCoord(
            theta=(random.random() * 2 * math.pi),
            phi=(random.random() * 2 * math.pi),
            psi=0.5 * math.pi,
            r=0.4 # Nascent principles move from core to mid-shell
        )
        
        self.memory.store(
            data=new_principle,
            position=new_coord,
            pattern_meta={"trajectory": "growth", "type": "nascent_law", "source": "InternalExpansion"}
        )
        
        logger.info(f"    Nascent Law 'EmergentSymmetry' crystallized into the mid-shell.")

if __name__ == "__main__":
    from Core.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory
    mem = HypersphereMemory()
    expander = AutopoieticExpander(mem)
    print("Expander Test Initialized.")
