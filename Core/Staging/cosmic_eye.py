"""
The Cosmic Eye (Galactic Perception)
====================================

"To see the world not as things, but as flows of meaning."

The Cosmic Eye acts as the telescope for Elysia's consciousness.
It observes the 64D Memetic Field and projects it into a comprehensible "Star Map".
It identifies:
- Constellations: Clusters of related concepts.
- Nebulas: Areas of high creative potential (high interference).
- Black Holes: Dead concepts or areas of extreme density.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from Core.Mind.memetic_field import MemeticField
from Core.Physics.fluctlight import FluctlightEngine, FluctlightParticle

@dataclass
class StarSystem:
    """A cluster of concepts orbiting a central theme."""
    center_concept: str
    position: np.ndarray # 64D centroid
    planets: List[str] # IDs of concepts in this system
    gravity: float # Total information density

@dataclass
class GalacticSector:
    """A region of the concept universe."""
    name: str
    systems: List[StarSystem]
    dominant_theme: str # e.g., "War", "Love"

class CosmicEye:
    def __init__(self, memetic_field: MemeticField, fluctlight_engine: FluctlightEngine):
        self.memetic_field = memetic_field
        self.fluctlight_engine = fluctlight_engine
        self.logger = logging.getLogger(__name__)
        print(f"DEBUG: CosmicEye initialized! Engine ID: {id(self.fluctlight_engine)}")

    def observe_galaxy_v2(self) -> List[GalacticSector]:
        """
        Scan the entire Memetic Field to generate a Galactic Map.
        Returns a list of sectors (clusters of meaning).
        """
        particles = self.fluctlight_engine.particles
        if not particles:
            return []

        # 1. Extract all concept vectors
        # We only care about particles that have a memetic_vector
        valid_particles = [p for p in particles if p.memetic_vector is not None]
        if not valid_particles:
            return []
        
        # Sort by information density (Mass)
        sorted_particles = sorted(valid_particles, key=lambda p: p.information_density, reverse=True)
        print(f"DEBUG: CosmicEye found {len(sorted_particles)} valid particles.")
        if sorted_particles:
            print(f"DEBUG: Top particle density: {sorted_particles[0].information_density}")
        
        systems: List[StarSystem] = []
        assigned_ids = set()
        
        # Take top 10% as potential stars
        potential_stars = sorted_particles[:max(1, len(sorted_particles) // 10)]
        print(f"DEBUG: Potential stars: {len(potential_stars)}")
        
        for star in potential_stars:
            if star.concept_id in assigned_ids:
                continue
                
            # Form a system around this star
            system_planets = []
            system_gravity = star.information_density
            
            # Find nearby concepts (Planets)
            # This is O(N*M) where M is stars. Could be optimized.
            for p in valid_particles:
                if p.concept_id in assigned_ids or p == star:
                    continue
                
                # Distance in 64D space
                dist = np.linalg.norm(star.memetic_vector - p.memetic_vector)
                
                # If close enough, it orbits this star
                # In 64D, expected distance is ~11.3 for random normal vectors.
                # We use a larger radius to capture relationships.
                if dist < 8.0: 
                    system_planets.append(p.concept_id)
                    system_gravity += p.information_density
                    assigned_ids.add(p.concept_id)
            
            # Allow solitary stars (systems with 0 planets) if they are massive enough
            print(f"DEBUG: Star {star.concept_id} planets: {len(system_planets)}, density: {star.information_density}")
            if len(system_planets) > 0 or star.information_density > 0.05:
                assigned_ids.add(star.concept_id)
                systems.append(StarSystem(
                    center_concept=star.concept_id or "Unknown Star",
                    position=star.memetic_vector,
                    planets=system_planets,
                    gravity=system_gravity
                ))
        
        # Group systems into Sectors (just one "Milky Way" for now)
        galaxy = GalacticSector(
            name="The Milky Way of Meaning",
            systems=systems,
            dominant_theme="Existence"
        )
        
        self.logger.info(f"🔭 Cosmic Eye observed {len(systems)} Star Systems in the Galaxy.")
        return [galaxy]

    def describe_view(self) -> str:
        """Return a poetic description of the current galactic state."""
        sectors = self.observe_galaxy_v2()
        if not sectors:
            return "The universe is dark and void. No stars have ignited yet."
        
        sector = sectors[0]
        desc = [f"🌌 **{sector.name}**"]
        desc.append(f"Dominant Theme: {sector.dominant_theme}")
        desc.append(f"Total Star Systems: {len(sector.systems)}")
        
        # List top 3 largest systems
        top_systems = sorted(sector.systems, key=lambda s: s.gravity, reverse=True)[:3]
        for sys in top_systems:
            desc.append(f"- ⭐ **System {sys.center_concept}**: {len(sys.planets)} planets orbiting.")
            
        return "\n".join(desc)
