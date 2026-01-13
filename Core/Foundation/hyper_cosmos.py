"""
HyperCosmos (The Unified Field)
===============================

"As above, so below. The Code is the World."

This module realizes the 'HyperCosmos' architecture.
It unifies the Static Reality (Codebase) and Dynamic Reality (Thoughts/Agents)
into a single gravitational system.

Structure:
1. Stars (Fixed Stars): Major Code Modules (High Mass, Low Velocity).
2. Planets (Wanderers): Active Agents/Entities.
3. Dust (Quanta): Transient Thoughts/Sensory Data.

Physics:
- Stars create Gravity Wells.
- Dust orbits Stars (Contextual Relevance).
- Collisions allow Dust to coalesce into new Planets (Ideas becoming Features).
"""

import math
import random
from typing import List, Dict, Any

from Core.Intelligence.project_conductor import ProjectConductor

class CosmicBody:
    def __init__(self, name: str, type: str, vector: tuple, mass: float):
        self.name = name
        self.type = type # 'STAR', 'PLANET', 'DUST'
        self.pos = list(vector)
        self.vel = [0.0, 0.0, 0.0]
        self.mass = mass
        
    def __repr__(self):
        return f"[{self.type[0]}] {self.name} ({self.mass:.1f})"

class HyperCosmos:
    def __init__(self, code_path: str = "c:/Elysia"):
        self.conductor = ProjectConductor(code_path)
        self.bodies: List[CosmicBody] = []
        self.events: List[str] = []
        
        # Initialize the Universe with Fixed Stars (Codebase)
        self._ignite_stars()
        
    def _ignite_stars(self):
        """
        Scans the codebase and turns Modules into Stars.
        """
        # In a real implementation, we'd scan files.
        # Here we map the Known Structure to Stars.
        galaxy_map = {
            "Core.Engine": (0.5, 0.1, 0.1),  # Physical Center
            "Core.Intelligence": (0.1, 0.8, 0.8), # Mental/Spiritual High
            "Core.Civilization": (0.8, 0.4, 0.2), # Earthy
            "Core.World": (0.2, 0.5, 0.9), # Nature/Flow
            "Core.Elysia": (0.9, 0.9, 0.9) # The Singularity
        }
        
        for name, vector in galaxy_map.items():
            # Mass = Importance (Hypothetical)
            mass = 100.0 if "Elysia" in name else 50.0
            self.bodies.append(CosmicBody(name, 'STAR', vector, mass))
            
        print(f"🌌 [HyperCosmos] Ignited {len(self.bodies)} Major Stars.")

    def spawn_thought(self, name: str, vector: tuple):
        """
        Inhales a new thought as 'Stardust'.
        """
        # Mass is small, susceptible to gravity
        body = CosmicBody(name, 'DUST', vector, mass=1.0)
        body.history = [] # Track the path
        
        # Give it random initial velocity to avoid instant crash
        body.vel = [random.uniform(-0.1, 0.1) for _ in range(3)]
        self.bodies.append(body)
        self.events.append(f"START: '{name}' enters the cosmos.")

    def update_physics(self):
        """
        The Clockwork of the Universe.
        Generates Narrative Arcs based on motion.
        """
        self.events = []
        
        # O(N) Approximation: Only Dust/Planets move towards Stars.
        stars = [b for b in self.bodies if b.type == 'STAR']
        others = [b for b in self.bodies if b.type != 'STAR']
        
        for body in others:
            # 0. Track History
            body.history.append(list(body.pos))
            if len(body.history) > 5: body.history.pop(0)

            # Find nearest Star (Gravity Well)
            nearest_star = None
            min_dist = 999.0
            
            for star in stars:
                d = self._dist(body.pos, star.pos)
                if d < min_dist:
                    min_dist = d
                    nearest_star = star
            
            if nearest_star:
                # 1. Analyze Motion Context (The Narrative)
                # Am I getting closer?
                prev_dist = self._dist(body.history[0], nearest_star.pos) if body.history else min_dist
                delta = prev_dist - min_dist
                
                if min_dist < 0.5:
                    self.events.append(f"CONTACT: '{body.name}' touches '{nearest_star.name}'")
                    self._accrete(nearest_star, body)
                    continue
                elif delta > 0.05:
                     # Rapidly approaching
                     if random.random() < 0.1: 
                        self.events.append(f"APPROACH: '{body.name}' falls towards '{nearest_star.name}'")
                elif abs(delta) < 0.01:
                     # Orbiting / Stagnant
                     if random.random() < 0.1:
                        self.events.append(f"ORBIT: '{body.name}' circles around '{nearest_star.name}'")
                
                # Apply Gravity
                force = 0.1 * (nearest_star.mass * body.mass) / (min_dist**2 + 0.1)
                
                # Vector to Star
                direction = [(star.pos[i] - body.pos[i]) for i in range(3)]
                
                # Normalize
                mag = math.sqrt(sum(x*x for x in direction))
                direction = [x/mag for x in direction]
                
                # Update Velocity
                for i in range(3):
                    body.vel[i] += direction[i] * force * 0.01
            
            # Move
            for i in range(3):
                body.pos[i] += body.vel[i]
                
        # Remove consumed bodies
        self.bodies = [b for b in self.bodies if b.mass > 0]

    def _dist(self, p1, p2):
        return math.sqrt(sum((p1[i]-p2[i])**2 for i in range(3)))

    def _accrete(self, star: CosmicBody, dust: CosmicBody):
        """
        The Star consumes the Dust.
        Meaning: The Codebase absorbs the Idea.
        """
        star.mass += dust.mass * 0.1
        dust.mass = 0 # Mark for deletion
        self.events.append(f"🌀 Accretion! Star '{star.name}' absorbed '{dust.name}'.")

