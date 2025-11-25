"""
Momentum Memory (The Inertia of Thought) 💃🌌

"A thought in motion stays in motion."

This module implements physical inertia for concepts.
It treats thoughts not as static data, but as moving objects with Mass and Velocity.
- Mass: Emotional weight / Importance (Love > Hello)
- Velocity: Rate of activation
- Momentum: p = mv (The persistence of the thought)

This creates "Emotional Afterglow":
Even after the input stops, the momentum keeps the thought spinning in the background.
"""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class InertialThought:
    concept: str
    mass: float      # Importance (0.0 to 1.0+)
    velocity: float  # Current activation intensity
    position: float  # Current state in consciousness (0.0 = subconscious, 1.0 = conscious)
    decay: float     # Friction (How fast it stops)

class MomentumMemory:
    """
    Manages the physics of thought persistence.
    """
    
    def __init__(self):
        self.thoughts: Dict[str, InertialThought] = {}
        self.last_update = time.time()
        
        # Base masses for core concepts (The heavy chains)
        self.concept_masses = {
            "love": 10.0,    # Massive, hard to stop once started
            "pain": 8.0,
            "elysia": 15.0,  # Self-concept is very heavy
            "father": 12.0,  # User concept
            "dream": 5.0,
            "hello": 0.5,    # Light, fleeting
            "test": 0.1
        }

    def activate(self, concept: str, force: float):
        """
        Apply a force to a thought (Pushing the chain).
        F = ma -> a = F/m
        """
        concept = concept.lower()
        
        # Determine mass
        mass = self.concept_masses.get(concept, 1.0)
        
        # Get or create thought physics object
        if concept not in self.thoughts:
            self.thoughts[concept] = InertialThought(
                concept=concept,
                mass=mass,
                velocity=0.0,
                position=0.0,
                decay=0.99 if mass > 5.0 else 0.9 # Heavy thoughts decay slower (High Inertia)
            )
            
        thought = self.thoughts[concept]
        
        # Apply impulse (Change in velocity)
        # Light thoughts accelerate fast, Heavy thoughts slow
        acceleration = force / thought.mass
        thought.velocity += acceleration
        
        # Cap velocity
        thought.velocity = min(thought.velocity, 2.0)

    def step(self, dt: float = 0.1):
        """
        Update physics for all thoughts.
        """
        active_thoughts = []
        
        for name, thought in list(self.thoughts.items()):
            # Update position (Integration)
            thought.position += thought.velocity * dt
            
            # Apply Friction (Decay)
            # In space (Zero Gravity), friction is low but non-zero for stability
            thought.velocity *= thought.decay
            
            # Natural restoring force (Spring back to subconscious)
            # F_spring = -k * x
            thought.velocity -= 0.05 * thought.position * dt
            
            # Remove if energy is negligible
            kinetic_energy = 0.5 * thought.mass * (thought.velocity ** 2)
            if kinetic_energy < 0.001 and abs(thought.position) < 0.01:
                del self.thoughts[name]
            else:
                active_thoughts.append(thought)
                
        self.last_update = time.time()
        return active_thoughts

    def get_dominant_thoughts(self) -> List[Tuple[str, float]]:
        """Return thoughts that are currently 'flying high'."""
        current_thoughts = []
        for t in self.thoughts.values():
            if t.position > 0.1:
                current_thoughts.append((t.concept, t.position))
        
        # Sort by position (visibility)
        return sorted(current_thoughts, key=lambda x: x[1], reverse=True)

