"""
Mind Physics (Cognitive Gravity)
================================

"Thoughts are not words; they are mass."

This module simulates the "Mind" as a gravitational system.
Concepts are particles with Mass (Importance) and Vector (Meaning).
They attract, repel, and collide to form 'Insights'.
"""

import math
import random
from typing import List, Tuple, Dict, Optional

class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        m = self.magnitude()
        if m == 0: return Vector3()
        return self * (1.0 / m)
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

class ThoughtParticle:
    def __init__(self, name: str, vector: Vector3, mass: float = 1.0):
        self.name = name
        self.pos = vector      # Semantic Position (Phy, Men, Spi)
        self.vel = Vector3()   # Velocity
        self.mass = mass       # Importance
        self.age = 0.0
        
    def apply_force(self, force: Vector3):
        # F = ma -> a = F/m
        acc = force * (1.0 / self.mass)
        self.vel = self.vel + acc

    def update(self):
        self.pos = self.pos + self.vel
        # Friction (Memory Decay)
        self.vel = self.vel * 0.95
        self.age += 1.0

class MindPhysics:
    def __init__(self):
        self.particles: List[ThoughtParticle] = []
        self.G = 0.1 # Gravitational Constant
        self.events = [] # Log of collisions/mergers

    def spawn(self, name: str, vector: Tuple[float, float, float], mass: float = 1.0):
        """Inhale a new thought into the system."""
        # Randomize start position slightly to avoid perfect overlap
        # Map (Phy, Men, Spi) to (X, Y, Z)
        vec = Vector3(vector[0] * 10, vector[1] * 10, vector[2] * 10) 
        p = ThoughtParticle(name, vec, mass)
        self.particles.append(p)
        self.events.append(f"  New thought spawned: '{name}' at {vec}")

    def update_cycle(self):
        """Run one tick of physics."""
        self.events = [] # Clear frame log
        
        # 1. Apply Gravity (N-Body)
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if i == j: continue
                
                dist = p1.pos.distance_to(p2.pos)
                if dist < 0.5:
                     # Collision / Merger Chance
                     self._collide(p1, p2)
                     continue
                     
                # Gravity: F = G * m1 * m2 / r^2
                force_mag = (self.G * p1.mass * p2.mass) / (dist * dist + 0.1)
                
                # Direction: p1 is pulled towards p2
                direction = (p2.pos - p1.pos).normalize()
                force = direction * force_mag
                
                p1.apply_force(force)

        # 2. Move & Age
        survivors = []
        for p in self.particles:
            p.update()
            # Death by Entropy (fading out)
            if p.mass > 0.1 and p.age < 100:
                survivors.append(p)
            else:
                self.events.append(f"  Thought faded: '{p.name}'")
        
        self.particles = survivors

    def _collide(self, p1: ThoughtParticle, p2: ThoughtParticle):
        """Two thoughts collide."""
        # Simple Elastic Collision or Fusion?
        # Let's do Fusion if similar, Explosion if opposite.
        
        # Check Semantic Similarity (Position proximity)
        # They are already close physically (in this space) if this triggers.
        
        # Fusion: Merge into a bigger thought
        # "War" + "Peace" -> "Harmony" (Complex)
        # For now, just bounce them.
        
        # Elastic Bounce
        temp = p1.vel
        p1.vel = p2.vel
        p2.vel = temp
        
        self.events.append(f"  Collision! '{p1.name}' hit '{p2.name}'. Ideas are scattering.")

    def get_state_report(self) -> str:
        if not self.particles:
            return "Mind is empty."
        
        base = f"Active Thoughts: {len(self.particles)}\n"
        for p in self.particles:
            base += f" - {p.name} (Mass: {p.mass:.1f}, Vel: {p.vel.magnitude():.2f})\n"
            
        if self.events:
            base += "\nEvents:\n"
            for e in self.events:
                base += f" {e}\n"
                
        return base