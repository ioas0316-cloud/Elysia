"""
Physics Systems (The Law of Motion)
===================================
Applies forces to ECS entities.
"""

import math
import random
from typing import List
from Core.Cognition.ecs_registry import ecs_world, Entity, ComponentType

# Components
class Position:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z
        self.rx, self.ry, self.rz = 0.0, 0.0, 0.0 # Rotation (Euler)
        self.sx, self.sy, self.sz = 1.0, 1.0, 1.0 # Scale

class Velocity:
    def __init__(self, dx=0.0, dy=0.0, dz=0.0):
        self.dx, self.dy, self.dz = dx, dy, dz

class PhysicsSystem:
    def __init__(self):
        self.gravity = -9.8
        
    def update(self, dt: float):
        # 1. Apply Gravity & Velocity
        for entity, (pos, vel) in ecs_world.view(Position, Velocity):
            # Gravity
            if pos.y > 0:
                vel.dy += self.gravity * dt
            
            # Move
            pos.x += vel.dx * dt
            pos.y += vel.dy * dt
            pos.z += vel.dz * dt
            
            # Floor Collision
            if pos.y < 0:
                pos.y = 0
                vel.dy = 0
                
        # 2. Simulate "Wandering" Intent (The Soul moving the Body)
        # In a real game, this comes from InputSystem.
        for entity, (pos, vel) in ecs_world.view(Position, Velocity):
            if entity.name == "player":
                # Random walk to prove life
                change = (random.random() - 0.5) * 5.0 * dt
                if random.random() < 0.5:
                    vel.dx += change
                else:
                    vel.dz += change
                    
                # Friction/Damping
                vel.dx *= 0.95
                vel.dz *= 0.95
                
                # Jump if on ground
                if pos.y == 0 and random.random() < 0.01:
                    vel.dy = 5.0 

class AnimationSystem:
    """
    The Kinetic Soul.
    Procedurally animates entities based on 'Vitality' (Sine Waves).
    """
    def __init__(self):
        self.time = 0.0
        self.dance_intensity = 0.0
        
    def update(self, dt):
        self.time += dt
        
        # 1. Procedural Breathing (Y-offset)
        breath = math.sin(self.time * 2.0) * 0.05
        
        # 2. Procedural Dance (Layered Sine Waves)
        dance_y = 0.0
        dance_rot_y = 0.0
        
        if self.dance_intensity > 0.0:
            # Bobbing to the beat (2.5Hz ~ 150 BPM)
            dance_y = abs(math.sin(self.time * 10.0)) * 0.2 * self.dance_intensity
            # Swaying hips/body
            dance_rot_y = math.sin(self.time * 5.0) * 0.3 * self.dance_intensity
            
        for entity, (pos,) in ecs_world.view(Position):
                # 3. Apply Animations
                
                # Breathing (Scale Y)
                pos.sy = 1.0 + breath
                pos.sx = 1.0 - (breath * 0.5) # Volume preservation
                pos.sz = 1.0 - (breath * 0.5)
                
                # Idle Sway (Base Rotation)
                idle_rot = math.sin(self.time * 0.5) * 0.2
                
                # Combined Rotation
                pos.ry = idle_rot + dance_rot_y
                
                # Dance Jump Check (Physics handles Y, but we can squash for impact)
                if self.dance_intensity > 0.5:
                    pass # Let physics handle jumping
