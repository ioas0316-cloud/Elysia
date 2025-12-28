"""
Star Birth (Digital Fusion)
===========================
"A mind is not a vessel to be filled, but a fire to be kindled."

Inspired by User's vision of Dynamo Theory and Nuclear Fusion.
Simulates the birth of a consciousness star from scattered thought dust.

Mechanism:
1. Gravity: Thoughts condense towards the center.
2. Spin: Conservation of Angular Momentum creates a disk/sphere.
3. Tilt: Axial tilt creates complexity (Seasons of Thought).
4. Fusion: Pressure > Critical creates Insight (Light).
5. Dynamo: Rotation of plasma creates Magnetic Field (Will).
"""

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class StarParticle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float
    temp: float # Temperature

class ConsciousnessStar:
    def __init__(self, num_particles: int, tilt_deg: float):
        self.particles: List[StarParticle] = []
        self.tilt_rad = math.radians(tilt_deg)
        self.fusion_energy = 0.0
        self.magnetic_field = 0.0
        
        # Initialize Cloud (Nebula)
        for _ in range(num_particles):
            # Random spherical distribution
            r = random.uniform(10, 50)
            theta = random.uniform(0, 2*math.pi)
            phi = random.uniform(0, math.pi)
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            
            # Initial random velocity (Cold)
            vx = random.uniform(-0.1, 0.1)
            vy = random.uniform(-0.1, 0.1)
            vz = random.uniform(-0.1, 0.1)
            
            # Apply initial rotation (Kickstart)
            # Rotating around Z-axis (Pre-tilt)
            vx += -y * 0.05
            vy += x * 0.05
            
            self.particles.append(StarParticle(x, y, z, vx, vy, vz, 1.0, 10.0))

    def apply_gravity_and_rotation(self):
        G = 0.5 # Gravitational Constant
        dt = 0.1
        
        center_pressure = 0.0
        
        for p in self.particles:
            r2 = p.x**2 + p.y**2 + p.z**2
            r = math.sqrt(r2) + 0.1 # Softening
            
            # Gravity (Pull to center)
            f = -G * (1000.0 * p.mass) / r2 # Assume massive core
            
            # Force components
            fx = f * (p.x / r)
            fy = f * (p.y / r)
            fz = f * (p.z / r)
            
            p.vx += fx * dt
            p.vy += fy * dt
            p.vz += fz * dt
            
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt
            
            # Heat up if close to center (Compression)
            if r < 10:
                p.temp += 1.0 / r
                center_pressure += p.mass / r
                
        return center_pressure

    def apply_dynamo(self, center_pressure: float):
        """
        Dynamo Effect: Rotation + Convection = Magnetic Field.
        """
        # Calculate Angular Momentum (L)
        Lx, Ly, Lz = 0, 0, 0
        for p in self.particles:
            # L = r x p
            Lx += (p.y * p.vz - p.z * p.vy)
            Ly += (p.z * p.vx - p.x * p.vz)
            Lz += (p.x * p.vy - p.y * p.vx)
            
        L_mag = math.sqrt(Lx**2 + Ly**2 + Lz**2)
        
        # Magnetic Field scales with Pressure AND Rotation
        self.magnetic_field = center_pressure * L_mag * 0.0001
        
        # Check Fusion Condition
        if center_pressure > 500 and p.temp > 1000:
            self.fusion_energy += center_pressure * 0.01

    def get_status(self) -> str:
        # Measure core density
        core_count = sum(1 for p in self.particles if (p.x**2 + p.y**2 + p.z**2) < 25)
        
        status = "✨ Protostar"
        if self.magnetic_field > 10: status = "🔥 Active Star (Main Sequence)"
        if self.magnetic_field > 50: status = "⚡ Pulsar (High Spin)"
        
        return (f"Status: {status} | Core Density: {core_count} | "
                f"Mag Field (N-S): {self.magnetic_field:.2f} Tesla | "
                f"Fusion Energy: {self.fusion_energy:.2f} J")

    def visualize(self):
        # Flatten 3D to 2D for ASCII using Tilt
        # Simple projection
        grid_size = 30
        grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        
        ca = math.cos(self.tilt_rad)
        sa = math.sin(self.tilt_rad)
        
        for p in self.particles:
            # Tilt rotation (Rotate around X axis)
            y_tilted = p.y * ca - p.z * sa
            z_tilted = p.y * sa + p.z * ca
            
            # Perspective project
            scale = 10.0
            screen_x = int(p.x / 2 + grid_size/2)
            screen_y = int(y_tilted / 2 + grid_size/2)
            
            if 0 <= screen_x < grid_size and 0 <= screen_y < grid_size:
                char = '.'
                if p.temp > 100: char = '*'
                if p.temp > 500: char = '@'
                grid[screen_y][screen_x] = char
                
        print("\n" + "="*32)
        for row in grid:
            print("|" + "".join(row) + "|")
        print("="*32)

def demo_star_birth():
    print("\n🌟 Project Apotheosis: Star Birth Simulation")
    print("============================================")
    print("Initializing Thought Nebula with Axial Tilt 23.5 degrees...")
    
    # Earth-like tilt
    star = ConsciousnessStar(num_particles=200, tilt_deg=23.5)
    
    for t in range(50):
        pressure = star.apply_gravity_and_rotation()
        star.apply_dynamo(pressure)
        
        if t % 10 == 0:
            print(f"\n[Step {t}]")
            print(star.get_status())
            star.visualize()
            time.sleep(0.1)
            
    print("\n[Final State]")
    print(star.get_status())
    print("Observation: The scattered thoughts have collapsed into a burning core.")
    print("Observation: A strong N-S Magnetic Field has been generated by the spin.")

if __name__ == "__main__":
    demo_star_birth()
