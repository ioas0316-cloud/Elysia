"""
Slime Mind (Plasma Fluid Consciousness)
=======================================
"Thoughts are not points; they are turbulent flows in a magnetic field."

Based on Gray-Scott Reaction-Diffusion Model,
Enhanced with "Plasma Turbulence" metaphor suggested by User.

Mechanism:
- U (Chemical A): The background consciousness (Logic).
- V (Chemical B): The catalyst (Inspiration/Entropy).
- Magnetic Field: The Will/Attention that guides the flow.

Logic:
    dU/dt = (Diff_U * Laplacian(U)) - (U * V^2) + F * (1 - U) + Magnetic_Twist
    dV/dt = (Diff_V * Laplacian(V)) + (U * V^2) - (K + F) * V
"""

import random
import math
from typing import List, Tuple

class PlasmaFluid:
    def __init__(self, size: int):
        self.size = size
        # 2D Grids for Chemical U and V
        self.grid_u = [[1.0 for _ in range(size)] for _ in range(size)]
        self.grid_v = [[0.0 for _ in range(size)] for _ in range(size)]
        
        # Parameters (Gray-Scott presets for "Cell Division" pattern)
        # Feed rate (F) and Kill rate (K)
        self.F = 0.0545
        self.K = 0.0620
        
        # Diffusion rates
        self.Da = 1.0
        self.Db = 0.5
        
        # Seed the center (Inject Idea)
        mid = size // 2
        r = 5
        for x in range(mid-r, mid+r):
            for y in range(mid-r, mid+r):
                self.grid_v[x][y] = 1.0 # High concentration of V

    def laplacian(self, grid: List[List[float]], x: int, y: int) -> float:
        """
        Calculates simple 3x3 convolution for diffusion.
        Center weight -4, Neighbors +1, Diagonals 0 (Simple)
        or weighted version.
        """
        val = grid[x][y] * -1.0
        neighbors = 0.0
        
        # Von Neumann neighborhood (+Cross)
        neighbors += grid[x+1][y] * 0.2 if x+1 < self.size else 0
        neighbors += grid[x-1][y] * 0.2 if x-1 >= 0 else 0
        neighbors += grid[x][y+1][y] * 0.2 if y+1 < self.size else 0
        neighbors += grid[x][y-1][y] * 0.2 if y-1 >= 0 else 0
        
        # Center correction to maintain mass approx
        return neighbors + val 

    def step(self):
        """
        Evolve the plasma field.
        """
        new_u = [[0.0] * self.size for _ in range(self.size)]
        new_v = [[0.0] * self.size for _ in range(self.size)]
        
        for x in range(1, self.size - 1):
            for y in range(1, self.size - 1):
                u = self.grid_u[x][y]
                v = self.grid_v[x][y]
                
                # Reaction (U * V^2)
                reaction = u * v * v
                
                # Diffusion (Spreading)
                # Using simplified direct calculation for speed in Python demo
                lap_u = (self.grid_u[x+1][y] + self.grid_u[x-1][y] + 
                         self.grid_u[x][y+1] + self.grid_u[x][y-1] - 4*u)
                lap_v = (self.grid_v[x+1][y] + self.grid_v[x-1][y] + 
                         self.grid_v[x][y+1] + self.grid_v[x][y-1] - 4*v)
                
                # Plasma Turbulence Operator (User Inspiration) 🌪️
                # "Magnetic Field" adds a twist/vortex effect based on phase
                # Representing 'Quantum Potential' or 'Will'
                magnetic_twist = math.sin(x * 0.1) * math.cos(y * 0.1) * 0.01
                
                # Update equations
                du = (self.Da * lap_u) - reaction + self.F * (1 - u) + magnetic_twist
                dv = (self.Db * lap_v) + reaction - (self.K + self.F) * v
                
                # Clamp results
                new_u[x][y] = min(1.0, max(0.0, u + du))
                new_v[x][y] = min(1.0, max(0.0, v + dv))
                
        self.grid_u = new_u
        self.grid_v = new_v

    def get_pattern_density(self) -> float:
        """Analyze complexity of the formed pattern."""
        active_cells = sum(1 for row in self.grid_v for val in row if val > 0.1)
        return active_cells / (self.size * self.size)

    def visualize_ascii(self):
        """Render the fluid mind state in ASCII."""
        scale = " .:-=+*#%@"
        print("-" * (self.size + 2))
        for row in self.grid_v:
            line = "|"
            for val in row:
                idx = int(val * 9)
                idx = max(0, min(9, idx))
                line += scale[idx]
            line += "|"
            print(line)
        print("-" * (self.size + 2))

def demo_slime_mind():
    print("\n💧 Slime Mind: Plasma Turbulence Simulation")
    print("===========================================")
    print("Initial Inspiration injected at center...")
    
    # 30x30 Grid for display fit
    mind = PlasmaFluid(30)
    
    mind.visualize_ascii()
    
    print("\n... Thinking (Diffusing & Reacting) ...")
    
    # Simulate time
    for t in range(20): # Only 20 steps for quick demo, real patterns enable at 1000+
        mind.step()
        
    mind.visualize_ascii()
    
    density = mind.get_pattern_density()
    print(f"\nFinal Pattern Density: {density:.2%} (The thought has taken shape)")
    
    print("\nObservation:")
    print("The thought is not a 'decision tree'.")
    print("It is a 'living stain' that grows, splits, and breathes.")
    print("Under Plasma Turbulence, it forms complex, localized structures.")

if __name__ == "__main__":
    demo_slime_mind()
