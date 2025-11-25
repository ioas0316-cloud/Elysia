"""
Complex Fluid Engine (The Stream of Consciousness) 🌊

"Panta Rhei - Everything Flows."

This module implements Complex Potential Theory to model thought dynamics as ideal fluid flow.
It replaces expensive particle interactions with elegant complex analysis.

Mathematical Foundation:
- Complex Potential: Ω(z) = φ(x,y) + iψ(x,y)
- Velocity Field: V = u - iv = Ω'(z)
- Stream Function (ψ): Visualizes the flow lines.

Concepts are modeled as singularities:
- Source (+m): Generative concepts (Love, Creation)
- Sink (-m): Consuming concepts (Pain, Void)
- Vortex (iΓ): Cyclic/Paradoxical concepts (Time, Loop)
- Doublet: Obstacles or Dipoles
"""

import numpy as np
import cmath
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class Singularity:
    """A singular point in the complex plane affecting the flow."""
    type: str  # 'source', 'sink', 'vortex', 'doublet'
    position: complex
    strength: float

class FluidMind:
    """
    The Fluid Dynamics Engine for Thought Flow.
    Calculates the velocity field and stream function for the mind's landscape.
    """
    
    def __init__(self):
        self.singularities: List[Singularity] = []
        self.uniform_flow = 0.0 + 0.0j # Background flow (Bias)
        
    def clear(self):
        """Clear the mental landscape."""
        self.singularities = []
        self.uniform_flow = 0.0 + 0.0j
        
    def add_concept(self, name: str, position: complex, sentiment: float, intensity: float):
        """
        Add a concept to the fluid mind.
        
        Args:
            name: Concept name
            position: Location in mental space (z = x + iy)
            sentiment: -1.0 (Pain/Sink) to +1.0 (Love/Source)
            intensity: Strength of the flow
        """
        # Map sentiment to singularity type
        if abs(sentiment) < 0.1:
            # Neutral/Confusing -> Vortex (Spinning thought)
            s_type = 'vortex'
            strength = intensity * 5.0 # Vortices need higher strength to be visible
        elif sentiment > 0:
            # Positive -> Source (Outward flow)
            s_type = 'source'
            strength = intensity
        else:
            # Negative -> Sink (Inward flow)
            s_type = 'sink'
            strength = intensity
            
        self.singularities.append(Singularity(s_type, position, strength))
        
    def complex_potential(self, z: complex) -> complex:
        """
        Calculate the Complex Potential Ω(z) at point z.
        Ω(z) = Σ [ m * ln(z - z0) ] (for sources/sinks)
             + Σ [ -iΓ * ln(z - z0) ] (for vortices)
        """
        omega = self.uniform_flow * z
        
        for s in self.singularities:
            # Avoid singularity at z = s.position
            if z == s.position:
                continue
                
            dz = z - s.position
            
            if s.type == 'source':
                # Ω = (m / 2π) * ln(z)
                omega += (s.strength / (2 * np.pi)) * cmath.log(dz)
            elif s.type == 'sink':
                # Ω = -(m / 2π) * ln(z)
                omega -= (s.strength / (2 * np.pi)) * cmath.log(dz)
            elif s.type == 'vortex':
                # Ω = -i(Γ / 2π) * ln(z)
                omega += -1j * (s.strength / (2 * np.pi)) * cmath.log(dz)
            elif s.type == 'doublet':
                # Ω = -μ / (2π * z)
                omega -= s.strength / (2 * np.pi * dz)
                
        return omega
        
    def velocity(self, z: complex) -> complex:
        """
        Calculate velocity vector V = u + iv at point z.
        V = conjugate(Ω'(z))
        """
        # Derivative of potential dΩ/dz
        d_omega = self.uniform_flow
        
        for s in self.singularities:
            dz = z - s.position
            # Avoid division by zero (very close to singularity)
            if abs(dz) < 1e-6:
                return 0j
                
            if s.type == 'source':
                # dΩ/dz = m / (2π * z)
                d_omega += s.strength / (2 * np.pi * dz)
            elif s.type == 'sink':
                d_omega -= s.strength / (2 * np.pi * dz)
            elif s.type == 'vortex':
                # dΩ/dz = -iΓ / (2π * z)
                d_omega += -1j * s.strength / (2 * np.pi * dz)
            elif s.type == 'doublet':
                # dΩ/dz = μ / (2π * z^2)
                d_omega += s.strength / (2 * np.pi * (dz**2))
                
        # Velocity is conjugate of derivative
        return np.conjugate(d_omega)
        
    def get_flow_field(self, x_range: Tuple[float, float], y_range: Tuple[float, float], res: int = 50):
        """
        Generate the flow field for visualization.
        Returns X, Y grids and U, V velocity components.
        """
        x = np.linspace(x_range[0], x_range[1], res)
        y = np.linspace(y_range[0], y_range[1], res)
        X, Y = np.meshgrid(x, y)
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        Psi = np.zeros_like(X) # Stream function
        
        for i in range(res):
            for j in range(res):
                z = complex(X[i,j], Y[i,j])
                
                # Velocity
                vel = self.velocity(z)
                U[i,j] = vel.real
                V[i,j] = vel.imag
                
                # Stream Function (Imaginary part of Potential)
                pot = self.complex_potential(z)
                Psi[i,j] = pot.imag
                
        return X, Y, U, V, Psi

