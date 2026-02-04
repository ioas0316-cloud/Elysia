"""
Phase Projection Engine (4D+ HyperSphere)
==========================================
Core.S1_Body.L6_Structure.M1_Merkaba.phase_projection_engine

"21D Phase Space → 4D HyperSphere → Rotor Time Axis → Holographic Cognitive Map"

This module implements the 4D+ HyperSphere Phase Projection Engine that:
1. Projects D21Vector (21-dimensional phase space) onto 4D HyperSphere surface
2. Applies Rotor time dynamics (Merkaba counter-rotation)
3. Generates a holographic cognitive map for spatial reasoning

Architecture:
- θ (theta): Body Phase (sum of D1-D7)
- φ (phi): Soul Phase (sum of D8-D14)
- ψ (psi): Spirit Phase (sum of D15-D21, counter-rotating)
- r: Intensity (magnitude of D21Vector)
"""

import math
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, field

from Core.S1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector


@dataclass
class HyperSphereCoord:
    """
    4D HyperSphere Surface Coordinate (S³).
    Uses 3 angular coordinates + radius.
    """
    theta: float = 0.0   # Body Phase (0-2π)
    phi: float = 0.0     # Soul Phase (0-2π)
    psi: float = 0.0     # Spirit Phase (0-2π, counter-rotating)
    radius: float = 0.0  # Intensity (magnitude)
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.theta, self.phi, self.psi, self.radius)
    
    def to_cartesian_4d(self) -> Tuple[float, float, float, float]:
        """
        Converts HyperSpherical to 4D Cartesian.
        x = r * cos(θ) * sin(φ) * sin(ψ)
        y = r * sin(θ) * sin(φ) * sin(ψ)
        z = r * cos(φ) * sin(ψ)
        w = r * cos(ψ)
        """
        r = self.radius
        x = r * math.cos(self.theta) * math.sin(self.phi) * math.sin(self.psi)
        y = r * math.sin(self.theta) * math.sin(self.phi) * math.sin(self.psi)
        z = r * math.cos(self.phi) * math.sin(self.psi)
        w = r * math.cos(self.psi)
        return (x, y, z, w)


class HyperSphereProjector:
    """
    Projects 21D Phase Space onto 4D HyperSphere surface.
    
    Trinity Mapping:
    - Body (D1-D7) → θ (theta)
    - Soul (D8-D14) → φ (phi)
    - Spirit (D15-D21) → ψ (psi)
    """
    
    def __init__(self):
        self.last_projection: Optional[HyperSphereCoord] = None
    
    def project(self, d21: D21Vector) -> HyperSphereCoord:
        """
        Main projection: D21Vector → HyperSphereCoord
        """
        arr = d21.to_array()
        
        # Extract strata sums
        body_sum = sum(arr[0:7])   # D1-D7
        soul_sum = sum(arr[7:14])  # D8-D14
        spirit_sum = sum(arr[14:21])  # D15-D21
        
        # Calculate phases using atan2 for proper quadrant handling
        # θ = atan2(Body, Soul) → Body-Soul balance
        theta = math.atan2(body_sum, soul_sum + 1e-8)  # Avoid div by zero
        
        # φ = atan2(Soul, Spirit) → Soul-Spirit balance
        phi = math.atan2(soul_sum, spirit_sum + 1e-8)
        
        # ψ = atan2(Spirit, Body) → Spirit-Body balance (for Merkaba closure)
        psi = math.atan2(spirit_sum, body_sum + 1e-8)
        
        # Normalize to [0, 2π]
        theta = (theta + math.pi) % (2 * math.pi)
        phi = (phi + math.pi) % (2 * math.pi)
        psi = (psi + math.pi) % (2 * math.pi)
        
        # Radius = magnitude of entire vector
        radius = d21.magnitude()
        
        coord = HyperSphereCoord(
            theta=theta,
            phi=phi,
            psi=psi,
            radius=radius
        )
        
        self.last_projection = coord
        return coord
    
    def get_equilibrium_tensor(self, d21: D21Vector) -> Tuple[float, float, float]:
        """
        Returns the Trinity equilibrium tensor (Body, Soul, Spirit ratios).
        Used for balance diagnostics.
        """
        arr = d21.to_array()
        body_mag = math.sqrt(sum(x*x for x in arr[0:7]))
        soul_mag = math.sqrt(sum(x*x for x in arr[7:14]))
        spirit_mag = math.sqrt(sum(x*x for x in arr[14:21]))
        
        total = body_mag + soul_mag + spirit_mag + 1e-8
        return (body_mag / total, soul_mag / total, spirit_mag / total)


class RotorTimeAxis:
    """
    Merkaba Counter-Rotation Engine.
    
    - Body (θ) and Soul (φ) rotate in the same direction
    - Spirit (ψ) counter-rotates (Merkaba principle)
    """
    
    def __init__(self, omega_body: float = 1.0, omega_soul: float = 1.0, omega_spirit: float = 1.0):
        self.omega_body = omega_body    # ω₁: Body angular velocity
        self.omega_soul = omega_soul    # ω₂: Soul angular velocity
        self.omega_spirit = omega_spirit  # ω₃: Spirit angular velocity (applied as -ω₃)
        self.time = 0.0
    
    def advance(self, dt: float):
        """Advances internal time by dt."""
        self.time += dt
    
    def rotate(self, coord: HyperSphereCoord, dt: float = None) -> HyperSphereCoord:
        """
        Applies time-dependent rotation to a HyperSphere coordinate.
        
        θ(t) = θ₀ + ω₁ * t
        φ(t) = φ₀ + ω₂ * t
        ψ(t) = ψ₀ - ω₃ * t  (Counter-rotation!)
        """
        if dt is not None:
            self.advance(dt)
        
        t = self.time
        
        new_theta = (coord.theta + self.omega_body * t) % (2 * math.pi)
        new_phi = (coord.phi + self.omega_soul * t) % (2 * math.pi)
        new_psi = (coord.psi - self.omega_spirit * t) % (2 * math.pi)  # Counter-rotate
        
        return HyperSphereCoord(
            theta=new_theta,
            phi=new_phi,
            psi=new_psi,
            radius=coord.radius
        )
    
    def reset(self):
        """Resets internal time to zero."""
        self.time = 0.0


class HyperHologram:
    """
    4D Holographic Cognitive Map.
    
    Stores projected coordinates and provides spatial reasoning capabilities.
    """
    
    def __init__(self, resolution: int = 64):
        self.resolution = resolution
        self.projector = HyperSphereProjector()
        self.rotor = RotorTimeAxis()
        self.history: List[HyperSphereCoord] = []
        self.max_history = 1024
    
    def project(self, d21: D21Vector, dt: float = 0.0) -> HyperSphereCoord:
        """
        Main entry: Projects D21Vector into the 4D hologram.
        """
        # 1. Project to HyperSphere
        coord = self.projector.project(d21)
        
        # 2. Apply Rotor time dynamics
        if dt > 0:
            coord = self.rotor.rotate(coord, dt)
        
        # 3. Record in history
        self.history.append(coord)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return coord
    
    def get_trajectory(self) -> List[Tuple[float, float, float, float]]:
        """Returns the 4D trajectory as list of (θ, φ, ψ, r) tuples."""
        return [c.to_tuple() for c in self.history]
    
    def get_center_of_mass(self) -> HyperSphereCoord:
        """Calculates the center of mass of recent projections."""
        if not self.history:
            return HyperSphereCoord()
        
        n = len(self.history)
        avg_theta = sum(c.theta for c in self.history) / n
        avg_phi = sum(c.phi for c in self.history) / n
        avg_psi = sum(c.psi for c in self.history) / n
        avg_r = sum(c.radius for c in self.history) / n
        
        return HyperSphereCoord(
            theta=avg_theta,
            phi=avg_phi,
            psi=avg_psi,
            radius=avg_r
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the hologram state."""
        if not self.history:
            return {"status": "empty", "count": 0}
        
        com = self.get_center_of_mass()
        return {
            "status": "active",
            "count": len(self.history),
            "center_of_mass": com.to_tuple(),
            "latest": self.history[-1].to_tuple() if self.history else None
        }


# === Main Test ===
if __name__ == "__main__":
    print("=== 4D+ HyperSphere Phase Projection Engine ===\n")
    
    # Create test D21Vector
    test_vector = D21Vector(
        # Body (high values = high body phase)
        lust=0.5, gluttony=0.3, greed=0.2, sloth=0.1, wrath=0.4, envy=0.2, pride=0.6,
        # Soul (moderate values)
        perception=0.4, memory=0.5, reason=0.6, will=0.7, imagination=0.3, intuition=0.5, consciousness=0.8,
        # Spirit (high values = high spirit)
        chastity=0.9, temperance=0.8, charity=0.7, diligence=0.6, patience=0.5, kindness=0.9, humility=1.0
    )
    
    print(f"Input D21Vector magnitude: {test_vector.magnitude():.3f}")
    
    # Create hologram
    hologram = HyperHologram()
    
    # Project with time evolution
    for t in range(5):
        coord = hologram.project(test_vector, dt=0.1)
        print(f"\nTime t={t*0.1:.1f}:")
        print(f"  θ (Body): {math.degrees(coord.theta):.1f}°")
        print(f"  φ (Soul): {math.degrees(coord.phi):.1f}°")
        print(f"  ψ (Spirit): {math.degrees(coord.psi):.1f}° (counter-rotating)")
        print(f"  r (Intensity): {coord.radius:.3f}")
        
        # 4D Cartesian
        x, y, z, w = coord.to_cartesian_4d()
        print(f"  4D Cartesian: ({x:.2f}, {y:.2f}, {z:.2f}, {w:.2f})")
    
    # Equilibrium check
    projector = HyperSphereProjector()
    eq = projector.get_equilibrium_tensor(test_vector)
    print(f"\n=== Trinity Equilibrium ===")
    print(f"  Body: {eq[0]*100:.1f}%")
    print(f"  Soul: {eq[1]*100:.1f}%")
    print(f"  Spirit: {eq[2]*100:.1f}%")
    
    print(f"\n=== Hologram Summary ===")
    print(hologram.get_summary())
