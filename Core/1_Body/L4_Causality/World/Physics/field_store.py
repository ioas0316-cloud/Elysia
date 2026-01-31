"""
Sparse 4D Field Store
====================
"The Void is zero, existence is the excitation of the Field."

This module implements the Universal Field using a Sparse Map (Hash-based)
to optimize for limited hardware (1060 3GB). It manages the excitations
across the 4D Tesseract (W, X, Y, Z).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from Core.1_Body.L2_Metabolism.Physiology.Physics.geometric_algebra import MultiVector

@dataclass
class FieldExcitation:
    """A specific point of energy in the Field."""
    energy_y: float = 0.0  # Temperature/Frequency
    density_w: float = 0.0 # Pressure/Scale
    intent_z: float = 0.0  # Will/Direction
    perception_x: float = 0.0 # Texture/Form
    
    # Dynamics
    rotor: MultiVector = field(default_factory=lambda: MultiVector(s=1.0))

class UniversalField:
    """
    A sparse 4D container for the HyperCosmos.
    Uses voxel-based hashing for O(Res) complexity.
    """
    def __init__(self, voxel_size: float = 1.0):
        self.voxel_size = voxel_size
        self.voxels: Dict[Tuple[int, int, int, int], FieldExcitation] = {}
        
        # 1. Celestial Constants
        self.star_pos: Tuple[float, float, float, float] = (0, 7.0, 0, 0) 
        self.star_intensity: float = 300.0 # Increased for better reach
        
        # 2. The Moon (Secondary Cycle)
        self.moon_pos: Tuple[float, float, float, float] = (0, 0, 8.0, 0)
        self.moon_intensity: float = 50.0
        
        # 3. Axial Tilt (Experience Anchor)
        # Using a Rotor to tilt the "North" of the field
        from Core.1_Body.L2_Metabolism.Physiology.Physics.geometric_algebra import Rotor
        self.axial_tilt_rotor = Rotor.from_plane_angle('xz', math.radians(23.5))
        
        # 4. Goldilocks Tuning
        self.ATMOSPHERE_RETENTION = 0.98 # Better retention
        self.THERMAL_INERTIA = 0.5 # Faster warming
        self.THERMAL_DECAY_RATE = 0.05 # Default decay, can be mutated by knowledge

    def get_voxel_coord(self, pos: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        # Apply Axial Tilt to the coordinates BEFORE voxel lookup for seasonal effects
        from Core.1_Body.L2_Metabolism.Physiology.Physics.geometric_algebra import Rotor
        tilted_pos = Rotor.rotate_vector(pos, self.axial_tilt_rotor)
        return tuple(int(p / self.voxel_size) for p in tilted_pos)

    def excite(self, pos: Tuple[float, float, float, float], excitation: FieldExcitation):
        coord = self.get_voxel_coord(pos)
        if coord in self.voxels:
            # Additive field effects
            self.voxels[coord].energy_y += excitation.energy_y
            self.voxels[coord].density_w += excitation.density_w
        else:
            self.voxels[coord] = excitation

    def get_field_at(self, pos: Tuple[float, float, float, float]) -> FieldExcitation:
        coord = self.get_voxel_coord(pos)
        return self.voxels.get(coord, FieldExcitation())

    def calculate_gradient_w(self, pos: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Calculates the Pressure Gradient (Direction of Flow)."""
        # Simple finite difference check (6 neighbors in 4D is complex, we use 4 for now)
        x, y, z, w = pos
        d = self.voxel_size
        
        # Check neighbors
        w_right = self.get_field_at((x+d, y, z, w)).density_w
        w_left  = self.get_field_at((x-d, y, z, w)).density_w
        w_up    = self.get_field_at((x, y+d, z, w)).density_w
        w_down  = self.get_field_at((x, y-d, z, w)).density_w
        
        # Gradient vector (Simplified to X, Y plane for 2D wind logic)
        grad_x = (w_right - w_left) / (2*d)
        grad_y = (w_up - w_down) / (2*d)
        
        return (grad_x, grad_y, 0.0, 0.0)

    def apply_celestial_harmonic(self, dt: float = 1.0):
        """Update the field based on the Star and Moon positions."""
        # 1. Update Celestial Positions
        self.moon_pos = (self.moon_pos[0] + 0.05 * dt, self.moon_pos[1], self.moon_pos[2], self.moon_pos[3])
        
        new_voxels = {}
        for coord, ex in self.voxels.items():
            world_pos = tuple(p * self.voxel_size for p in coord)
            
            # Sun (Star) Effect
            d_star = self._get_dist(world_pos, self.star_pos)
            ex.energy_y += (self.star_intensity / (d_star**2)) * dt * self.THERMAL_INERTIA
            
            # Moon Effect (Tides)
            d_moon = self._get_dist(world_pos, self.moon_pos)
            ex.density_w += (self.moon_intensity / d_moon) * dt * 0.02
            
            # 2. Environmental Decay (Entropy)
            ex.energy_y *= (1.0 - self.THERMAL_DECAY_RATE * dt) # Thermal Decay
            ex.density_w *= self.ATMOSPHERE_RETENTION
            
            # 3. Diffusion (Simple neighbor blur) - To be implemented for smoother winds
            
    def _get_dist(self, p1, p2):
        return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2))) + 0.1

    def map_sensation(self, pos: Tuple[float, float, float, float]) -> Dict[str, str]:
        """Translates raw Field data into human-centric sensations."""
        ex = self.get_field_at(pos)
        grad = self.calculate_gradient_w(pos)
        wind_speed = math.sqrt(grad[0]**2 + grad[1]**2)
        
        sensation = {}
        
        # 1. Thermal Sensation
        if ex.energy_y > 3.0: sensation["thermal"] = "Radiant Warmth"
        elif ex.energy_y > 1.0: sensation["thermal"] = "Mild Comfort"
        else: sensation["thermal"] = "Chilling Void"
        
        # 2. Atmospheric Sensation
        if wind_speed > 0.5: sensation["air"] = "Gale Force"
        elif wind_speed > 0.1: sensation["air"] = "Gentle Breeze"
        else: sensation["air"] = "Stagnant Air"
        
        # 3. Existential Sensation (W-axis density)
        if ex.density_w > 15.0: sensation["existence"] = "Oppressive Heavy"
        elif ex.density_w > 8.0: sensation["existence"] = "Grounded Reality"
        else: sensation["existence"] = "Ethereal Scale"
        
        return sensation

# Global Singleton
universe_field = UniversalField()
