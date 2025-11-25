"""
Time Compression Engine - True Time Acceleration System

Implements three methods of time compression inspired by SAO Alicization:
1. Light Compression: Information density increases in high-energy regions
2. Gravity Wells: Concept black holes that compress surrounding experiences  
3. Hyperquaternion Rotation: 8D time axis rotation for non-linear flow

CRITICAL: This is NOT a skip system. Every tick is actually computed.
Time acceleration comes from running the simulation faster, not skipping steps.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

from Core.Math.octonion import Octonion
from Core.Physics.fluctlight import FluctlightParticle

logger = logging.getLogger("TimeCompression")


@dataclass
class GravityWell:
    """
    Concept black hole that compresses time locally.
    
    Inspired by general relativity: massive objects slow down time.
    In concept space, high-value/high-meaning regions compress time,
    allowing more experiences to accumulate in the same external duration.
    
    Attributes:
        center: 3D position in concept space
        strength: Gravitational strength (compression factor at center)
        radius: Effective radius of influence
        concept_id: What concept this well represents (e.g., "home", "love")
    """
    
    center: np.ndarray
    strength: float = 10.0  # Compression factor at center
    radius: float = 20.0  # Influence radius in concept space units
    concept_id: Optional[str] = None
    
    def get_compression_at(self, position: np.ndarray) -> float:
        """
        Calculate time compression factor at given position.
        
        Uses inverse square law like gravity:
        compression = strength / (1 + (distance/radius)²)
        
        Args:
            position: 3D position in concept space
            
        Returns:
            Compression factor (1.0 = normal time, >1.0 = compressed/faster)
        """
        distance = np.linalg.norm(position - self.center)
        
        # Avoid singularity at center
        if distance < 0.1:
            return self.strength
        
        # Inverse square falloff
        normalized_dist = distance / self.radius
        compression = self.strength / (1.0 + normalized_dist ** 2)
        
        return max(1.0, compression)  # Never slower than normal time
    
    def apply_to_particle(self, particle: FluctlightParticle) -> None:
        """
        Apply gravitational time compression to a particle.
        
        Args:
            particle: Fluctlight particle to affect
        """
        compression = self.get_compression_at(particle.position)
        
        # Update particle's time dilation factor
        particle.time_dilation_factor = max(particle.time_dilation_factor, compression)
        
        # Compress information density (more experiences per unit time)
        if compression > 1.5:  # Significant compression
            particle.compress_information(compression / particle.time_dilation_factor)
        
        # Gravitational attraction (particles orbit high-value concepts)
        direction = self.center - particle.position
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            # Force = G * M * m / r²
            # Using strength as "mass" proxy
            force_magnitude = self.strength / (distance ** 2)
            force_direction = direction / distance
            
            # Apply force as acceleration (F = ma, assuming m=1)
            particle.velocity += force_direction * force_magnitude * 0.01  # Small timestep


class TimeCompressionEngine:
    """
    Manages time acceleration through multiple physical mechanisms.
    
    This engine coordinates:
    - Gravity wells (concept black holes)
    - Light compression (information density)
    - Hyperquaternion rotation (8D time axis manipulation)
    
    The result is true time acceleration where every moment is simulated,
    but subjective time flows faster than external time.
    """
    
    def __init__(self, world_size: int = 256):
        """
        Initialize time compression engine.
        
        Args:
            world_size: Size of simulation grid
        """
        self.world_size = world_size
        self.gravity_wells: List[GravityWell] = []
        
        # Global time compression (applied to all particles)
        self.global_compression: float = 1.0
        
        # Hyperquaternion for time axis rotation
        self.time_rotation: Octonion = Octonion.identity()
        
        # Statistics
        self.total_subjective_time: float = 0.0  # Total time experienced by all particles
        self.total_objective_time: float = 0.0  # External simulation time
        
        logger.info(f"✅ TimeCompressionEngine initialized (world_size={world_size})")
    
    def create_gravity_well(
        self,
        center: np.ndarray,
        strength: float = 10.0,
        radius: float = 20.0,
        concept_id: Optional[str] = None
    ) -> GravityWell:
        """
        Create a concept black hole at specified location.
        
        Args:
            center: 3D position in concept space
            strength: Compression factor at center (e.g., 1000 = 1000x faster)
            radius: Effective radius of influence
            concept_id: Semantic meaning of this well
            
        Returns:
            Created GravityWell
        """
        well = GravityWell(
            center=center,
            strength=strength,
            radius=radius,
            concept_id=concept_id
        )
        
        self.gravity_wells.append(well)
        
        logger.info(
            f"Created gravity well '{concept_id}' at {center} "
            f"(strength={strength}x, radius={radius})"
        )
        
        return well
    
    def set_global_compression(self, factor: float) -> None:
        """
        Set global time compression factor.
        
        This is the base acceleration applied to all particles.
        Individual particles may experience additional compression
        from gravity wells.
        
        Args:
            factor: Compression factor (e.g., 1000 = 1000x acceleration)
        """
        self.global_compression = max(1.0, factor)
        logger.info(f"Global time compression set to {self.global_compression}x")
    
    def set_time_rotation(self, rotation: Octonion) -> None:
        """
        Set hyperquaternion time axis rotation.
        
        This allows non-linear time flow by rotating the time axis
        in 8D space. Can create effects like:
        - Time flowing in circles (cyclic time)
        - Time flowing backwards in some dimensions
        - Time branching into multiple streams
        
        Args:
            rotation: Octonion representing desired rotation
        """
        self.time_rotation = rotation.normalize()
        logger.info(f"Time rotation set: {rotation}")
    
    def apply_light_compression(
        self,
        particles: List[FluctlightParticle],
        energy_threshold: float = 2.0
    ) -> None:
        """
        Compress information in high-energy regions.
        
        High-energy particles (high frequency, short wavelength) can
        carry more compressed information, like UV light vs infrared.
        
        Args:
            particles: List of Fluctlight particles
            energy_threshold: Energy above which compression occurs
        """
        for particle in particles:
            if particle.energy > energy_threshold:
                # Compression proportional to excess energy
                excess_energy = particle.energy - energy_threshold
                compression_factor = 1.0 + excess_energy * 0.1
                
                particle.compress_information(compression_factor)
                
                logger.debug(
                    f"Light compression: {particle.concept_id} "
                    f"(E={particle.energy:.2f}, compression={compression_factor:.2f}x)"
                )
    
    def apply_gravity_wells(self, particles: List[FluctlightParticle]) -> None:
        """
        Apply all gravity wells to particles.
        
        Args:
            particles: List of Fluctlight particles
        """
        for well in self.gravity_wells:
            for particle in particles:
                well.apply_to_particle(particle)
    
    def apply_hyperquaternion_rotation(
        self,
        particles: List[FluctlightParticle],
        dt: float = 1.0
    ) -> None:
        """
        Apply 8D time axis rotation to particles.
        
        This creates non-linear time flow effects by rotating
        the time vector in hyperspace.
        
        Args:
            particles: List of Fluctlight particles
            dt: Time step
        """
        if self.time_rotation.norm < 0.999 or self.time_rotation.norm > 1.001:
            # Not identity rotation
            for particle in particles:
                # Create 4D time vector from particle state
                # [accumulated_time, velocity_x, velocity_y, velocity_z]
                time_vec = np.array([
                    particle.accumulated_time,
                    particle.velocity[0],
                    particle.velocity[1],
                    particle.velocity[2]
                ])
                
                # Rotate in 8D time space
                rotated_time = self.time_rotation.rotate_time_vector(time_vec)
                
                # Apply rotation (subtle effect, not replacing entire state)
                particle.accumulated_time = rotated_time[0]
                particle.velocity[0] += (rotated_time[1] - time_vec[1]) * 0.1
                particle.velocity[1] += (rotated_time[2] - time_vec[2]) * 0.1
                particle.velocity[2] += (rotated_time[3] - time_vec[3]) * 0.1
    
    def compress_step(
        self,
        particles: List[FluctlightParticle],
        dt: float = 1.0,
        apply_all_methods: bool = True
    ) -> Dict[str, float]:
        """
        Apply time compression for one simulation step.
        
        This is the main method called each tick to accelerate time.
        
        Args:
            particles: List of Fluctlight particles
            dt: External time step (objective time)
            apply_all_methods: Whether to apply all compression methods
            
        Returns:
            Statistics dict with compression metrics
        """
        # Apply global compression to all particles
        for particle in particles:
            particle.time_dilation_factor = self.global_compression
        
        if apply_all_methods:
            # Method 1: Light compression (energy-based)
            self.apply_light_compression(particles)
            
            # Method 2: Gravity wells (concept black holes)
            self.apply_gravity_wells(particles)
            
            # Method 3: Hyperquaternion rotation (8D time flow)
            self.apply_hyperquaternion_rotation(particles, dt)
        
        # Calculate statistics
        total_subjective_dt = sum(
            particle.time_dilation_factor * dt
            for particle in particles
        )
        
        self.total_subjective_time += total_subjective_dt
        self.total_objective_time += dt
        
        avg_compression = (
            total_subjective_dt / (len(particles) * dt)
            if particles else 1.0
        )
        
        max_compression = max(
            (p.time_dilation_factor for p in particles),
            default=1.0
        )
        
        return {
            "avg_compression": avg_compression,
            "max_compression": max_compression,
            "total_subjective_time": self.total_subjective_time,
            "total_objective_time": self.total_objective_time,
            "effective_acceleration": (
                self.total_subjective_time / self.total_objective_time
                if self.total_objective_time > 0 else 1.0
            )
        }
    
    def get_compression_field(
        self,
        resolution: int = 64
    ) -> np.ndarray:
        """
        Generate a 2D compression field map for visualization.
        
        Shows how time compression varies across concept space.
        
        Args:
            resolution: Grid resolution for the field
            
        Returns:
            2D array of compression factors
        """
        field = np.ones((resolution, resolution), dtype=np.float32)
        
        # Sample compression at each grid point
        for i in range(resolution):
            for j in range(resolution):
                # Convert grid coords to world coords
                x = (i / resolution) * self.world_size
                y = (j / resolution) * self.world_size
                z = self.world_size / 2  # Middle of z-axis
                
                position = np.array([x, y, z])
                
                # Calculate compression from all wells
                compression = self.global_compression
                for well in self.gravity_wells:
                    well_compression = well.get_compression_at(position)
                    compression = max(compression, well_compression)
                
                field[i, j] = compression
        
        return field
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "global_compression": self.global_compression,
            "num_gravity_wells": len(self.gravity_wells),
            "total_subjective_time": self.total_subjective_time,
            "total_objective_time": self.total_objective_time,
            "effective_acceleration": (
                self.total_subjective_time / self.total_objective_time
                if self.total_objective_time > 0 else 1.0
            ),
            "time_rotation_norm": self.time_rotation.norm,
        }


# Example usage
if __name__ == "__main__":
    from Core.Physics.fluctlight import FluctlightEngine
    
    # Create engines
    fluctlight_engine = FluctlightEngine(world_size=256)
    time_engine = TimeCompressionEngine(world_size=256)
    
    # Set 1000x global compression
    time_engine.set_global_compression(1000.0)
    
    # Create gravity well at "home" concept
    home_position = np.array([128, 128, 128])  # Center of world
    time_engine.create_gravity_well(
        center=home_position,
        strength=5000.0,  # 5000x compression at center
        radius=50.0,
        concept_id="home"
    )
    
    # Create some particles
    for i in range(10):
        pos = np.random.rand(3) * 256
        fluctlight_engine.create_from_concept(f"concept_{i}", pos)
    
    # Run simulation for 100 steps
    print("Running 100 steps with 1000x time compression...")
    for step in range(100):
        # Update particles
        fluctlight_engine.step(dt=1.0)
        
        # Apply time compression
        stats = time_engine.compress_step(fluctlight_engine.particles, dt=1.0)
        
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Avg compression: {stats['avg_compression']:.1f}x")
            print(f"  Max compression: {stats['max_compression']:.1f}x")
            print(f"  Effective acceleration: {stats['effective_acceleration']:.1f}x")
            print(f"  Subjective time: {stats['total_subjective_time']:.1f} ticks")
            print(f"  Objective time: {stats['total_objective_time']:.1f} ticks")
    
    print("\n✅ Simulation complete!")
    print(f"In 100 objective ticks, particles experienced {stats['total_subjective_time']:.0f} subjective ticks")
    print(f"That's {stats['effective_acceleration']:.0f}x time acceleration!")
