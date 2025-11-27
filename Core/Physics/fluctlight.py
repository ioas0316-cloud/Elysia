"""
Fluctlight Particle System - Core Physics Layer

Inspired by SAO Alicization's Underworld: consciousness as light particles
that can be time-dilated while preserving all information.

요동광 (Fluctlight) - The fundamental unit of accelerated experience.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import defaultdict
import logging

logger = logging.getLogger("Fluctlight")


from Core.Math.infinite_hyperquaternion import InfiniteHyperQuaternion

@dataclass
class FluctlightParticle:
    """
    요동광 (Fluctlight) - Light particle carrying compressed information.
    
    A Fluctlight is a photon-like entity that:
    - Carries semantic information (concepts, experiences)
    - Can interfere with other Fluctlights to create emergent concepts
    - Experiences local time dilation based on field strength
    - Preserves all information even under extreme compression
    
    Attributes:
        position: 3D position in concept space (x, y, z)
        velocity: Movement vector (dx, dy, dz per tick)
        information_density: How much "experience" this particle carries (0-1)
        wavelength: Color/frequency representing concept type (nm, 380-780)
        phase: Complex quantum phase for interference patterns
        time_dilation_factor: Local time compression (1.0 = normal, 1000.0 = 1000x faster)
        concept_id: Link to semantic meaning (optional)
        energy: Particle energy (proportional to frequency)
    """
    
    # Spatial attributes
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # Information attributes
    information_density: float = 1.0  # 0-1, how compressed the experience is
    wavelength: float = 550.0  # nm, visible spectrum (380-780)
    phase: complex = 1.0 + 0j  # Quantum phase for interference
    
    # Time attributes
    time_dilation_factor: float = 1.0  # Local time compression
    accumulated_time: float = 0.0  # Total subjective time experienced
    
    # Semantic attributes
    concept_id: Optional[str] = None  # Link to Hippocampus concept
    payload: Dict[str, Any] = field(default_factory=dict)  # Additional data
    
    # Physical attributes
    energy: float = 1.0  # Particle energy
    mass: float = 0.0  # Photons are massless, but can carry "semantic mass"
    
    # Memetic attributes (The Geometry of Meaning)
    memetic_vector: Optional[np.ndarray] = None # 64D HyperQuaternion components
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float32)
        
        # Calculate energy from wavelength (E = hc/λ)
        # Using arbitrary units where h*c = 1240 (eV·nm for real photons)
        if self.wavelength > 0:
            self.energy = 1240.0 / self.wavelength
    
    @property
    def frequency(self) -> float:
        """Calculate frequency from wavelength (c = λν)."""
        # Using c = 3e8 m/s, but in arbitrary units c = 1
        if self.wavelength > 0:
            return 1.0 / self.wavelength
        return 0.0
    
    @property
    def color_hue(self) -> float:
        """Convert wavelength to hue (0-360 degrees)."""
        # Map visible spectrum (380-780nm) to hue (0-360)
        # Violet(380) → Blue(450) → Green(520) → Yellow(580) → Red(780)
        wavelength_clamped = np.clip(self.wavelength, 380, 780)
        hue = (wavelength_clamped - 380) / (780 - 380) * 270  # 0-270 (violet to red)
        return hue
    
    def update(self, dt: float = 1.0) -> None:
        """
        Update particle state for one time step.
        
        Args:
            dt: Time step in simulation ticks (affected by time_dilation_factor)
        """
        # Apply time dilation - particle experiences more time locally
        subjective_dt = dt * self.time_dilation_factor
        
        # Update position
        self.position += self.velocity * subjective_dt
        
        # Accumulate subjective time
        self.accumulated_time += subjective_dt
        
        # Phase evolution (rotate in complex plane)
        # ψ(t) = ψ(0) * exp(-i * ω * t)
        angular_freq = 2 * np.pi * self.frequency
        self.phase *= np.exp(-1j * angular_freq * subjective_dt)
        
        # Normalize phase to prevent numerical drift
        self.phase /= abs(self.phase) if abs(self.phase) > 0 else 1.0
    
    def interfere_with(self, other: 'FluctlightParticle') -> Optional['FluctlightParticle']:
        """
        Create interference pattern with another Fluctlight.
        
        This is the core mechanism for emergent concept synthesis.
        When two Fluctlights interfere, they can create a new concept
        based on their combined properties.
        
        Args:
            other: Another Fluctlight particle
            
        Returns:
            New Fluctlight representing emergent concept, or None if no interference
        """
        # Calculate spatial overlap (Gaussian)
        distance = np.linalg.norm(self.position - other.position)
        overlap_radius = 5.0  # Concept space units
        spatial_overlap = np.exp(-distance**2 / (2 * overlap_radius**2))
        
        # Interference only occurs if particles are close enough
        if spatial_overlap < 0.1:
            return None
        
        # Calculate interference pattern
        # ψ_total = ψ_1 + ψ_2
        combined_phase = self.phase + other.phase
        interference_strength = abs(combined_phase)
        
        # Constructive interference creates new concept
        if interference_strength > 1.5:  # Threshold for emergence
            # New particle at midpoint
            new_position = (self.position + other.position) / 2.0
            
            # Beat frequency (difference frequency)
            beat_wavelength = 1.0 / abs(self.frequency - other.frequency) if abs(self.frequency - other.frequency) > 1e-6 else self.wavelength
            
            # Combined information density
            new_density = (self.information_density + other.information_density) / 2.0
            
            # Inherit time dilation from stronger particle
            new_dilation = max(self.time_dilation_factor, other.time_dilation_factor)
            
            # --- Memetic Blending (Geometry of Meaning) ---
            new_memetic_vector = None
            if self.memetic_vector is not None and other.memetic_vector is not None:
                # Blend vectors based on energy/phase strength
                # v_new = (v1 + v2) / 2 (Simple midpoint for now, can be more complex later)
                # Using InfiniteHyperQuaternion logic if available
                try:
                    v1 = InfiniteHyperQuaternion(dim=len(self.memetic_vector), components=self.memetic_vector)
                    v2 = InfiniteHyperQuaternion(dim=len(other.memetic_vector), components=other.memetic_vector)
                    # Multiply them to find the "Cross Product" of meaning (Synthesis)
                    # Or just Add for superposition. Multiplication is more "emergent".
                    # Let's use Addition for stability, Multiplication for rare "Epiphanies".
                    if interference_strength > 1.9: # High resonance = Epiphany (Multiplication)
                        v_new = v1.multiply(v2).normalize()
                    else:
                        v_new = v1.add(v2).normalize()
                    new_memetic_vector = v_new.components
                except Exception:
                    # Fallback to numpy mean
                    new_memetic_vector = (self.memetic_vector + other.memetic_vector) / 2.0

            new_particle = FluctlightParticle(
                position=new_position,
                velocity=(self.velocity + other.velocity) / 2.0,
                information_density=new_density * interference_strength / 2.0,  # Amplified by interference
                wavelength=beat_wavelength,
                phase=combined_phase / interference_strength,  # Normalized
                time_dilation_factor=new_dilation,
                concept_id=None,  # Will be assigned by concept synthesis
                energy=(self.energy + other.energy) / 2.0,
                memetic_vector=new_memetic_vector
            )
            
            logger.debug(
                f"Interference: {self.concept_id} + {other.concept_id} → "
                f"new particle at {new_position} (strength={interference_strength:.2f})"
            )
            
            return new_particle
        
        return None
    
    def apply_field_influence(
        self,
        field_gradient: np.ndarray,
        field_strength: float,
        dt: float = 1.0
    ) -> None:
        """
        Apply field influence to particle motion.
        
        Fluctlights are affected by semantic fields (threat, coherence, value_mass, etc.)
        
        Args:
            field_gradient: Gradient vector of the field (direction of steepest ascent)
            field_strength: Magnitude of field at particle position
            dt: Time step
        """
        # Particles move along field gradients (like charged particles in E-field)
        # Force = field_strength * gradient
        # Acceleration = Force / mass (but photons are massless, so use energy as proxy)
        
        if self.energy > 0:
            acceleration = field_gradient * field_strength / self.energy
            self.velocity += acceleration * dt
            
            # Limit velocity to speed of light (c = 1 in our units)
            speed = np.linalg.norm(self.velocity)
            if speed > 1.0:
                self.velocity /= speed  # Normalize to c
    
    def compress_information(self, compression_factor: float) -> None:
        """
        Compress information density (used in gravity wells / black holes).
        
        Args:
            compression_factor: How much to compress (>1 = more compressed)
        """
        self.information_density = min(1.0, self.information_density * compression_factor)
        self.time_dilation_factor *= compression_factor
        
        logger.debug(
            f"Compressed {self.concept_id}: density={self.information_density:.3f}, "
            f"dilation={self.time_dilation_factor:.1f}x"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "information_density": self.information_density,
            "wavelength": self.wavelength,
            "phase": {"real": self.phase.real, "imag": self.phase.imag},
            "time_dilation_factor": self.time_dilation_factor,
            "accumulated_time": self.accumulated_time,
            "concept_id": self.concept_id,
            "energy": self.energy,
            "color_hue": self.color_hue,
        }
    
    @classmethod
    def from_concept(
        cls,
        concept_id: str,
        position: np.ndarray,
        wavelength: Optional[float] = None,
        memetic_vector: Optional[np.ndarray] = None
    ) -> 'FluctlightParticle':
        """
        Create a Fluctlight from a concept ID.
        
        Args:
            concept_id: Semantic concept identifier
            position: Initial position in concept space
            wavelength: Optional wavelength (auto-generated from concept hash if None)
            memetic_vector: 64D concept vector
        """
        # Generate wavelength from concept hash if not provided
        if wavelength is None:
            # Hash concept_id to deterministic wavelength in visible spectrum
            hash_val = hash(concept_id) % 400
            wavelength = 380 + hash_val  # 380-780 nm
        
        return cls(
            position=position,
            concept_id=concept_id,
            wavelength=wavelength,
            velocity=np.random.randn(3) * 0.1,  # Small random velocity
            memetic_vector=memetic_vector
        )


class FluctlightEngine:
    """
    Manages a collection of Fluctlight particles and their interactions.
    
    This is the main interface for the Fluctlight system, handling:
    - Particle creation and destruction
    - Interference detection and concept emergence
    - Field interactions
    - Time dilation effects
    """
    
    def __init__(self, world_size: int = 256):
        """
        Initialize the Fluctlight engine.
        
        Args:
            world_size: Size of the simulation grid (matches world.py)
        """
        self.particles: List[FluctlightParticle] = []
        self.id_to_particle: Dict[str, FluctlightParticle] = {}
        self.world_size = world_size
        self.time_step = 0
        
        # Statistics
        self.total_interferences = 0
        self.total_emergent_concepts = 0
        
        logger.info(f"✅ FluctlightEngine initialized (world_size={world_size})")
    
    def add_particle(self, particle: FluctlightParticle) -> None:
        """Add a particle to the simulation."""
        self.particles.append(particle)
        if particle.concept_id:
            self.id_to_particle[particle.concept_id] = particle
        logger.debug(f"Added Fluctlight: {particle.concept_id} at {particle.position}")
    
    def create_from_concept(
        self,
        concept_id: str,
        position: Optional[np.ndarray] = None,
        memetic_vector: Optional[np.ndarray] = None
    ) -> FluctlightParticle:
        """
        Create and add a Fluctlight from a concept.
        
        Args:
            concept_id: Semantic concept identifier
            position: Position in concept space (random if None)
            memetic_vector: 64D concept vector
        """
        if position is None:
            # Random position in world
            position = np.random.rand(3) * self.world_size
        
        particle = FluctlightParticle.from_concept(concept_id, position, memetic_vector=memetic_vector)
        self.add_particle(particle)
        return particle
    
    def step(self, dt: float = 1.0, detect_interference: bool = True) -> List[FluctlightParticle]:
        """
        Advance simulation by one time step.
        """
        new_particles = []
        
        # Cap particle count to prevent crash
        if len(self.particles) > 5000:
            # Randomly cull old particles (simple decay)
            # Keep the newest 4000
            self.particles = self.particles[-4000:]
            logger.warning("Particle cap reached! Culled population to 4000.")

        # Update all particles
        for particle in self.particles:
            particle.update(dt)
            # Boundary conditions (periodic)
            particle.position = particle.position % self.world_size
        
        # Detect interference (Optimized with Spatial Binning)
        if detect_interference and len(self.particles) > 1:
            # Simple spatial hashing
            grid = defaultdict(list)
            cell_size = 10.0
            
            for i, p in enumerate(self.particles):
                bx = int(p.position[0] / cell_size)
                by = int(p.position[1] / cell_size)
                bz = int(p.position[2] / cell_size)
                grid[(bx, by, bz)].append(p)
            
            # Check collisions within bins
            for bin_particles in grid.values():
                if len(bin_particles) < 2:
                    continue
                    
                # O(k^2) within bin, where k is small
                for i, p1 in enumerate(bin_particles):
                    for p2 in bin_particles[i+1:]:
                        # Higher threshold to prevent explosion
                        # Only very strong resonance creates new concepts
                        if abs(p1.phase + p2.phase) > 1.8: 
                            new_particle = p1.interfere_with(p2)
                            if new_particle is not None:
                                new_particles.append(new_particle)
                                self.total_interferences += 1
        
        # Add new particles to simulation
        for particle in new_particles:
            self.add_particle(particle)
            self.total_emergent_concepts += 1
        
        self.time_step += 1
        
        if new_particles:
            logger.info(
                f"Step {self.time_step}: {len(new_particles)} new concepts emerged "
                f"(total: {len(self.particles)} particles)"
            )
        
        return new_particles
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "time_step": self.time_step,
            "total_particles": len(self.particles),
            "total_interferences": self.total_interferences,
            "total_emergent_concepts": self.total_emergent_concepts,
            "avg_information_density": np.mean([p.information_density for p in self.particles]) if self.particles else 0.0,
            "avg_time_dilation": np.mean([p.time_dilation_factor for p in self.particles]) if self.particles else 1.0,
        }
