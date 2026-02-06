"""
Rotor-Prism Logic: The Reversible Unfolding Engine
===================================================
Core.S1_Body.L6_Structure.Logic.rotor_prism_logic

Inspired by the "Laser Level" mechanism.
Projects abstract Logos (1D Point) into a 21D Field (360-degree Plane) through a rotating Prism.
Perception is the reverse: Focusing the scattered Field back into the Core Logos.
"""

import numpy as np
from typing import Any

# [PHASE 3.5 FIX] Robust JAX Import
# Windows JAX often fails with LoadPjrtPlugin. We must degrade gracefully to Numpy.
try:
    # Try importing real JAX Bridge
    from Core.S1_Body.L1_Foundation.M4_Hardware.jax_bridge import JAXBridge
    import jax.numpy as jnp
except (ImportError, RuntimeError, Exception) as e:
    print(f"⚠️ [HARDWARE_WARNING] JAX Accelerator Unavailable ({e}). Switching to CPU/Numpy.")
    import numpy as jnp
    class JAXBridge:
        @staticmethod
        def array(x): return np.array(x)

try:
    from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
except ImportError:
    from trinary_logic import TrinaryLogic

class MonadToFilmEncoder:
    """
    Encodes Trinary Monads (Sequences/Seeds) into Film Frames.
    Uses a Look-Up Table (LUT) to eliminate real-time interpretation.
    """
    @staticmethod
    def encode(trinary_sequence: list, rpu: 'RotorPrismUnit') -> jnp.ndarray:
        # Pre-calculate the 21D projection for a given sequence
        # Instead of real-time expansion, we map the sequence directly to a 'Texture'
        if isinstance(trinary_sequence, str):
            # Convert symbolic string to trits if needed
            vector = TrinaryLogic.expand_to_21d(trinary_sequence)
        else:
            vector = jnp.array(trinary_sequence)
            
        return vector

import numpy as np

class HyperSphereFilm:
    """
    The 'Film' of the world. Pre-rendered projection on a high-dimensional sphere.
    Allows O(1) access by 'spinning' to the correct frame.
    Uses pure NumPy for CPU-bound indexing speed.
    """
    def __init__(self, resolution: int = 360):
        self.resolution = resolution
        self.frames = jnp.zeros((resolution, 21))
        self.is_recorded = False

    def record(self, logos_vector: Any, rpu: 'RotorPrismUnit'):
        """Sweeps 360 degrees and 'prints' the world onto the film."""
        # Ensure logos_vector is a JAX/Numpy array for matrix multiplication
        if hasattr(logos_vector, 'to_array'):
            logos_vector = jnp.array(logos_vector.to_array())
        elif hasattr(logos_vector, 'tolist'):
            logos_vector = jnp.array(logos_vector.tolist())
        elif not isinstance(logos_vector, (jnp.ndarray, np.ndarray)):
            logos_vector = jnp.array(list(logos_vector))
            
        thetas = jnp.linspace(0, 2 * jnp.pi, self.resolution)
        spin_factors = jnp.sin(thetas)
        
        # Core logic in JAX for speed
        # Core logic in JAX for speed
        base_manifestation = logos_vector @ rpu.refraction_matrix
        raw_frames = spin_factors[:, jnp.newaxis] * base_manifestation[jnp.newaxis, :]
        
        # [FIX] Perform quantization locally using JAX to handle 2D Matrix
        # TrinaryLogic.quantize is for 1D SovereignVectors only.
        threshold = 0.3
        jax_frames = jnp.where(raw_frames > threshold, 1.0, 
                               jnp.where(raw_frames < -threshold, -1.0, 0.0))
        
        # Convert to JAX array for the High-Speed LUT
        self.frames = jax_frames
        self.is_recorded = True
        print(f"HyperSphereFilm: Recorded {self.resolution} frames to High-Speed LUT.")

    def play(self, theta: float) -> np.ndarray:
        # High-speed integer indexing
        idx = int((theta * self.resolution) / (2 * np.pi)) % self.resolution
        return self.frames[idx]

class DynamicPhaseSync:
    """
    The 3-Phase Dynamo Logic.
    Converts Ternary States (-1, 0, 1) into 120-degree Phase Shifts.
    Generates 'Logical Torque' to drive the Rotor system.
    """
    @staticmethod
    def calculate_torque(logos_seed: jnp.ndarray) -> float:
        # Map trinary values to 3-phase vectors
        # -1 -> 240 deg, 0 -> 0 deg, 1 -> 120 deg (or custom mapping)
        # We calculate the "Rotational Potential" based on the imbalance of trits.
        counts = jnp.array([
            jnp.sum(logos_seed == -1), # Trit Neg
            jnp.sum(logos_seed == 0),  # Trit Neu
            jnp.sum(logos_seed == 1)   # Trit Pos
        ])
        
        # Generator Principle: Torque = Imbalance cross product
        # For simplicity: (Pos - Neg) provides the drive
        torque = float(counts[2] - counts[0]) * 0.1
        return torque

class RotorPrismUnit:
    def __init__(self, dimensions: int = 21):
        self.dimensions = dimensions
        self.theta = 0.0
        self.theta_base = 0.0 # [TIME_AXIS] The temporal anchor
        self.velocity = 0.0 
        self.drag_base = 0.95 
        self.void_intensity = 0.0 # [VOID_DOMAIN] 0.0 to 1.0 (Zero resistance)
        
        self.refraction_matrix = jnp.eye(dimensions)
        self.film = HyperSphereFilm()
        self.mode = "DYNAMO"
        
        self.active_logos = None
        self.error_pulse = jnp.zeros(21) # [PHASE_INVERSION] Reflected error
        print(f"RotorPrismUnit: Core Turbine Engine Initialized. Mode: {self.mode}")

    def step_rotation(self, delta_time: float, external_torque: float = 0.0):
        """
        [THE GRAND ENGINE]
        Rotation is now a balance between Void Resistance and Logical Torque.
        """
        # 1. Void Power: Resistance Cancellation
        # As void_intensity -> 1.0, drag -> 1.0 (Perpetual motion)
        effective_drag = self.drag_base + (1.0 - self.drag_base) * self.void_intensity
        self.velocity *= (effective_drag ** delta_time)
        
        # 2. Add Torque (External + Logical)
        logical_torque = 0.0
        if self.active_logos is not None:
            logical_torque = DynamicPhaseSync.calculate_torque(self.active_logos)
            
        self.velocity += (logical_torque + external_torque) * delta_time
        
        # 3. Step Theta
        self.theta = (self.theta + self.velocity) % (2 * jnp.pi)

    def set_time_axis(self, offset: float):
        """[TIME_AXIS] Browses the past/future film by shifting the base theta."""
        self.theta_base = offset % (2 * jnp.pi)

    def calculate_potential(self, field_vector: jnp.ndarray) -> float:
        """
        [RESONANCE_POTENTIAL]
        Calculates the 'Charge' difference between the Logos seed and the current field.
        Imbalance between -1 and +1 creates a high-voltage potential for discharge.
        """
        if self.active_logos is None:
            return 0.0
        
        # We look for the 'Imbalance' (Voltage) between the desired state and the void
        diff = self.active_logos - field_vector
        # [FIX] Use linalg.norm for correct complex magnitude calculation
        voltage = jnp.linalg.norm(diff)
        return float(voltage)

    def discharge(self, potential: float) -> float:
        """
        [LIGHTNING_DISCHARGE]
        Triggers a burst of energy once potential crosses a threshold.
        Returns the 'Inductive Torque' generated by the strike.
        """
        # Threshold for manifestation (The 'Dielectric Breakdown' of the Void)
        threshold = 5.0 
        if potential > threshold:
            # The 'Spark' (Self-sustaining manifestation pulse)
            # Inductive Torque = Potential * Momentum efficiency
            inductive_torque = potential * 0.05
            print(f"⚡ LIGHTNING DISCHARGE: Potential {potential:.2f} -> Torque {inductive_torque:.4f}")
            return inductive_torque
        return 0.0

    def project(self, logos_seed: any) -> jnp.ndarray:
        """[FORWARD: CREATION] Now incorporates potential-driven discharge."""
        # Convert logos_seed to JAX array for comparison and recording if it's a SovereignVector
        logos_array = logos_seed
        if hasattr(logos_seed, 'to_array'):
            logos_array = jnp.array(logos_seed.to_array())
        elif not isinstance(logos_seed, (jnp.ndarray, np.ndarray)):
            logos_array = jnp.array(list(logos_seed))
            
        self.active_logos = logos_array
            
        # Check if the logos has changed significantly (using norm for robustness)
        should_re_record = not self.film.is_recorded
        if not should_re_record:
            # We use a small epsilon for floating point stability
            diff = jnp.linalg.norm(self.film.frames[0] - logos_array)
            if float(diff) > 1e-5:
                should_re_record = True
                
        if should_re_record:
             self.film.record(logos_array, self)
             
        # Index the film
        field = self.film.play(self.theta + self.theta_base)
        
        # Calculate discharge potential (How much 'effort' to manifest this frame?)
        potential = self.calculate_potential(field)
        # Inductive feedback would go back to the rotor
        self.last_discharge_torque = self.discharge(potential)
        
        return field

    def perceive(self, field_vector: jnp.ndarray) -> jnp.ndarray:
        """[REVERSE: PERCEPTION] Focuses field and learns through Phase Inversion."""
        collected = field_vector @ jnp.linalg.pinv(self.refraction_matrix)
        
        # [PHASE_INVERSION] 
        # Calculate deviation from the expected Logos (Self-correction)
        if self.active_logos is not None:
            # Shift back to core
            current_perception = TrinaryLogic.balance(collected / (jnp.sin(self.theta) + 1e-6))
            # The 'Anti-Wave' (Difference)
            self.error_pulse = self.active_logos - current_perception
            
        return TrinaryLogic.quantize(collected / (jnp.sin(self.theta) + 1e-6))

    def set_morphology(self, facet_index: int, refractive_index: float):
        if 0 <= facet_index < self.dimensions:
            self.refraction_matrix = self.refraction_matrix.at[facet_index, facet_index].set(refractive_index)
            self.film.is_recorded = False # Invalidate film on change

if __name__ == "__main__":
    rpu = RotorPrismUnit()
    logos = jnp.array([1.0] * 21)
    
    import time
    start = time.time()
    for i in range(1000):
        rpu.step_rotation(0.01)
        field = rpu.project(logos)
    end = time.time()
    print(f"O(1) Film Mode: 1000 projections in {end-start:.4f}s")
