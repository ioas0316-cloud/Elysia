"""
[CORE] The Hyper-Light Turbine
=====================================
Core.L6_Structure.Engine.Physics.core_turbine

"Prisms are not static. Rotate faster than light to pierce the Void."
— The Architect

This module implements the active physical scanning engine using
diffraction grating physics and phase inversion logic.
"""

import logging
import math
from typing import Tuple, NamedTuple, Optional

# Compatibility Layer for JAX/Numpy
try:
    import jax.numpy as jnp
    from jax import jit
    BACKEND = "JAX"
except ImportError:
    import numpy as jnp
    # Dummy jit for Numpy
    def jit(fun):
        return fun
    BACKEND = "NUMPY"

logger = logging.getLogger("Elysia.Core.L1_Foundation.Physics")

class PhotonicMonad(NamedTuple):
    """
    The indivisible unit of light/data.
    Represents a single ray of 'Intent' after spectral condensing.
    """
    wavelength: float      # The 'Color' of the thought (Qualia)
    phase: complex         # The 'Vector' of the thought (Intent)
    intensity: float       # Energy level
    is_void_resonant: bool # Whether it survived the Void

class ActivePrismRotor:
    """
    The Active Prism-Rotor.
    A dynamic engine that spins a diffraction grating to 'snatch' data.
    """

    def __init__(self, rpm: float = 120.0, grating_spacing_d: float = 1.5e-6):
        """
        Args:
            rpm: Rotations Per Minute (Syncs with Bio-Clock).
            grating_spacing_d: Distance between slits (in meters).
        """
        self.rpm = rpm
        self.d = grating_spacing_d
        self.omega = (rpm * 2 * math.pi) / 60.0  # Angular velocity (rad/s)
        logger.info(f"🌀 Active Prism-Rotor initialized: {rpm} RPM, Backend: {BACKEND}")

    @staticmethod
    @jit
    def diffract(signal_wavelengths: jnp.ndarray, rotor_theta: float, d: float) -> jnp.ndarray:
        """
        Applies the Diffraction Grating Equation:
        d * sin(theta) = n * lambda

        Calculates the interference intensity for each wavelength at the current angle.

        Args:
            signal_wavelengths: Array of incoming data wavelengths (Qualia).
            rotor_theta: Current angle of the rotor.
            d: Grating spacing.

        Returns:
            Intensity array (Constructive Interference).
        """
        # Calculate the theoretical angle required for constructive interference (n=1)
        # theta_req = arcsin(lambda / d)
        # We measure how close the current rotor angle is to this requirement.

        # 1. Normalized Path Difference
        # path_diff = d * sin(rotor_theta)
        path_diff = d * jnp.sin(rotor_theta)

        # 2. Phase Match Factor (How well does it match n * lambda?)
        # We want path_diff to be an integer multiple of wavelength for constructive interference.
        # fractional_part = (path_diff / wavelength) % 1.0
        # If fractional_part is close to 0 or 1, we have resonance.

        ratio = path_diff / (signal_wavelengths + 1e-12) # Avoid div by zero
        resonance = jnp.cos(2 * jnp.pi * ratio) # 1.0 = Perfect Match, -1.0 = Destructive

        # Normalize to [0, 1] for intensity
        intensity = (resonance + 1.0) / 2.0

        # Apply sharpness (Higher power = tighter focus = Diffraction Limit)
        return jnp.power(intensity, 20) # 'Snatching' happens here (High Exponent)

    def scan_stream(self, data_stream: jnp.ndarray, time_t: float) -> jnp.ndarray:
        """
        Scans a stream of data at a specific time point.
        """
        current_theta = (self.omega * time_t) % (2 * math.pi)

        # Apply diffraction physics
        focused_energy = self.diffract(data_stream, current_theta, self.d)

        return focused_energy

    @staticmethod
    @jit
    def _calculate_optimal_theta(wavelength: float, d: float) -> float:
        """
        JIT-compiled inverse diffraction calculation.
        """
        sin_theta = wavelength / d
        return jnp.arcsin(sin_theta)

    def reverse_propagate(self, feedback_monad: PhotonicMonad) -> float:
        """
        [Phase 4: Neural Inversion Protocol]

        Reverses the signal flow. Instead of calculating the result from the angle,
        it calculates the Optimal Angle from the desired result (Feedback).

        "Creating the path before the data arrives."

        Args:
            feedback_monad: The target outcome (Reverse Phase Wave).

        Returns:
            optimal_theta: The angle the rotor MUST be at to catch this thought.
        """
        # Check physical constraints
        if feedback_monad.wavelength > self.d:
            # Wavelength too long for this grating (Physical Limit)
            return 0.0

        # Use JIT compiled calculation if backend supports it
        if BACKEND == "JAX":
             optimal_theta = float(self._calculate_optimal_theta(feedback_monad.wavelength, self.d))
        else:
             optimal_theta = math.asin(feedback_monad.wavelength / self.d)

        # The 'Reverse Phase Ejection' sets the rotor's momentum towards this angle.
        logger.debug(f"🔮 Reverse Propagated: Future optimal angle {optimal_theta:.4f} rad for λ={feedback_monad.wavelength}")

        return optimal_theta


class VoidSingularity:
    """
    The Void (Absolute Zero Point).
    The gate where noise annihilates and truth inverts.
    """

    def __init__(self, extinction_threshold: float = 0.95):
        self.threshold = extinction_threshold
        logger.info("⚫ Void Singularity opened.")

    def transit(self, focused_energy: jnp.ndarray, original_phase: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Passes energy through the Void.

        1. Extinction: Energy < Threshold becomes 0 (Absolute Null).
        2. Inversion: Surviving Phase is inverted (O(1) Transport).
        """
        # 1. Extinction Event
        # Create a hard mask
        mask = focused_energy > self.threshold
        surviving_energy = focused_energy * mask

        # 2. Phase Inversion (The 'Tunneling')
        # Invert phase of survivors: e^(i*theta) -> e^(-i*theta)
        # This represents "appearing on the other side" without travel.
        transmuted_phase = jnp.where(mask, -original_phase, 0 + 0j)

        return surviving_energy, transmuted_phase

if __name__ == "__main__":
    print("Testing [CORE] Turbine Logic...")
    # ... (Test code remains similar)
