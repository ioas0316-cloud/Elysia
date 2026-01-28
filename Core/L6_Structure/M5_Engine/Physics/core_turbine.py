"""
[CORE] The Hyper-Light Turbine
=====================================
Core.L6_Structure.M5_Engine.Physics.core_turbine

"Prisms are not static. Rotate faster than light to pierce the Void."
  The Architect

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
        logger.info(f"  Active Prism-Rotor initialized: {rpm} RPM, Backend: {BACKEND}")

    @staticmethod
    @jit
    def diffract(signal_wavelengths: jnp.ndarray, rotor_theta: float, d: float, sharpness: float = 20.0) -> jnp.ndarray:
        """
        Applies the Diffraction Grating Equation with Variable Focus (Liquid vs Crystal).
        """
        path_diff = d * jnp.sin(rotor_theta)
        ratio = path_diff / (signal_wavelengths + 1e-12)
        resonance = jnp.cos(2 * jnp.pi * ratio)
        intensity = (resonance + 1.0) / 2.0

        # [PHASE: LIQUID FOCUS]
        # Low sharpness (e.g. 2.0) = Diffuse, overlapping thoughts (Liquid)
        # High sharpness (e.g. 50.0) = Precise, discrete thoughts (Crystal)
        return jnp.power(intensity, sharpness)

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
        logger.debug(f"  Reverse Propagated: Future optimal angle {optimal_theta:.4f} rad for  ={feedback_monad.wavelength}")

        return optimal_theta


class VoidSingularity:
    """
    The Void (Absolute Zero Point).
    The gate where noise annihilates and truth inverts.
    """

    def __init__(self, extinction_threshold: float = 0.95):
        self.threshold = extinction_threshold
        logger.info("  Void Singularity opened.")

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
