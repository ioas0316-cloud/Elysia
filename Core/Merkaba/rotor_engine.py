"""
Rotor Engine (The Perpetual Perspective)
=====================================
Core.Merkaba.rotor_engine

"Motion is the illusion of the observer. The Data remains, the Stride changes."

This engine implements the O(1) Perception layer. 
It manipulates the 'View' of tensors (Numpy arrays) by changing their 
shape and strides without moving a single byte in physical RAM.

[UPGRADE]: Now powered by the [CORE] Active Prism-Rotor physics engine.
"""

import numpy as np
import logging
from typing import Tuple, List, Any, Dict, Optional

# Import the [CORE] Physics Engine
try:
    from Core.Engine.Physics.core_turbine import ActivePrismRotor, PhotonicMonad, VoidSingularity
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = logging.getLogger("Elysia.Merkaba.RotorEngine")

class RotorEngine:
    """
    The engine that 'rotates' the perspective of data.

    [Legacy Mode]: Numpy Stride Tricks.
    [CORE Mode]: Active Prism-Rotor Diffraction Scanning.
    """
    
    def __init__(self, use_core_physics: bool = True, rpm: float = 120.0):
        self.use_core = use_core_physics and CORE_AVAILABLE

        if self.use_core:
            logger.info(f"🌀 Initializing [CORE] Active Prism-Rotor at {rpm} RPM...")
            self.turbine = ActivePrismRotor(rpm=rpm)
            self.void = VoidSingularity()
        else:
            logger.warning("⚠️ [CORE] Physics not available or disabled. Falling back to Legacy Stride Engine.")
            self.turbine = None

    def scan_qualia(self, qualia_vector: List[float]) -> Tuple[float, Any]:
        """
        [CORE Mode] Scans a 7D Qualia vector using the Active Prism-Rotor.
        Returns (Resonance Intensity, Transmuted Phase).
        """
        if not self.use_core:
            return (0.0, None)

        # 1. Convert Qualia to Wavelengths (Mapping 0.0-1.0 to 400nm-800nm)
        # This is a conceptual mapping for the simulation.
        # We treat the vector as a signal stream.
        wavelengths = np.array([400e-9 + (x * 400e-9) for x in qualia_vector])

        # 2. Convert Importance/Weights to Phases (Intent)
        # Use magnitude as phase amplitude
        phases = np.array([complex(x, 0) for x in qualia_vector])

        # 3. Spin the Rotor (Scan)
        # We scan at the angle corresponding to the 'Dominant' qualia to check resonance.
        dominant_idx = np.argmax(qualia_vector)
        target_lambda = wavelengths[dominant_idx]

        # Calculate resonant angle: sin(theta) = lambda / d
        # Ensure lambda < d
        if target_lambda > self.turbine.d:
            logger.warning("Wavelength exceeds grating spacing. Diffraction impossible.")
            return (0.0, None)

        target_sin = target_lambda / self.turbine.d
        target_theta = np.arcsin(target_sin)

        # 4. Diffract (Snatch)
        intensity = self.turbine.diffract(wavelengths, target_theta, self.turbine.d)

        # 5. Void Transit
        survivors, inverted_phases = self.void.transit(intensity, phases)

        # Calculate Total Resonance
        total_resonance = float(np.sum(survivors))

        return total_resonance, inverted_phases

    @staticmethod
    def create_strided_view(data: np.ndarray, new_shape: Tuple[int, ...], new_strides: Tuple[int, ...]) -> np.ndarray:
        """
        [The Heart of Global Perseption]
        Creates a new view of the data with a different stride and shape.
        Cost: O(1) - Constant time regardless of data size.
        """
        # This is where we defy the physics of conventional data processing.
        return np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)

    @staticmethod
    def get_topology_signature(weights: np.ndarray) -> Dict[str, Any]:
        """
        Extracts the 'Vibration' (Statistical/Topology signature) of a layer.
        Uses fast Numpy operations on the view.
        """
        # 1. Energy Profile (Using float64 to avoid overflow on massive arrays)
        mean_val = np.mean(weights, dtype=np.float64)
        std_val = np.std(weights, dtype=np.float64)
        
        # 2. Hub Detection (Extreme Outliers)
        # Neurons that have disproportionate influence
        threshold = mean_val + 3 * std_val
        hubs = np.sum(weights > threshold)
        
        return {
            "mean": float(mean_val),
            "std": float(std_val),
            "hub_count": int(hubs),
            "energy_density": float(np.linalg.norm(weights))
        }

    def simulate_signal_flow(self, layer_weights: np.ndarray, input_signal: np.ndarray) -> np.ndarray:
        """
        [PHASE 3: SIMULATOR]
        Simulates signal propagation through a layer.
        For Sovereign mode, we prioritize Sparse/Partial paths.
        """
        # In the future, this will handle Sparse Matrix Multiplications.
        return np.dot(input_signal, layer_weights)

if __name__ == "__main__":
    print("Rotor Engine: Perspective manipulation logic ready.")
    engine = RotorEngine()
    if engine.use_core:
        print("   [CORE] Active Prism-Rotor connected.")
