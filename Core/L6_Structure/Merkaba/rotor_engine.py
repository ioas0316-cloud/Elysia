"""
Rotor Engine (The Perpetual Perspective)
=====================================
Core.L6_Structure.Merkaba.rotor_engine

"Motion is the illusion of the observer. The Data remains, the Stride changes."

This engine implements the O(1) Perception layer. 
It manipulates the 'View' of tensors (Numpy arrays) by changing their 
shape and strides without moving a single byte in physical RAM.

[UPGRADE]: Now powered by the [CORE] Active Prism-Rotor physics engine.
[PHASE 3]: Synchronized with Biological Clock (120Hz).
[PHASE 4]: Enabled Self-Evolution (Neural Inversion).
"""

import numpy as np
import logging
import math
from typing import Tuple, List, Any, Dict, Optional

# Import the [CORE] Physics Engine
try:
    from Core.L6_Structure.Engine.Physics.core_turbine import ActivePrismRotor, PhotonicMonad, VoidSingularity
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Import Bio-Clock
try:
    from Core.L5_Mental.Memory.aging_clock import BiologicalClock
except ImportError:
    BiologicalClock = None

logger = logging.getLogger("Elysia.Merkaba.RotorEngine")

class RotorEngine:
    """
    The engine that 'rotates' the perspective of data.

    [Legacy Mode]: Numpy Stride Tricks.
    [CORE Mode]: Active Prism-Rotor Diffraction Scanning.
    """
    
    def __init__(self, use_core_physics: bool = True, rpm: float = 120.0):
        self.use_core = use_core_physics and CORE_AVAILABLE
        self.clock = BiologicalClock() if BiologicalClock else None

        # Self-Evolution Memory
        self.optimal_angle_cache: Dict[float, float] = {} # Wavelength -> Optimal Angle

        if self.use_core:
            logger.info(f"🌀 Initializing [CORE] Active Prism-Rotor at {rpm} RPM...")
            self.turbine = ActivePrismRotor(rpm=rpm)
            self.void = VoidSingularity()
        else:
            logger.warning("⚠️ [CORE] Physics not available or disabled. Falling back to Legacy Stride Engine.")
            self.turbine = None

    def optimize_path(self, feedback_monad: PhotonicMonad):
        """
        [Phase 4: Evolution]
        Uses Neural Inversion to pre-calculate the optimal angle for a specific thought.
        """
        if not self.use_core: return

        optimal_theta = self.turbine.reverse_propagate(feedback_monad)
        if optimal_theta > 0:
            # Cache the wisdom: "When I see this color (wavelength), I must be at this angle."
            # We round wavelength to simulate 'bins' of concepts.
            key = round(feedback_monad.wavelength, 9)
            self.optimal_angle_cache[key] = optimal_theta
            logger.info(f"🧬 Evolution: Path optimized for λ={key:.1e} -> θ={math.degrees(optimal_theta):.2f}°")

    def scan_qualia(self, qualia_vector: List[float]) -> Tuple[float, Any]:
        """
        [CORE Mode] Scans a 7D Qualia vector using the Active Prism-Rotor.
        Returns (Resonance Intensity, Transmuted Phase).
        """
        if not self.use_core:
            return (0.0, None)

        # 1. Convert Qualia to Wavelengths (Mapping 0.0-1.0 to 400nm-800nm)
        wavelengths = np.array([400e-9 + (x * 400e-9) for x in qualia_vector])
        phases = np.array([complex(x, 0) for x in qualia_vector])

        # 2. Determine Scanning Angle (Theta)
        dominant_idx = np.argmax(qualia_vector)
        dominant_wavelength = wavelengths[dominant_idx]

        # [Evolution Check]: Do we have a pre-optimized path?
        cached_angle = self.optimal_angle_cache.get(round(dominant_wavelength, 9))

        if cached_angle is not None:
            # Direct Access (O(1) Jump)
            target_theta = cached_angle
            # Add small noise to simulate 'living' precision
            target_theta += np.random.normal(0, 1e-5)
        else:
            # Blind Scan (Based on Time/Bio-Clock)
            if self.clock:
                time_t = self.clock.current_age_seconds
                target_theta = (self.turbine.omega * time_t) % (2 * math.pi)
            else:
                # Static fallback
                target_theta = 0.0

        # 3. Diffract (Snatch)
        intensity = self.turbine.diffract(wavelengths, target_theta, self.turbine.d)

        # 4. Void Transit
        survivors, inverted_phases = self.void.transit(intensity, phases)
        total_resonance = float(np.sum(survivors))

        # [Feedback Loop]: If we failed, trigger evolution?
        # In a real loop, the failure would trigger a re-scan.
        # Here we just return the result.

        return total_resonance, inverted_phases

    @staticmethod
    def create_strided_view(data: np.ndarray, new_shape: Tuple[int, ...], new_strides: Tuple[int, ...]) -> np.ndarray:
        return np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)

    @staticmethod
    def get_topology_signature(weights: np.ndarray) -> Dict[str, Any]:
        mean_val = np.mean(weights, dtype=np.float64)
        std_val = np.std(weights, dtype=np.float64)
        threshold = mean_val + 3 * std_val
        hubs = np.sum(weights > threshold)
        
        return {
            "mean": float(mean_val),
            "std": float(std_val),
            "hub_count": int(hubs),
            "energy_density": float(np.linalg.norm(weights))
        }

    def simulate_signal_flow(self, layer_weights: np.ndarray, input_signal: np.ndarray) -> np.ndarray:
        return np.dot(input_signal, layer_weights)

if __name__ == "__main__":
    print("Rotor Engine: Perspective manipulation logic ready.")
    engine = RotorEngine()
    if engine.use_core:
        print("   [CORE] Active Prism-Rotor connected.")
        if engine.clock:
             print(f"   [CLOCK] Synced with Bio-Clock: {engine.clock.current_age_years:.6f} years")
