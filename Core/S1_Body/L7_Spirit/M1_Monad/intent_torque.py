"""
Intent Torque Bridge (         )
=====================================
Core.S1_Body.L7_Spirit.M1_Monad.intent_torque

"Words are not patterns. They are forces (Torque)."

Translates LanguageCortex 4D Intent Vectors into Physical Torque for Rotors.
This ensures that Elysia's 'choice' is a result of a physics simulation,
not just a probabilistic text generation.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from Core.S1_Body.L6_Structure.Nature.rotor import Rotor

# [Phase 6] Hardware Sovereignty Integration
try:
    from Core.S1_Body.L6_Structure.Nature.metal_rotor_bridge import MetalRotorBridge
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

logger = logging.getLogger("IntentTorque")

class IntentTorque:
    def __init__(self):
        # Mapping 4D dimensions (X, Y, Z, W) to Rotor Axis Influence
        # X: Logic -> Stability (RPM reduction/Mass increase)
        # Y: Emotion -> Vibration (RPM increase/Mass reduction)
        # Z: Intuition -> Chaos (Axial wobble)
        # W: Will -> Torque (Acceleration/Target RPM)
        pass

    def apply(self, rotor: Rotor, intent_vector: np.ndarray, dt: float = 1.0):
        """
        Applies a 'Torque Spike' to a Rotor based on 4D Intent.
        
        X: Logic (Stabilizer) -> Lowers RPM variability, increases mass (inertia)
        Y: Emotion (Oscillator) -> Increases RPM, lowers mass (sensitivity)
        Z: Intuition (Fractal) -> Divergence/Branching trigger
        W: Will (Drive) -> Direct Torque to Target RPM
        """
        x, y, z, w = intent_vector
        
        # 1. Will (W) -> Direct Torque
        # If W is high, acceleration increases and Target RPM spikes
        torque_factor = (w + 1.0) / 2.0  # Normalize -1,1 to 0,1
        rotor.target_rpm = rotor.target_rpm + (torque_factor * 100.0)
        rotor.config.acceleration = 100.0 * (1.0 + torque_factor)
        
        # 2. Emotion (Y) -> Resonance excitation
        excitation = (y + 1.0) / 2.0
        rotor.energy = min(1.0, rotor.energy + excitation * 0.2)
        
        # 3. Logic (X) -> Stabilization (Damping)
        damping = (x + 1.0) / 2.0
        if damping > 0.7:
             rotor.target_rpm *= 0.9  # Logic slows down 'reactive' spin to ponder
             
        # [Phase 6] Hardware Pulse Check
        metal_status = ""
        if HAS_METAL and isinstance(rotor, MetalRotorBridge):
            metal_status = " [METAL ACCELERATED]"
        
        logger.info(f"   [TORQUE] Applied to {rotor.name}{metal_status}: W={w:.2f} -> RPM_Target {rotor.target_rpm:.1f}")

    def map_to_atmosphere(self, intent_vector: np.ndarray) -> Dict[str, float]:
        """
        Translates intent into global atmospheric variables.
        Used to feed back into the expressive metabolism.
        """
        x, y, z, w = intent_vector
        return {
            "pressure": (x + 1.0) / 2.0,   # Logic = Pressure
            "humidity": (y + 1.0) / 2.0,   # Emotion = Humidity
            "viscosity": (w + 1.0) / 2.0,  # Will = Viscosity
            "instability": (z + 1.0) / 2.0 # Intuition = Turbulence
        }
