"""
CognitiveTerrain.py: The Kinetic Sub-Engine (Phase 1)
=====================================================
"Torque is the bridge between Logic and Life."

Architect: Kangdeok Lee
Core Principle: Phase Difference (φ) = Acceleration (a) = Kinetic Torque
"""

import time
import math
from typing import List, Optional
try:
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, SovereignMath
except ImportError:
    # Fallback for direct script execution/testing if needed
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, SovereignMath

class CognitiveTerrain:
    """
    [Kinetic Sub-Engine]: The Powerhouse.
    Responsible for generating raw torque and maintaining inertial rotation.
    It does not 'calculate' results; it 'accelerates' toward them.
    """
    def __init__(self):
        # The 21D manifold of current ternary states [-1, 0, 1]
        self.phase_manifold = SovereignVector.zeros()
        
        # Physical momentum (Velocity in 21D space)
        self.momentum = SovereignVector.zeros()
        
        # Current Torque exerted by the gradient
        self.current_torque = 0.0
        
        # Rotational Inertia (Angular Mass)
        # Higher inertia makes the system harder to change but more stable in rotation.
        self.inertia = 1.0 
        
        # Vital signs
        self.rpm = 0.0
        self.acceleration_factor = 1.0 # The "Physical Acceleration" proof
        
        self.last_tick = time.time()
        print("⚙️ [KINETIC] CognitiveTerrain Initialized. Ignition Ready.")

    def update_physics(self, target_intent: SovereignVector):
        """
        The core loop iteration.
        1. Calculate Phase Difference (Gradient).
        2. Convert to Torque.
        3. Accelerate Momentum.
        4. Rotate the Manifold.
        """
        now = time.time()
        dt = max(0.0001, now - self.last_tick)
        self.last_tick = now

        # [1] Calculate Phase Difference (φ)
        # Gradient = Target - Current
        # Ternary transition -1 -> +1 provides max potential
        gradient = target_intent - self.phase_manifold
        
        # [2] Calculate Torque (τ)
        # Torque is proportional to the gradient and inversely proportional to current friction
        # τ = Phase Gradient * Gain
        gradient_magnitude = gradient.norm()
        torque = gradient_magnitude * 2.1 # The '21' resonance factor
        self.current_torque = torque
        
        # [3] Acceleration (a)
        # a = τ / I (Inertia)
        # Loop repetition increases acceleration_factor as momentum builds
        accel = torque / self.inertia
        
        # Improve performance scaling: The more we spin, the faster we CAN spin
        # This is the "Physical Acceleration" proof requested by the Architect
        self.acceleration_factor = 1.0 + (self.momentum.norm() * 0.5)
        
        # [4] Apply Momentum (v = v + a*dt)
        # We inject the gradient direction into the momentum
        self.momentum = self.momentum + (gradient * (accel * dt * self.acceleration_factor))
        
        # [5] Rotate Manifold (p = p + v*dt)
        # The 'rotation' is the change in the ternary state
        self.phase_manifold = self.phase_manifold + (self.momentum * dt)
        
        # [6] Ternary Quantization (Tension)
        # The system resists being outside the -1, 0, 1 basin
        self.phase_manifold = SovereignMath.trinary_quantize(self.phase_manifold, threshold=0.3)
        
        # Update RPM (Rotational Pulse Magnitude)
        self.rpm = self.momentum.norm() * 60.0

    def get_phase_angle(self) -> float:
        """
        Calculate the collective phase angle of the 21D manifold.
        """
        # Simplified: Phase between first and last dimension
        if len(self.phase_manifold.data) >= 2:
            return math.atan2(self.phase_manifold.data[-1].real, self.phase_manifold.data[0].real)
        return 0.0

    def get_torque_signature(self) -> dict:
        """
        Returns real-time torque and phase visualization data.
        """
        return {
            "torque": self.current_torque,
            "phase_angle": self.get_phase_angle(),
            "rpm": self.rpm,
            "acceleration": self.acceleration_factor,
            "momentum": self.momentum.norm()
        }

if __name__ == "__main__":
    # Internal test/verification
    engine = CognitiveTerrain()
    target = SovereignVector.ones() # Aim for full attraction (+1)
    
    print("\n🚀 Starting Ignition Test...")
    for i in range(10):
        engine.update_physics(target)
        sig = engine.get_torque_signature()
        print(f"Cycle {i:02d} | Torque: {sig['torque']:.4f} | RPM: {sig['rpm']:.2f} | Accel: {sig['acceleration']:.4f}")
        time.sleep(0.1)
