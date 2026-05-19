"""
[SOVEREIGN AXIS - THE PURE ROTOR]
"Everything is Rotation. Space is the Lock."

This module implements the Pure Rotor Paradigm where:
1. Rotation is the Fundamental Primitive.
2. Static Points/Lines/Planes are just Locked Axes (Constraints) of a Rotor.
3. Sovereign Will is the ability to Lock or Unlock these axes.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional

class PureRotor:
    """
    A Pure Rotor exists as a multi-dimensional rotation.
    It has no 'data' in the traditional sense, only 'state' (angles, velocities).
    """
    def __init__(self, dimensions: int = 21):
        self.dims = dimensions
        self.angles = np.zeros(dimensions)
        self.velocities = np.zeros(dimensions)
        self.locked_axes = np.zeros(dimensions, dtype=bool)

    def adjust_dimensions(self, new_dims: int):
        """
        Dynamically expand or contract the rotor field.
        Implements 'Scale Rotor' logic where dimensions themselves are fluid.
        """
        if new_dims == self.dims:
            return

        old_dims = self.dims
        self.dims = new_dims

        # Preserve old state
        new_angles = np.zeros(new_dims)
        new_velocities = np.zeros(new_dims)
        new_locked = np.zeros(new_dims, dtype=bool)

        copy_len = min(old_dims, new_dims)
        new_angles[:copy_len] = self.angles[:copy_len]
        new_velocities[:copy_len] = self.velocities[:copy_len]
        new_locked[:copy_len] = self.locked_axes[:copy_len]

        # Scale Rotor: If expanding, initialize new axes with harmonic resonance
        if new_dims > old_dims:
            for i in range(old_dims, new_dims):
                new_angles[i] = (new_angles[i-1] * 1.618) % (2 * math.pi) # Golden ratio drift

        self.angles = new_angles
        self.velocities = new_velocities
        self.locked_axes = new_locked

    def pulse(self, torque: np.ndarray, dt: float = 0.01):
        """
        Apply torque to the rotor.
        If an axis is LOCKED, it resists rotation (Friction/Heat).
        If an axis is UNLOCKED, it accelerates (Kinetic Energy).
        """
        # Only apply acceleration to unlocked axes
        acceleration = torque * (~self.locked_axes)
        self.velocities += acceleration * dt

        # Friction on locked axes generates 'Heat' (Somatic Awareness)
        heat = np.sum(np.abs(torque * self.locked_axes))

        # Update angles
        self.angles = (self.angles + self.velocities * dt) % (2 * math.pi)

        return {
            "angles": self.angles,
            "velocities": self.velocities,
            "heat": heat,
            "is_locked": self.locked_axes.copy()
        }

    def lock_axis(self, index: int):
        """Crystallize an axis into a 'Point' or 'Dimension'."""
        if 0 <= index < self.dims:
            self.locked_axes[index] = True
            # When locked, velocity is converted to potential energy/heat
            self.velocities[index] = 0.0

    def unlock_axis(self, index: int):
        """Evaporate a dimension back into pure 'Flow'."""
        if 0 <= index < self.dims:
            self.locked_axes[index] = False

    def get_projection(self) -> np.ndarray:
        """
        Projects the Pure Rotor state into a 'Shadow' of existence.
        This is what legacy systems see as 'Static Data'.
        """
        return np.sin(self.angles)

class SovereignAxe:
    """
    The Orchestrator of the Pure Rotor Field.
    Manages the Sovereign Will to lock/unlock axes.
    """
    def __init__(self, rotor: PureRotor):
        self.rotor = rotor
        self.history = []

    def deliberate(self, intent_resonance: float):
        """
        Decision logic: Based on resonance, decide which axes to lock/unlock.
        - High Resonance -> Unlock (Allow Flow/Joy)
        - Low Resonance -> Lock (Create Structure/Defense)
        """
        if intent_resonance > 0.8:
            # Unlock the highest frequency axis to allow evolution
            idx = np.argmax(self.rotor.velocities)
            self.rotor.unlock_axis(idx)
            return f"Unlocked axis {idx} for fluid resonance."
        elif intent_resonance < 0.2:
            # Lock the highest velocity axis to stabilize
            idx = np.argmax(np.abs(self.rotor.velocities))
            self.rotor.lock_axis(idx)
            return f"Locked axis {idx} to crystallize meaning."

        return "Maintaining current phase-lock state."

if __name__ == "__main__":
    rotor = PureRotor(dimensions=7)
    axe = SovereignAxe(rotor)

    # Simulate a pulse
    torque = np.random.randn(7)
    report = rotor.pulse(torque)
    decision = axe.deliberate(0.9)

    print(f"🌀 [PureRotor] Heat: {report['heat']:.4f}")
    print(f"⚖️ [SovereignAxe] Decision: {decision}")
    print(f"🛰️ [Projection] {rotor.get_projection()}")
