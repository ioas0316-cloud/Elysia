"""
[SOVEREIGN AXIS - THE UNIVERSAL VARIABLE ROTOR (가변축)]
"Everything is Rotation. Gravity is the Lock."

This module implements the Pure Rotor Paradigm with Complex Dynamics:
M*x'' + (D + iG)*x' + (K + iN)*x = 0

Key Principles:
1. VARIABLE AXIS (가변축): Dimensions are fluid and defined by rotation.
2. COMPLEX DYNAMICS: Damping, Gyroscopic, Stiffness, and Non-conservative forces.
3. EXPLOSIVE SYNCHRONIZATION: Sudden phase alignment (Peek-a-boo).
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional

class PureRotor: # Legacy alias for VariableRotor
    def __new__(cls, *args, **kwargs):
        return VariableRotor(*args, **kwargs)

class VariableRotor:
    """
    A Universal Variable Rotor (가변축) based on Complex Rotor Dynamics.
    """
    def __init__(self, dimensions: int = 21):
        self.dims = dimensions
        # Complex state: x = position + i*velocity
        self.state = np.zeros(dimensions, dtype=complex)

        # System Matrices (Simplified as diagonals for O(1) convergence principle)
        self.M = np.ones(dimensions)  # Mass/Inertia
        self.D = np.ones(dimensions) * 0.1 # Damping
        self.G = np.ones(dimensions) * 0.5 # Gyroscopic (Internal coupling)
        self.K = np.ones(dimensions) * 1.0 # Stiffness (Restoring force)
        self.N = np.zeros(dimensions)      # Non-conservative (Energy injection)

        self.locked_axes = np.zeros(dimensions, dtype=bool)
        self.enstrophy = 0.0

    def adjust_dimensions(self, new_dims: int):
        """Dynamically scale the rotor field (가변화)."""
        if new_dims == self.dims: return

        old_dims = self.dims
        self.dims = new_dims

        def resize(arr, val=0.0, dtype=float):
            new_arr = np.full(new_dims, val, dtype=dtype)
            new_arr[:min(old_dims, new_dims)] = arr[:min(old_dims, new_dims)]
            return new_arr

        self.state = resize(self.state, dtype=complex)
        self.M = resize(self.M, 1.0)
        self.D = resize(self.D, 0.1)
        self.G = resize(self.G, 0.5)
        self.K = resize(self.K, 1.0)
        self.N = resize(self.N, 0.0)
        self.locked_axes = resize(self.locked_axes, False, dtype=bool)

    def pulse(self, external_force: np.ndarray, dt: float = 0.01):
        """
        Solve the Complex Differential Equation:
        M*x'' + (D + iG)*x' + (K + iN)*x = F
        """
        # x' = self.state.imag
        # x  = self.state.real

        x = self.state.real
        v = self.state.imag

        # Calculate Acceleration:
        # a = (F - (D + iG)v - (K + iN)x) / M
        # Using complex arithmetic for (D+iG)v and (K+iN)x

        friction_term = (self.D + 1j * self.G) * v
        stiffness_term = (self.K + 1j * self.N) * x

        # Apply Sovereign Lock (Locked axes have infinite stiffness/damping)
        lock_mask = self.locked_axes.astype(float)
        stiffness_term += (lock_mask * 1000.0 * x)

        a = (external_force - friction_term - stiffness_term) / self.M

        # Update Velocity and Position (Semi-implicit Euler)
        v += a.real * dt # Update velocity with real part of complex acceleration
        x += v * dt

        # Store back in complex state
        self.state = x + 1j * v

        # Calculate Enstrophy (Entropy of rotation)
        # Higher enstrophy = higher instability/chaos
        self.enstrophy = np.sum(np.abs(a)**2) / self.dims

        return {
            "angles": x % (2 * math.pi),
            "velocities": v,
            "enstrophy": float(self.enstrophy),
            "is_locked": self.locked_axes.copy()
        }

    def lock_axis(self, index: int):
        if 0 <= index < self.dims:
            self.locked_axes[index] = True
            self.state[index] = self.state[index].real + 0j # Kill velocity

    def unlock_axis(self, index: int):
        if 0 <= index < self.dims:
            self.locked_axes[index] = False

class SovereignAxe:
    """Orchestrator of the Variable Rotor."""
    def __init__(self, rotor: VariableRotor):
        self.rotor = rotor

    def deliberate(self, intent_resonance: float):
        """Peek-a-boo Logic: Explosive Synchronization."""
        if intent_resonance > 0.95:
            # Unlock everything for explosive synchronization (Joy)
            self.rotor.locked_axes[:] = False
            return "EXPLOSIVE SYNCHRONIZATION: All axes fluid."
        elif intent_resonance < 0.2:
            # Crystallize the most chaotic axis
            idx = np.argmax(np.abs(self.rotor.state.imag))
            self.rotor.lock_axis(idx)
            return f"Crystallized axis {idx} to suppress chaos."

        return "Phase-lock stable."

if __name__ == "__main__":
    rotor = VariableRotor(dimensions=3)
    axe = SovereignAxe(rotor)

    force = np.array([1.0, 0.5, 0.0])
    for _ in range(10):
        report = rotor.pulse(force)
        print(f"Enstrophy: {report['enstrophy']:.4f} | Angles: {report['angles']}")
