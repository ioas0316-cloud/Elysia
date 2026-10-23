import numpy as np
from typing import Dict, Any, List, Tuple

class MachineInternalWorld:
    """
    [Machine Internal World - Minimal Toy Domain & Primitive Operators]
    Represents the machine's primary internal physical dynamics prior to external language injection.

    Features:
    1. Minimal Toy Domain: Low-dimensional continuous state space (alpha, beta) with physical/computational friction,
       reluctance hysteresis, and state inertia.
    2. Primitive Exploration Operators:
       - 'push_against_resistance': Applies drive vector u to push state against reluctance & boundary friction.
       - 'tune_frequency': Modulates internal rotor/oscillator frequency to search for resonance.
       - 'probe_friction': Samples local topological friction and impedance.
    """
    def __init__(self, state_dim: int = 2, base_reluctance: float = 0.5):
        self.state_dim = state_dim
        # Internal state (alpha, beta, ...)
        self.state = np.zeros(state_dim, dtype=np.float64)
        self.velocity = np.zeros(state_dim, dtype=np.float64)
        self.frequency = 1.0  # Internal oscillator frequency
        self.phase = 0.0

        # Physical constraints & Hysteresis parameters
        self.base_reluctance = base_reluctance
        self.reluctance_field = np.ones(state_dim, dtype=np.float64) * base_reluctance
        self.hysteresis_memory = np.zeros(state_dim, dtype=np.float64)
        self.boundary_limits = np.ones(state_dim, dtype=np.float64) * 5.0

        # Metrics
        self.accumulated_friction = 0.0
        self.last_impedance = 0.0
        self.history: List[Dict[str, Any]] = []

    def push_against_resistance(self, drive_vector: np.ndarray, dt: float = 0.05) -> Dict[str, Any]:
        """
        Primitive Operator 1: Push against internal state reluctance & boundary friction.
        """
        drive = np.asarray(drive_vector, dtype=np.float64)
        if drive.shape[0] != self.state_dim:
            raise ValueError(f"Drive vector dimension mismatch. Expected {self.state_dim}, got {drive.shape[0]}")

        # 1. Effective reluctance with hysteresis
        effective_reluctance = self.reluctance_field + 0.3 * np.tanh(self.hysteresis_memory)

        # 2. Reluctance friction force based on drive and current velocity
        reluctance_force = -effective_reluctance * (self.velocity + 0.1 * drive)

        # Boundary restore / friction if approaching boundary_limits
        boundary_friction = np.zeros_like(self.state)
        for i in range(self.state_dim):
            if abs(self.state[i]) > self.boundary_limits[i] * 0.8:
                overflow = abs(self.state[i]) - self.boundary_limits[i] * 0.8
                boundary_friction[i] = -np.sign(self.state[i]) * overflow * 10.0

        net_force = drive + reluctance_force + boundary_friction

        # 3. Acceleration and integration
        acceleration = net_force / 1.0  # Unit mass
        self.velocity += acceleration * dt
        self.state += self.velocity * dt

        # 4. Update hysteresis memory (lagging response)
        self.hysteresis_memory = 0.9 * self.hysteresis_memory + 0.1 * self.velocity

        # 5. Calculate instantaneous friction & impedance
        instant_friction = float(np.linalg.norm(reluctance_force + boundary_friction))
        impedance = float(np.dot(effective_reluctance, np.abs(self.velocity)) + instant_friction)

        self.accumulated_friction += instant_friction * dt
        self.last_impedance = impedance

        result = {
            "state": self.state.copy(),
            "velocity": self.velocity.copy(),
            "instant_friction": instant_friction,
            "impedance": impedance,
            "reluctance": effective_reluctance.copy()
        }
        self.history.append(result)
        return result

    def tune_frequency(self, delta_freq: float, dt: float = 0.05) -> Dict[str, Any]:
        """
        Primitive Operator 2: Adjust oscillator frequency seeking resonance.
        """
        self.frequency = max(0.1, min(10.0, self.frequency + delta_freq))
        self.phase = (self.phase + 2.0 * np.pi * self.frequency * dt) % (2.0 * np.pi)

        # Oscillator coupling to state
        oscillation = np.sin(self.phase) * 0.2
        self.state[0] += oscillation * dt

        return {
            "frequency": self.frequency,
            "phase": self.phase,
            "state": self.state.copy()
        }

    def probe_friction(self) -> Dict[str, float]:
        """
        Primitive Operator 3: Sample local topological friction.
        """
        return {
            "accumulated_friction": self.accumulated_friction,
            "last_impedance": self.last_impedance,
            "average_reluctance": float(np.mean(self.reluctance_field))
        }

    def reset_state(self):
        self.state.fill(0.0)
        self.velocity.fill(0.0)
        self.accumulated_friction = 0.0
        self.history.clear()
