import numpy as np
from typing import Optional, List, Dict, Any

class PhaseSpaceNode:
    """
    [Phase Space Node + @]
    Represents a continuous phase-space causal node containing:
    - Known state / position (S_t)
    - Phase differential trajectory (dS/dt)
    - Initial state reference (S_0)
    - Core potential energy (U_core)
    - Open Latent Valence Vector (+@) representing unclosed causal potential
    - Physical field parameters (Radius, Intensity, Decay Gamma)
    """

    def __init__(
        self,
        node_id: int,
        position: np.ndarray,
        latent_valence: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        core_potential: float = 1.0,
        field_radius: float = 5.0,
        field_intensity: float = 1.0,
        decay_gamma: float = 0.5,
    ):
        self.node_id = node_id
        self.position = np.array(position, dtype=np.float64)
        self.initial_position = self.position.copy()

        if velocity is None:
            self.velocity = np.zeros_like(self.position, dtype=np.float64)
        else:
            self.velocity = np.array(velocity, dtype=np.float64)

        self.core_potential = float(core_potential)

        # Open Latent Valence Vector (+@)
        self.latent_valence = np.array(latent_valence, dtype=np.float64)

        # Physical field properties
        self.field_radius = float(field_radius)
        self.field_intensity = float(field_intensity)
        self.decay_gamma = float(decay_gamma)

        # Dynamic accumulated state
        self.tension_force = np.zeros_like(self.position, dtype=np.float64)
        self.total_interference = 0.0

    @property
    def state_trajectory(self) -> Dict[str, np.ndarray]:
        """Returns the continuous phase space trajectory tuple (S_0, S_t, ΔS)."""
        return {
            "S_0": self.initial_position,
            "S_t": self.position,
            "delta_S": self.velocity,
        }

    def update_physics(self, dt: float = 0.01, damping: float = 0.1):
        """
        Updates continuous phase space dynamics:
        dS/dt = velocity
        d(velocity)/dt = tension_force - damping * velocity
        """
        accel = self.tension_force - damping * self.velocity
        self.velocity += accel * dt
        self.position += self.velocity * dt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "latent_valence": self.latent_valence.tolist(),
            "core_potential": self.core_potential,
            "total_interference": self.total_interference,
            "tension_force": self.tension_force.tolist(),
        }
