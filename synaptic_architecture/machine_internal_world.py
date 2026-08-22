"""
Machine Internal World Module.

Implements the internal topological field of the machine before external symbol injection.
Operates on continuous state space with reluctance hysteresis, boundary friction,
homeostatic drive, entropic decay, epistemic telos, and multi-field heterogeneous sensor collision.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class MachineInternalWorld:
    """
    Represents the Machine Internal World (2D or N-D topological field)
    working on minimal primitive dynamics.
    """

    def __init__(
        self,
        grid_size: int = 32,
        reluctance_coeff: float = 0.15,
        friction_coeff: float = 0.2,
        decay_rate: float = 0.02,
        homeostatic_target: float = 0.5,
    ) -> None:
        """
        Initialize the machine internal world state space.

        Args:
            grid_size: Spatial dimensions for the 2D alpha-beta field.
            reluctance_coeff: Reluctance hysteresis coefficient.
            friction_coeff: Boundary friction coefficient.
            decay_rate: Entropic decay rate over time.
            homeostatic_target: Target structural integrity / potential level.
        """
        self.grid_size = grid_size
        self.reluctance_coeff = reluctance_coeff
        self.friction_coeff = friction_coeff
        self.decay_rate = decay_rate
        self.homeostatic_target = homeostatic_target

        # 2D Topological space coordinates (alpha, beta) in range [-1.0, 1.0]
        alpha = np.linspace(-1.0, 1.0, grid_size)
        beta = np.linspace(-1.0, 1.0, grid_size)
        self.alpha_grid, self.beta_grid = np.meshgrid(alpha, beta)

        # Internal potential field state V(alpha, beta)
        self.potential_field = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Hysteresis remanence state S_r(alpha, beta)
        self.remanence_state = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Current state position in (alpha, beta)
        self.current_pos = np.array([0.0, 0.0], dtype=np.float32)

        # Heterogeneous Sensor Fields
        self.spatial_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.temporal_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.somatic_field = np.ones((grid_size, grid_size), dtype=np.float32) * homeostatic_target

        # Epistemic Telos & Experience History
        self.accumulated_resonance = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.preference_valleys = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.internal_entropy = 0.1

    def step_entropic_decay(self) -> float:
        """Apply entropic decay to the internal field and compute current entropy."""
        self.potential_field -= self.decay_rate * self.potential_field
        self.internal_entropy = float(np.var(self.potential_field))
        return self.internal_entropy

    def apply_homeostatic_drive(self) -> float:
        """
        Calculate homeostatic pressure driving the system towards optimal potential.
        Returns homeostatic imbalance error.
        """
        current_mean_somatic = float(np.mean(self.somatic_field))
        error = abs(current_mean_somatic - self.homeostatic_target)
        # Entropic dissipation
        self.potential_field += self.decay_rate * (self.homeostatic_target - self.potential_field)
        self.internal_entropy = float(np.var(self.potential_field) + error)
        return error

    def push_against_resistance(self, delta_alpha: float, delta_beta: float) -> Tuple[float, float]:
        """
        Primitive operator: Push state against reluctance hysteresis and boundary friction.

        Returns:
            Tuple of (effective_movement_norm, friction_encountered)
        """
        target_pos = self.current_pos + np.array([delta_alpha, delta_beta], dtype=np.float32)
        target_pos = np.clip(target_pos, -1.0, 1.0)

        # Map position to grid indices
        ix = int((target_pos[0] + 1.0) / 2.0 * (self.grid_size - 1))
        iy = int((target_pos[1] + 1.0) / 2.0 * (self.grid_size - 1))

        # Reluctance hysteresis effect
        reluctance = float(self.remanence_state[iy, ix] * self.reluctance_coeff)
        gradient_x, gradient_y = np.gradient(self.potential_field)
        local_grad = np.array([gradient_x[iy, ix], gradient_y[iy, ix]], dtype=np.float32)

        # Movement against gradient and reluctance causes friction
        input_vec = np.array([delta_alpha, delta_beta], dtype=np.float32)
        friction = float(np.dot(input_vec, local_grad) + abs(reluctance) * self.friction_coeff)
        friction = float(max(0.01, friction))

        # Update position damped by friction
        movement = input_vec / (1.0 + friction)
        self.current_pos = np.clip(self.current_pos + movement, -1.0, 1.0)

        # Plastic deformation of remanence state (hysteresis)
        self.remanence_state[iy, ix] += float(0.1 * np.linalg.norm(movement))
        self.remanence_state = np.clip(self.remanence_state, -1.0, 1.0)

        return float(np.linalg.norm(movement)), friction

    def tune_frequency(self, frequency: float, phase: float) -> float:
        """
        Primitive operator: Inject temporal frequency into the temporal sensor field.

        Returns:
            Resonance score with internal potential field.
        """
        wave = np.sin(frequency * self.alpha_grid + phase) * np.cos(frequency * self.beta_grid + phase)
        self.temporal_field = wave.astype(np.float32)

        # Coherence with potential field
        overlap = np.mean(self.temporal_field * self.potential_field)
        resonance = float(1.0 / (1.0 + np.exp(-overlap)))

        # Update preference valleys (Path-dependent Telos)
        if resonance > 0.5:
            self.accumulated_resonance += (wave * (resonance - 0.5)).astype(np.float32)
            self.preference_valleys -= 0.05 * (self.accumulated_resonance)
            self.preference_valleys = np.clip(self.preference_valleys, -2.0, 2.0)

        return resonance

    def probe_friction(self, external_signal: np.ndarray) -> Dict[str, float]:
        """
        Primitive operator: Probe cross-modal friction between external signal,
        spatial, temporal, and somatic fields.

        Returns:
            Dictionary containing spatial_friction, temporal_friction,
            cross_modal_friction, and total_impedance.
        """
        if external_signal.shape != (self.grid_size, self.grid_size):
            signal = np.resize(external_signal, (self.grid_size, self.grid_size)).astype(np.float32)
        else:
            signal = external_signal.astype(np.float32)

        # Update spatial field
        self.spatial_field = 0.8 * self.spatial_field + 0.2 * signal

        # Heterogeneous Sensor Collision (Cross-Modal Friction)
        spatial_temp_diff = np.mean(np.abs(self.spatial_field - self.temporal_field))
        somatic_stress = np.mean(np.abs(self.somatic_field - self.homeostatic_target))

        cross_modal_friction = float(spatial_temp_diff * 0.6 + somatic_stress * 0.4)
        total_impedance = float(cross_modal_friction + np.mean(self.remanence_state) * self.reluctance_coeff)

        return {
            "spatial_friction": float(np.mean(np.abs(self.spatial_field - signal))),
            "temporal_friction": float(spatial_temp_diff),
            "somatic_stress": float(somatic_stress),
            "cross_modal_friction": cross_modal_friction,
            "total_impedance": total_impedance,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return the current internal state dictionary."""
        return {
            "current_pos": self.current_pos.tolist(),
            "mean_remanence": float(np.mean(self.remanence_state)),
            "mean_potential": float(np.mean(self.potential_field)),
            "internal_entropy": self.internal_entropy,
            "mean_preference": float(np.mean(self.preference_valleys)),
        }
