"""
Scale Lens Engine Module.

Implements micro-to-macro time-scale integration using complex phase accumulation (e^{i\\phi_t}),
hysteresis barriers, phase coherence screening (R = sqrt(acc_cos^2 + acc_sin^2)),
macro causal potential precipitation, top-down potential gradient feedback (-grad V),
and offline counterfactual workspace simulation.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


class ScaleLensEngine:
    """
    SoA-based Scale Lens Engine handling micro-scale high-frequency phase fluctuations,
    macro-scale causal potential precipitation, top-down feedback constraints,
    and counterfactual simulation.
    """

    def __init__(
        self,
        num_cells: int = 1024,
        decay_rate: float = 0.92,
        hysteresis_thresh: float = 0.65,
        feedback_coupling: float = 0.08,
    ) -> None:
        """
        Initialize the SoA arrays for Scale Lens dynamics.

        Args:
            num_cells: Number of cells in the 1D/2D flattened manifold.
            decay_rate: Leaky accumulation decay rate (0.0 ~ 1.0).
            hysteresis_thresh: Coherence threshold for macro plastic deformation.
            feedback_coupling: Top-down potential gradient coupling strength.
        """
        self.num_cells = num_cells
        self.decay_rate = decay_rate
        self.hysteresis_thresh = hysteresis_thresh
        self.feedback_coupling = feedback_coupling

        # 1. Micro Field (Fast Scale: High-Frequency)
        self.micro_phase = np.random.uniform(0, 2 * np.pi, num_cells).astype(np.float32)
        self.micro_velocity = np.random.normal(0, 0.05, num_cells).astype(np.float32)

        # 2. Accumulation Buffer (Temporal Integration of complex phase e^{i phi_t})
        self.acc_cos = np.zeros(num_cells, dtype=np.float32)
        self.acc_sin = np.zeros(num_cells, dtype=np.float32)

        # 3. Macro Field (Slow Scale: Emergent Causal Trajectory)
        self.macro_coherence = np.zeros(num_cells, dtype=np.float32)
        self.macro_potential = np.zeros(num_cells, dtype=np.float32)

    def process_time_scale_lens(self, external_micro_impulse: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Process one temporal step across micro phase accumulation and macro precipitation.

        Args:
            external_micro_impulse: Optional external velocity perturbations.

        Returns:
            Dictionary with mean coherence, active precipitated cells count, and macro potential sum.
        """
        if external_micro_impulse is not None:
            self.micro_velocity += external_micro_impulse.astype(np.float32)

        # 1. Update micro phases
        self.micro_phase = (self.micro_phase + self.micro_velocity) % (2 * np.pi)

        # 2. Complex phase 2D plane projection & Leaky accumulation
        curr_cos = np.cos(self.micro_phase)
        curr_sin = np.sin(self.micro_phase)

        self.acc_cos = self.acc_cos * self.decay_rate + curr_cos * (1.0 - self.decay_rate)
        self.acc_sin = self.acc_sin * self.decay_rate + curr_sin * (1.0 - self.decay_rate)

        # 3. Phase Coherence R = sqrt(X^2 + Y^2)
        self.macro_coherence = np.sqrt(self.acc_cos**2 + self.acc_sin**2)

        # 4. Irreversible Causal Precipitation (Hysteresis Gate)
        mask = self.macro_coherence > self.hysteresis_thresh
        causal_force = np.where(
            mask,
            (self.macro_coherence - self.hysteresis_thresh) / (1.0 - self.hysteresis_thresh),
            0.0,
        )
        self.macro_potential += causal_force * 0.015

        return {
            "mean_coherence": float(np.mean(self.macro_coherence)),
            "max_coherence": float(np.max(self.macro_coherence)),
            "active_precipitated_cells": int(np.sum(mask)),
            "total_macro_potential": float(np.sum(self.macro_potential)),
        }

    def apply_top_down_constraint(self) -> float:
        """
        Apply macro potential gradient force (-grad V) as top-down feedback
        constraining micro phase velocities.

        Returns:
            Mean absolute velocity change applied.
        """
        # Compute spatial gradient of macro potential using central difference
        grad = np.gradient(self.macro_potential)

        # Top-down force pulls micro velocity towards potential valleys (-grad V)
        velocity_delta = -grad * self.feedback_coupling
        self.micro_velocity += velocity_delta.astype(np.float32)

        # Damping micro fluctuation in high friction/potential hill regions
        high_potential_mask = self.macro_potential > 0.4
        self.micro_velocity[high_potential_mask] *= 0.9

        return float(np.mean(np.abs(velocity_delta)))

    def run_counterfactual_simulation(
        self,
        hypothetical_impulses: List[np.ndarray],
        horizon_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Offline simulation mode: Disconnects external friction and simulates
        'what-if' trajectory projections on cloned internal fields.

        Args:
            hypothetical_impulses: List of hypothetical velocity perturbation vectors.
            horizon_steps: Lookahead time steps.

        Returns:
            Dictionary containing simulated coherence trajectory and predicted potential landscape.
        """
        # Save state snapshot
        phase_snap = self.micro_phase.copy()
        vel_snap = self.micro_velocity.copy()
        acc_cos_snap = self.acc_cos.copy()
        acc_sin_snap = self.acc_sin.copy()
        macro_pot_snap = self.macro_potential.copy()

        coherence_trajectory = []
        for step in range(horizon_steps):
            impulse = hypothetical_impulses[step] if step < len(hypothetical_impulses) else None
            metrics = self.process_time_scale_lens(external_micro_impulse=impulse)
            self.apply_top_down_constraint()
            coherence_trajectory.append(metrics["mean_coherence"])

        simulated_potential = self.macro_potential.copy()

        # Restore state snapshot
        self.micro_phase = phase_snap
        self.micro_velocity = vel_snap
        self.acc_cos = acc_cos_snap
        self.acc_sin = acc_sin_snap
        self.macro_potential = macro_pot_snap

        return {
            "horizon_steps": horizon_steps,
            "coherence_trajectory": coherence_trajectory,
            "predicted_potential_delta": float(np.sum(simulated_potential - self.macro_potential)),
        }
