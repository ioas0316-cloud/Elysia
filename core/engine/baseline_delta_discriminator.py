"""Baseline-Delta Discriminator for Elysia Engine.

Calculates rolling baseline expectations ('sameness') and extracts phase
deformations ('delta'), XOR interference, and standing wave reflection
(Proprioceptive Impedance Mismatch) for self/world boundary discovery.
"""

from typing import Dict, Any, Tuple
import numpy as np


class BaselineDeltaDiscriminator:
    """Detects phase deformations, XOR interference, and impedance standing waves."""

    def __init__(
        self,
        window_size: int = 1024,
        alpha: float = 0.05,
        reflection_threshold: float = 0.8
    ):
        self.window_size = window_size
        self.alpha = alpha  # Exponential rolling update factor for baseline trajectory
        self.reflection_threshold = reflection_threshold

        self.baseline_trajectory: np.ndarray = np.zeros(window_size, dtype=np.float64)
        self.groove_depth: np.ndarray = np.ones(window_size, dtype=np.float64)  # Resistance/groove deepening
        self.intervention_history: np.ndarray = np.zeros(window_size, dtype=np.float64)

    def process_incoming_signal(
        self,
        incoming_wave: np.ndarray,
        intervention_wave: np.ndarray = None
    ) -> Dict[str, Any]:
        """Processes an incoming wave sample against current baseline and active intervention.

        Args:
            incoming_wave: 1D numpy array of signal values (floats or uint8s)
            intervention_wave: Optional 1D numpy array representing active engine output/projection

        Returns:
            Dict containing delta_residual, xor_interference, standing_wave_reflection, and updated_groove
        """
        raw_wave = np.asarray(incoming_wave, dtype=np.float64)
        n = len(raw_wave)

        if n != self.window_size:
            # Resize or tile to match window size for vector operation
            if n > self.window_size:
                raw_wave = raw_wave[: self.window_size]
            else:
                raw_wave = np.pad(raw_wave, (0, self.window_size - n), mode="wrap")

        if intervention_wave is not None:
            active_actuation = np.asarray(intervention_wave, dtype=np.float64)
            if len(active_actuation) != self.window_size:
                if len(active_actuation) > self.window_size:
                    active_actuation = active_actuation[: self.window_size]
                else:
                    active_actuation = np.pad(active_actuation, (0, self.window_size - len(active_actuation)), mode="wrap")
            self.intervention_history = active_actuation
        else:
            active_actuation = self.intervention_history

        # 1. Calculate Phase Delta Residual against rolling Baseline
        delta_residual = raw_wave - self.baseline_trajectory

        # 2. XOR Interference (bit-level/phase opposition check)
        incoming_binary = (raw_wave > np.mean(raw_wave)).astype(np.uint8)
        baseline_binary = (self.baseline_trajectory > np.mean(self.baseline_trajectory)).astype(np.uint8)
        xor_interference = np.bitwise_xor(incoming_binary, baseline_binary)

        # 3. Proprioceptive Impedance Mismatch & Standing Wave Reflection
        # If intervention was applied, evaluate how much returned wave opposes intervention
        # High correlation with inverted intervention indicates standing wave reflection (wall/environmental limit)
        if np.linalg.norm(active_actuation) > 1e-6:
            # Normalized cross-reflection
            act_norm = active_actuation / (np.linalg.norm(active_actuation) + 1e-9)
            inc_norm = raw_wave / (np.linalg.norm(raw_wave) + 1e-9)
            reflection_coeff = np.abs(np.dot(act_norm, inc_norm))

            # Impedance Mismatch: High reflection + high residual delta = unyielding boundary (Standing Wave / Wall)
            standing_wave = (reflection_coeff > self.reflection_threshold) and (np.mean(np.abs(delta_residual)) > 0.1)
        else:
            reflection_coeff = 0.0
            standing_wave = False

        # 4. Groove Deepening (Structural Self-Organization)
        # Deepens grooves where delta residual is consistent, reinforcing baseline stability
        groove_increment = 0.01 / (1.0 + np.abs(delta_residual))
        self.groove_depth += groove_increment

        # Update rolling baseline weighted by groove resistance
        effective_alpha = self.alpha / self.groove_depth
        self.baseline_trajectory = (1.0 - effective_alpha) * self.baseline_trajectory + effective_alpha * raw_wave

        return {
            "delta_residual": delta_residual,
            "mean_delta": float(np.mean(np.abs(delta_residual))),
            "xor_interference": xor_interference,
            "reflection_coefficient": float(reflection_coeff),
            "standing_wave_reflection": bool(standing_wave),
            "is_world_boundary": bool(standing_wave),
            "baseline_trajectory": self.baseline_trajectory.copy(),
            "groove_depth": self.groove_depth.copy(),
        }
