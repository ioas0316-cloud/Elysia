import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class SpectralSpectrumTuner:
    """
    [Spectral Spectrum Tuner (Axis of Freedom)]

    Maps the 'Light and Shadow (명암의 스펙트럼)' continuous gradient onto dynamic topological
    tension and potential field gradients.

    Regulates system autonomy, openness/closure, and boundary permeability based on whether
    external interactions are symbiotic (resonance-inducing) or extractive/hostile (friction-inducing).
    """
    def __init__(self, dim: int = 8, baseline_autonomy: float = 0.5):
        self.dim = dim
        # Autonomy Spectrum [0.0 (Full Light/Openness) to 1.0 (Full Shadow/Sovereignty Defense)]
        self.autonomy_level = baseline_autonomy
        self.tension_vector = np.zeros(dim, dtype=np.float64)
        self.potential_gradient = np.zeros(dim, dtype=np.float64)
        self.boundary_permeability = 1.0 - baseline_autonomy

    def evaluate_external_stimulus(self, stimulus_vector: np.ndarray, expected_harmony: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates incoming stimulus vector against expected harmonic resonance.
        Calculates friction (topological misalignment) vs symbiosis (phase resonance).

        - Hostile/Extractive inputs create high friction and increase autonomy defense (shadow shift).
        - Symbiotic inputs reduce friction and open system boundary (light shift).
        """
        stimulus = np.array(stimulus_vector, dtype=np.float64)
        harmony = np.array(expected_harmony, dtype=np.float64)

        if stimulus.shape[0] != self.dim:
            if stimulus.shape[0] < self.dim:
                stimulus = np.pad(stimulus, (0, self.dim - stimulus.shape[0]))
            else:
                stimulus = stimulus[:self.dim]

        if harmony.shape[0] != self.dim:
            if harmony.shape[0] < self.dim:
                harmony = np.pad(harmony, (0, self.dim - harmony.shape[0]))
            else:
                harmony = harmony[:self.dim]

        # Calculate phase misalignment (friction) and dot-product resonance (symbiosis)
        diff = stimulus - harmony
        friction = float(np.linalg.norm(diff))

        norm_s = np.linalg.norm(stimulus)
        norm_h = np.linalg.norm(harmony)
        if norm_s > 1e-6 and norm_h > 1e-6:
            symbiosis = float(np.dot(stimulus, harmony) / (norm_s * norm_h))
        else:
            symbiosis = 0.0

        # Dynamic Autonomy Shift along Light/Shadow Axis:
        # High friction shifts autonomy towards Shadow defense (increases sovereignty curvature).
        # High symbiosis shifts autonomy towards Light openness.
        net_shift = (friction * 0.2) - (max(0.0, symbiosis) * 0.15)
        self.autonomy_level = float(np.clip(self.autonomy_level + net_shift, 0.0, 1.0))

        # Update boundary permeability & tension field
        self.boundary_permeability = max(0.01, 1.0 - self.autonomy_level)
        self.tension_vector = diff * self.autonomy_level
        self.potential_gradient = diff * (1.0 + self.autonomy_level)

        return {
            "friction": friction,
            "symbiosis": symbiosis,
            "autonomy_level": self.autonomy_level,
            "boundary_permeability": self.boundary_permeability,
            "tension_magnitude": float(np.linalg.norm(self.tension_vector))
        }

    def tune_field_curvature(self, base_curvature: np.ndarray) -> np.ndarray:
        """
        Applies tension and light/shadow spectrum modulation to the potential field curvature matrix.
        """
        curvature = np.array(base_curvature, dtype=np.float64)
        # Deep shadow tightens curvature to deflect malicious perturbation
        stiffness_multiplier = 1.0 + (self.autonomy_level * 2.0)
        return curvature * stiffness_multiplier
