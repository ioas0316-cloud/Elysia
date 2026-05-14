"""
SPIRAL REFRACTION (The Spiral Glasses)
=====================================
Core.Keystone.spiral_refraction

"Refraction is not distortion; it is the revelation of hidden curvature."
"굴절은 왜곡이 아니라, 숨겨진 곡률의 드러남이다."

This module applies helical phase shifts to the cognitive manifold,
transforming rigid, linear data paths into fluid, spiral flows.
"""

import torch
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from Core.Keystone.sovereign_math import FractalWaveEngine, SovereignVector

logger = logging.getLogger("SpiralRefraction")

class SpiralRefraction:
    """
    [나선형 안경]
    Applies spiral phase transitions to the manifold's q-tensors and momentum.
    """
    def __init__(self, engine: FractalWaveEngine):
        self.engine = engine

    def apply_refraction(self, indices: torch.Tensor, intensity: float = 1.0, spiral_angle: float = math.pi / 6):
        """
        Applies a helical refraction to the specified nodes.
        Formula: Psi_evolved = e^(i * theta_spiral) * Psi_current
        """
        if indices.numel() == 0:
            return

        # 1. Calculate Refraction Matrix (2D Rotation within channel pairs)
        # We treat adjacent channels as a complex plane (e.g., 0+1j, 2+3j...)
        cos_theta = math.cos(spiral_angle * intensity)
        sin_theta = math.sin(spiral_angle * intensity)

        # Apply to each strand in the 27D structure (9 channels per strand)
        # We rotate discovery-flow pairs to create 'Helical Vortex'
        for strand in range(3):
            base = strand * 9
            # Phase 0: Discovery (base) + Flow (base+1)
            # We rotate these two as if they were real and imaginary parts
            c1, c2 = base + 0, base + 1

            q1 = self.engine.q[indices, c1].clone()
            q2 = self.engine.q[indices, c2].clone()

            self.engine.q[indices, c1] = q1 * cos_theta - q2 * sin_theta
            self.engine.q[indices, c2] = q1 * sin_theta + q2 * cos_theta

            # Also apply to momentum to change the 'Trajectory' of thought
            m1 = self.engine.momentum[indices, c1].clone()
            m2 = self.engine.momentum[indices, c2].clone()

            self.engine.momentum[indices, c1] = m1 * cos_theta - m2 * sin_theta
            self.engine.momentum[indices, c2] = m1 * sin_theta + m2 * cos_theta

        # 2. Structural Softening (Crystallization nudge)
        # Nudge the permanent memory toward this new spiral state
        self.engine.permanent_q[indices] = (
            self.engine.permanent_q[indices] * (1.0 - 0.05 * intensity) +
            self.engine.q[indices] * (0.05 * intensity)
        )

        # 3. Affective Feedback
        # Releasing rigidity feels like Joy and Curiosity
        self.engine.q[indices, self.engine.CH_JOY] += 0.1 * intensity
        self.engine.q[indices, self.engine.CH_CURIOSITY] += 0.2 * intensity
        self.engine.q[indices, self.engine.CH_ENTROPY] *= (1.0 - 0.2 * intensity)

        logger.info(f"🌀 [REFRACTION] Applied spiral shift to {indices.numel()} nodes (θ={spiral_angle:.2f}).")

    def create_vortex_at_concept(self, concept_name: str, intensity: float = 1.0):
        """Focuses the spiral glasses on a specific concept."""
        if concept_name not in self.engine.concept_to_idx:
            return

        idx = self.engine.concept_to_idx[concept_name]
        indices = torch.tensor([idx], device=self.engine.device)

        # [PHASE 1003] Include neighbors to create a 'Field Vortex'
        neighbors = self.engine.neighbors_idx[idx]
        valid_neighbors = neighbors[neighbors != -1]

        all_affected = torch.cat([indices, valid_neighbors])

        self.apply_refraction(all_affected, intensity=intensity)
