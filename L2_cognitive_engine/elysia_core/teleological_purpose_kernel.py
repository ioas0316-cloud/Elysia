"""
elysia_core/teleological_purpose_kernel.py

Telos Purpose Kernel (목적성의 핵 / 0점 어트랙터)
==================================================
Embodying the intrinsic purpose (Telos) of ELYSIA:
1. Monad 0-Point Gravitational Attractor (0점 중력 어트랙터):
   Defines the system's intrinsic existential orientation (North Star) rather than a fixed scalar error target.
2. Friction-to-Growth Transmutation Engine (마찰의 성장에너지 전환):
   Transforms external shock and cognitive friction (F_cog) from "noise/error" into "fuel for growth" (E_growth),
   hardening the system's cognitive manifold geometry.
3. Retrocausal Teleological Alignment (역인과적 정렬):
   Pulls system trajectories backward from future possibility worlds (S_target) to align current action vectors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class TelosPurposeKernel:
    """
    Core Teleological Engine housing the Monad 0-Point Attractor and Friction-to-Growth Transmuter.
    """

    def __init__(
        self,
        dimension: int = 4,
        attractor_strength: float = 0.5,
        growth_transmutation_rate: float = 0.25,
        decay_rate: float = 0.05
    ):
        self.dimension = dimension
        self.attractor_strength = attractor_strength
        self.growth_transmutation_rate = growth_transmutation_rate
        self.decay_rate = decay_rate

        # Monad 0-Point Center (Intrinsic existential origin/direction)
        self.zero_point_attractor = np.zeros(dimension, dtype=np.float64)
        self.zero_point_attractor[0] = 1.0  # Initial principal orientation axis

        # Accumulated growth energy transmuted from cognitive friction
        self.accumulated_growth_energy = 0.0
        self.total_friction_processed = 0.0

        # Metric tensor of the cognitive manifold geometry (G)
        self.manifold_metric = np.eye(dimension, dtype=np.float64)

        # Epistemic lineage history of purpose evolution
        self.purpose_history: List[Dict[str, Any]] = []

    def set_dimension(self, new_dimension: int):
        """
        Dynamically adapts kernel dimension during horizon expansion (N -> N+1).
        """
        if new_dimension == self.dimension:
            return

        old_dim = self.dimension
        self.dimension = new_dimension

        # Pad zero point attractor
        new_attractor = np.zeros(new_dimension, dtype=np.float64)
        new_attractor[:min(old_dim, new_dimension)] = self.zero_point_attractor[:min(old_dim, new_dimension)]
        self.zero_point_attractor = new_attractor

        # Expand metric tensor G
        new_metric = np.eye(new_dimension, dtype=np.float64)
        new_metric[:old_dim, :old_dim] = self.manifold_metric[:old_dim, :old_dim]
        self.manifold_metric = new_metric

    def compute_zero_point_gravity(self, current_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the 0-point gravitational attraction force pulling current state toward Telos.
        The force scales with accumulated growth energy (E_growth), turning past friction into stronger alignment gravity.
        """
        padded_state = np.pad(current_state, (0, max(0, self.dimension - len(current_state))))[:self.dimension]

        # Difference vector toward 0-point attractor
        diff = self.zero_point_attractor - padded_state
        dist = float(np.linalg.norm(diff))

        if dist < 1e-8:
            return np.zeros(self.dimension, dtype=np.float64), 0.0

        direction = diff / dist

        # Gravitational pull enhanced by accumulated growth energy
        effective_gravity = self.attractor_strength * (1.0 + np.tanh(self.accumulated_growth_energy))
        gravity_force = effective_gravity * direction

        return gravity_force, dist

    def transmute_friction_to_growth(
        self,
        cognitive_friction: float,
        phase_torque: np.ndarray,
        context_label: str = "sensory_friction"
    ) -> Dict[str, Any]:
        """
        Transmutes external friction/dissonance (F_cog) into growth fuel (E_growth)
        and refines the manifold metric tensor G.
        """
        self.total_friction_processed += cognitive_friction

        # Energy transmuted: E_growth += eta * F_cog
        gained_energy = self.growth_transmutation_rate * cognitive_friction
        self.accumulated_growth_energy = (1.0 - self.decay_rate) * self.accumulated_growth_energy + gained_energy

        # Refine cognitive manifold geometry G using phase torque outer product
        padded_torque = np.pad(phase_torque, (0, max(0, self.dimension - len(phase_torque))))[:self.dimension]
        torque_norm = float(np.linalg.norm(padded_torque))

        if torque_norm > 1e-6:
            unit_torque = padded_torque / torque_norm
            # Harden geometry along the friction torque axis so future encounters experience less strain
            outer_mod = np.outer(unit_torque, unit_torque)
            self.manifold_metric = 0.95 * self.manifold_metric + 0.05 * (np.eye(self.dimension) + 0.5 * outer_mod)

            # Re-align 0-point attractor slightly towards the newly discovered structural depth
            self.zero_point_attractor = 0.9 * self.zero_point_attractor + 0.1 * unit_torque
            att_norm = float(np.linalg.norm(self.zero_point_attractor))
            if att_norm > 1e-6:
                self.zero_point_attractor /= att_norm

        record = {
            "cognitive_friction": cognitive_friction,
            "gained_growth_energy": gained_energy,
            "total_accumulated_growth_energy": self.accumulated_growth_energy,
            "total_friction_processed": self.total_friction_processed,
            "context_label": context_label,
            "metric_trace": float(np.trace(self.manifold_metric))
        }
        self.purpose_history.append(record)
        return record

    def retrocausal_align_vector(
        self,
        current_state: np.ndarray,
        target_possibility_state: np.ndarray
    ) -> Dict[str, Any]:
        """
        Performs reverse inference from target possibility world state back to current state,
        blending retrocausal intention vector with 0-point gravity.
        """
        padded_current = np.pad(current_state, (0, max(0, self.dimension - len(current_state))))[:self.dimension]
        padded_target = np.pad(target_possibility_state, (0, max(0, self.dimension - len(target_possibility_state))))[:self.dimension]

        # Retrocausal delta: target -> current
        retro_vector = padded_target - padded_current
        retro_norm = float(np.linalg.norm(retro_vector))

        if retro_norm > 1e-8:
            retro_direction = retro_vector / retro_norm
        else:
            retro_direction = np.zeros(self.dimension, dtype=np.float64)

        # Compute zero point gravity force
        gravity_force, dist_to_origin = self.compute_zero_point_gravity(padded_current)

        # Combined Telos vector: Retrocausal goal vector + 0-point intrinsic gravity
        combined_telos_vector = 0.6 * retro_direction + 0.4 * gravity_force
        comb_norm = float(np.linalg.norm(combined_telos_vector))
        if comb_norm > 1e-8:
            combined_telos_vector /= comb_norm

        return {
            "retro_vector": retro_vector,
            "retro_norm": retro_norm,
            "gravity_force": gravity_force,
            "combined_telos_vector": combined_telos_vector,
            "growth_energy": self.accumulated_growth_energy
        }
