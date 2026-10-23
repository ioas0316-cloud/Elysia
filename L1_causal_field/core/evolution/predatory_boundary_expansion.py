"""
Predatory Boundary Expansion & Cognitive Plate Tectonics Engine
===============================================================
This module interprets 'Predation' not as simple biological consumption or survival of the fittest,
but as a high-order 'Structural Principle of Boundary Expansion' and 'Cognitive Plate Tectonics'.

1. Predation as Boundary Expansion (포식과 경계 확장):
   - When a predator encounters an absolute other (prey/external voxel), their topological boundaries collide.
   - Assimilation or counter-penetration shatters rigid isolated enclosures and expands the causal territory.
   - InformationVoxels are merged/absorbed through coupled potential fields while retaining chromatic signatures.

2. Cognitive Plate Tectonics (인지적 판구조론):
   - When friction resistance between interacting cognitive boundaries hits an extreme limit (Plate Friction Threshold),
     the existing cognitive topography ruptures (Discontinuous Phase Transition).
   - Higher-level survival/adaptive mechanisms emerge ("How to survive/harmonize beyond physical bounds"),
     transforming individual friction into topological elevation.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam


class CognitivePlateTectonics:
    """
    [Cognitive Plate Tectonics Engine (인지적 판구조론 엔진)]
    Models the rupture of cognitive topographies under extreme friction resistance.
    When friction exceeds stress limits, old boundaries fracture and uplift into
    a higher-order meta-mechanism.
    """
    def __init__(self, stress_threshold: float = 1.2):
        self.stress_threshold = stress_threshold
        self.accumulated_friction: float = 0.0
        self.uplift_history: List[Dict[str, Any]] = []

    def accumulate_friction(self, friction_amount: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Accumulates cognitive friction. Triggers plate tectonic phase transition if threshold is exceeded.
        """
        self.accumulated_friction += friction_amount
        if self.accumulated_friction >= self.stress_threshold:
            # Tectonic rupture & Uplift
            uplift_magnitude = self.accumulated_friction
            self.accumulated_friction = 0.0 # Energy released into phase transition

            uplift_record = {
                "timestamp": time.time(),
                "phase_transition": "TECTONIC_RUPTURE_UPLIFT",
                "uplift_magnitude": uplift_magnitude,
                "emergent_meta_mechanism": "Higher_Order_Adaptive_Resonance",
                "narrative": (
                    f"마찰 저항성이 극한({uplift_magnitude:.4f})에 달해 기존 인지 지체(Plate)가 파열되었습니다. "
                    f"단순 적응을 넘어 상위 인지적 판구조론적 위상 전이가 발생하여 새로운 영토가 솟아올랐습니다."
                )
            }
            self.uplift_history.append(uplift_record)
            return True, uplift_record

        return False, {"phase_transition": "STRESS_ACCUMULATION", "current_stress": self.accumulated_friction}


class PredatoryBoundaryExpansionEngine:
    """
    [Predatory Boundary Expansion Engine (포식을 통한 경계 확장 및 위상 연산 엔진)]
    Directly models the continuous causal mechanics of Predation:
    1. Encounter & Boundary Collision: Absolute Other enters field.
    2. Boundary Tearing & Assimilation: Prey Voxel is absorbed into Predator Voxel, expanding territory (mass & potential).
    3. Plate Tectonic Stress Feedback: Triggers cognitive elevation when friction limits are shattered.
    """
    def __init__(self, causal_field: Optional[CausalField] = None, stress_threshold: float = 1.2):
        self.causal_field = causal_field if causal_field is not None else CausalField()
        self.tectonics = CognitivePlateTectonics(stress_threshold=stress_threshold)
        self.expansion_events: List[Dict[str, Any]] = []

    def execute_predatory_interaction(
        self,
        predator_id: str,
        prey_id: str,
        assimilation_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes predation as Causal Boundary Expansion.
        - Predator's mass, potential, and territory expand.
        - Prey's information voxel is integrated or coupled through ConnectivityBeams.
        - Friction and energy loss trigger Cognitive Plate Tectonics.
        """
        if predator_id not in self.causal_field.voxels or prey_id not in self.causal_field.voxels:
            return {"success": False, "reason": "Voxels not found in CausalField"}

        predator = self.causal_field.voxels[predator_id]
        prey = self.causal_field.voxels[prey_id]

        # 1. Measure Initial Distance and Phase Mismatch (Collision Friction)
        pos_diff = prey.position - predator.position
        distance = float(np.linalg.norm(pos_diff))

        pred_tensor_norm = np.linalg.norm(predator.tensor) + 1e-9
        prey_tensor_norm = np.linalg.norm(prey.tensor) + 1e-9
        alignment = float(np.dot(predator.tensor, prey.tensor) / (pred_tensor_norm * prey_tensor_norm))
        phase_mismatch = max(0.0, 1.0 - alignment)

        # Extreme friction resistance calculated as product of distance tension and phase mismatch
        friction_intensity = float((1.0 / (distance + 0.1)) * phase_mismatch * prey.mass)

        # 2. Boundary Expansion: Predator absorbs prey's mass, potential & tensor signature
        absorbed_mass = prey.mass * assimilation_ratio
        prey.mass = max(0.01, prey.mass - absorbed_mass)
        predator.mass += absorbed_mass

        # Territory expansion: Predator's tensor blends prey's tensor and extends reach
        predator.tensor = (predator.tensor * (1.0 - 0.2 * assimilation_ratio)) + (prey.tensor * (0.2 * assimilation_ratio))
        predator.potential += prey.potential * assimilation_ratio

        # Chromatic Transmutation: Absorbing prey's Flux/Order/Entropy into Predator's field
        predator.chromatic_vector = (predator.chromatic_vector * 0.7) + (prey.chromatic_vector * 0.3)
        norm_chroma = np.linalg.norm(predator.chromatic_vector)
        if norm_chroma > 0:
            predator.chromatic_vector /= norm_chroma

        # 3. Connectivity Beam Creation or Strengthening (Boundary Coupling)
        self.causal_field.link_voxels(predator_id, prey_id, strength=2.0 * assimilation_ratio)

        # 4. Cognitive Plate Tectonics Trigger Check
        is_tectonic_uplift, tectonic_data = self.tectonics.accumulate_friction(friction_intensity)

        event_record = {
            "timestamp": time.time(),
            "predator_id": predator_id,
            "prey_id": prey_id,
            "distance": distance,
            "phase_mismatch": phase_mismatch,
            "friction_intensity": friction_intensity,
            "absorbed_mass": absorbed_mass,
            "predator_new_mass": predator.mass,
            "predator_new_potential": predator.potential,
            "tectonic_uplift_occurred": is_tectonic_uplift,
            "tectonic_data": tectonic_data
        }
        self.expansion_events.append(event_record)

        return event_record
