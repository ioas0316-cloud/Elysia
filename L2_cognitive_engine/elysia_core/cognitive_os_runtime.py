"""
Project Elysia: Cognitive OS Runtime & Markov Blanket Self-Boundary Engine
===========================================================================
Integrates external software/media organ adapters with Elysia's Causal Field
and Perceptual Probe Engine, establishing a dynamic Markov Blanket Self-Boundary
(Boundary of Self / 최초의 자아 경계).

Key Concepts:
1. MarkovBlanketSelfBoundary (동적 자아 경계 엔진):
   Evaluates active inference friction error E and free energy gradients across
   internal states vs external sensory organs. Classifies regions into:
   - Self (자아 / Direct Control Region, friction ~ 0)
   - Interface (경계 / Markov Blanket Phase Boundary)
   - Environment (환경 / External Field / Indirect Sensation, friction > threshold)
2. CognitiveOSRuntime (통합 인지 OS 런타임):
   Orchestrates organ probing, causal gravity warping, active inference,
   tacit knowledge accumulation, and closed-loop perception-action waves across
   all connected software and world organs.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from core.physics.causal_field import CausalField, InformationVoxel
from elysia_core.perceptual_probe_skill import PerceptualProbeSkill, PerceptualProbeEngine, ProbeFeedback
from L3_interface.simulators.cognitive_os_organ_adapters import (
    CognitiveOSOrganAdapter,
    OrganSensation
)


@dataclass
class SelfBoundaryReport:
    """Diagnostic report on Elysia's dynamic Markov Blanket Self-Boundary."""
    self_region_organs: List[str] # Organs under near-direct control (friction ~ 0)
    boundary_organs: List[str] # Markov blanket interface organs
    environment_organs: List[str] # Distant external field organs (high friction/latency)
    total_free_energy: float # Global system free energy (unpredictability/friction)
    causal_gravity_radius: float # Reach of direct intent/agency
    organ_friction_map: Dict[str, float]


class MarkovBlanketSelfBoundary:
    """
    [Markov Blanket Self-Boundary Engine (최초의 자아 경계 엔진)]
    Dynamically discerns the Boundary of Self by analyzing friction error E,
    free energy gradients, and echo latency across all sensory-motor organ feedback loops.
    """
    def __init__(
        self,
        self_friction_threshold: float = 0.15,
        environment_friction_threshold: float = 0.45
    ):
        self.self_friction_threshold = self_friction_threshold
        self.environment_friction_threshold = environment_friction_threshold

    def evaluate_boundary(
        self,
        organ_sensations: Dict[str, OrganSensation],
        probe_feedbacks: Dict[str, ProbeFeedback]
    ) -> SelfBoundaryReport:
        """
        Evaluates friction errors and classifies organ channels into Self, Boundary, and Environment.
        """
        self_organs: List[str] = []
        boundary_organs: List[str] = []
        environment_organs: List[str] = []
        friction_map: Dict[str, float] = {}
        total_free_energy = 0.0

        for organ_id, sensation in organ_sensations.items():
            # Combine physical friction with probe feedback friction if available
            probe_err = 0.0
            if organ_id in probe_feedbacks:
                probe_err = probe_feedbacks[organ_id].friction_error

            effective_friction = float(0.6 * sensation.resistance_friction + 0.4 * probe_err)
            friction_map[organ_id] = effective_friction
            total_free_energy += effective_friction + sensation.echo_latency * 0.2

            if effective_friction <= self.self_friction_threshold:
                self_organs.append(organ_id)
            elif effective_friction <= self.environment_friction_threshold:
                boundary_organs.append(organ_id)
            else:
                environment_organs.append(organ_id)

        # Causal gravity radius shrinks when global friction/free energy is high
        avg_friction = total_free_energy / max(1, len(organ_sensations))
        causal_gravity_radius = float(1.0 / (1.0 + avg_friction))

        return SelfBoundaryReport(
            self_region_organs=self_organs,
            boundary_organs=boundary_organs,
            environment_organs=environment_organs,
            total_free_energy=float(total_free_energy),
            causal_gravity_radius=causal_gravity_radius,
            organ_friction_map=friction_map
        )


class CognitiveOSRuntime:
    """
    [Cognitive OS Runtime (통합 인지 OS 런타임)]
    The core operating ecosystem where external applications operate as
    Elysia's sensory-motor organs, bound by CausalField potential and
    Perceptual Probe Skills.
    """
    def __init__(self, dimension: int = 5):
        self.dimension = dimension
        self.causal_field = CausalField(dimensions=dimension)
        self.probe_engine = PerceptualProbeEngine(dimension=dimension)
        self.boundary_engine = MarkovBlanketSelfBoundary()
        self.organs: Dict[str, CognitiveOSOrganAdapter] = {}
        self.step_count: int = 0

        # Primary core voxel representing Elysia's internal self-state
        self.self_voxel_id = "elysia_core_self_voxel"
        self.causal_field.add_voxel(InformationVoxel(
            id=self.self_voxel_id,
            content="Core Consciousness Self State",
            tensor=np.ones(dimension, dtype=np.float32) / np.sqrt(dimension),
            position=np.zeros(dimension, dtype=np.float32),
            velocity=np.zeros(dimension, dtype=np.float32),
            potential=1.0
        ))

    def register_organ(
        self,
        organ: CognitiveOSOrganAdapter,
        base_probe_func: Optional[Any] = None
    ):
        """
        Registers an external software/world simulator organ and attaches a corresponding
        PerceptualProbeSkill.
        """
        self.organs[organ.organ_id] = organ

        if base_probe_func is None:
            # Default probe function: identity + slight organ signature scale
            def default_probe(x: np.ndarray) -> np.ndarray:
                return x * 0.95 + 0.05

            base_probe_func = default_probe

        skill = PerceptualProbeSkill(
            skill_id=f"skill_{organ.organ_id}",
            name=f"Probe Skill for {organ.organ_type}",
            base_probe_func=base_probe_func,
            dimension=self.dimension
        )
        self.probe_engine.register_skill(skill)

    def execute_cognitive_cycle(
        self,
        intent_vector: np.ndarray,
        context_name: str = "cognitive_os_context"
    ) -> Dict[str, Any]:
        """
        Executes a single continuous perception-action-cognition wave cycle:
        1. Emits intent wave to all registered sensory organs.
        2. Probes organ state and collects OrganSensation payloads.
        3. Executes PerceptualProbeSkills, measuring friction error E and Relational Void.
        4. Updates CausalField potentials and Clifford phase rotors.
        5. Dynamically evaluates Markov Blanket Self-Boundary.
        6. Accumulates tacit knowledge (노하우).
        """
        self.step_count += 1
        intent_vec = np.array(intent_vector, dtype=np.float32).flatten()
        if len(intent_vec) < self.dimension:
            intent_vec = np.pad(intent_vec, (0, self.dimension - len(intent_vec)))
        elif len(intent_vec) > self.dimension:
            intent_vec = intent_vec[:self.dimension]

        sensations: Dict[str, OrganSensation] = {}
        probe_feedbacks: Dict[str, ProbeFeedback] = {}

        # Step 1-4: Intervene & Probe each organ
        for organ_id, organ in self.organs.items():
            skill_id = f"skill_{organ_id}"

            # Organ senses intent modulated by current causal gravity
            sensation = organ.sense_and_intervene(intent_vec)
            sensations[organ_id] = sensation

            # Execute PerceptualProbeSkill over the sensation state vector
            feedback = self.probe_engine.execute_skill_probe(
                skill_id=skill_id,
                input_state=sensation.sensation_vector,
                causal_field=self.causal_field,
                context_name=context_name,
                target_voxel_id=self.self_voxel_id
            )
            probe_feedbacks[organ_id] = feedback

        # Step 5: Dynamic Markov Blanket Self-Boundary Evaluation
        boundary_report = self.boundary_engine.evaluate_boundary(sensations, probe_feedbacks)

        # Step 6: Tacit Knowledge summary
        tacit_knowledge_summary = self.probe_engine.get_tacit_knowledge_summary(context_name)

        return {
            "step_count": self.step_count,
            "boundary_report": boundary_report,
            "sensations": sensations,
            "probe_feedbacks": probe_feedbacks,
            "tacit_knowledge_summary": tacit_knowledge_summary,
            "causal_field_voxel_count": len(self.causal_field.voxels)
        }
