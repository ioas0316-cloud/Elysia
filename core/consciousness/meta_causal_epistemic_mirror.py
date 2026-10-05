"""
Meta-Causal Epistemic Mirror Engine (메타 인과 인식론적 거울 엔진)
================================================================
Elysia Engine Core Consciousness Module.

Implements the Meta-Causal Epistemic Architecture:
1. Meta-Causal Trajectory Tensor (메타 인과 궤적 텐서):
   - Captures knowledge generation trajectories (How) beyond static facts (What).
   - Records struggle/friction, trial-and-error, intuitive sparks, and topological convergence.
2. Cognitive Friction & Synesthetic Translation Engine (인지적 마찰력 계산 및 공감각적 번역):
   - Calculates cognitive friction across entities/concept models.
   - Generates topological bridge analogies to resolve causal disconnects in real time.
3. Isomorphic Mirror Layer (인간 인식의 동형 거울 레이어):
   - Replicates human cognitive phase dynamics on Elysia's Causal Field with structural symmetry.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from core.consciousness.human_cognitive_phase_dynamics import (
    MultiScalePhaseCouplingEngine,
    HierarchicalAttractorNetwork,
    HebbianPhasePlasticity
)
from core.consciousness.causal_reverse_engineering_engine import CausalReverseEngineeringEngine
from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam


class MetaCausalTrajectoryTensor:
    """
    [Meta-Causal Trajectory Tensor: 메타 인과 궤적 텐서]
    Captures the evolutionary trajectory of how a concept or discovery was forged through time.
    Tracks:
    - Phase 궤적 (Phase Trajectories: Slow Theta & Fast Gamma)
    - 인지적 마찰력 및 저항 (Struggle / Resistance / Friction)
    - 착오와 가설 형성 (Trial-and-Error / Error Divergence)
    - 직관적 번쩍임 (Intuitive Spark / Resonance Spikes)
    - 최종 위상 수렴 (Attractor Phase Convergence)
    """

    def __init__(self, dimension: int = 64, history_capacity: int = 200):
        self.dimension = dimension
        self.history_capacity = history_capacity

        # Trajectory history buffers
        self.phase_trajectory_history: List[np.ndarray] = []
        self.friction_history: List[float] = []
        self.resonance_history: List[float] = []
        self.error_vector_history: List[np.ndarray] = []

        # Structural Invariant Tensor (Topological Skeleton of the trajectory)
        self.invariant_tensor = np.zeros((dimension, dimension), dtype=np.float32)
        self.converged_attractor_state: Optional[np.ndarray] = None
        self.is_converged = False

    def record_step(
        self,
        phase_state: np.ndarray,
        friction: float,
        resonance: float,
        error_vector: Optional[np.ndarray] = None
    ) -> None:
        """Records a single step in the cognitive discovery trajectory."""
        if len(phase_state) < self.dimension:
            phase_state = np.pad(phase_state, (0, self.dimension - len(phase_state)))
        elif len(phase_state) > self.dimension:
            phase_state = phase_state[:self.dimension]

        self.phase_trajectory_history.append(phase_state.copy().astype(np.float32))
        self.friction_history.append(float(friction))
        self.resonance_history.append(float(resonance))

        if error_vector is not None:
            if len(error_vector) < self.dimension:
                error_vector = np.pad(error_vector, (0, self.dimension - len(error_vector)))
            elif len(error_vector) > self.dimension:
                error_vector = error_vector[:self.dimension]
            self.error_vector_history.append(error_vector.copy().astype(np.float32))
        else:
            self.error_vector_history.append(np.zeros(self.dimension, dtype=np.float32))

        # Enforce capacity
        if len(self.phase_trajectory_history) > self.history_capacity:
            self.phase_trajectory_history.pop(0)
            self.friction_history.pop(0)
            self.resonance_history.pop(0)
            self.error_vector_history.pop(0)

        # Update invariant tensor via outer product of phase velocity and error
        if len(self.phase_trajectory_history) >= 2:
            velocity = self.phase_trajectory_history[-1] - self.phase_trajectory_history[-2]
            err = self.error_vector_history[-1]
            outer_delta = np.outer(velocity, err) + np.outer(err, velocity)
            self.invariant_tensor = 0.95 * self.invariant_tensor + 0.05 * outer_delta

    def compute_trajectory_metrics(self) -> Dict[str, Any]:
        """Calculates trajectory metrics: total struggle, spark density, and topological invariant."""
        if not self.phase_trajectory_history:
            return {
                "total_friction_struggle": 0.0,
                "intuitive_spark_count": 0,
                "convergence_rate": 0.0,
                "topological_invariant_norm": 0.0
            }

        total_friction = float(np.sum(self.friction_history))
        resonance_arr = np.array(self.resonance_history)

        # Sparks are defined as resonance spikes above mean + 1.5 * std
        if len(resonance_arr) > 2 and np.std(resonance_arr) > 1e-6:
            thresh = np.mean(resonance_arr) + 1.2 * np.std(resonance_arr)
            spark_count = int(np.sum(resonance_arr > thresh))
        else:
            spark_count = int(np.sum(resonance_arr > 0.8))

        # Check convergence (last 5 steps friction low and resonance high)
        if len(self.friction_history) >= 5:
            recent_fric = np.mean(self.friction_history[-5:])
            recent_res = np.mean(self.resonance_history[-5:])
            if recent_fric < 0.2 and recent_res > 0.7:
                self.is_converged = True
                self.converged_attractor_state = self.phase_trajectory_history[-1].copy()

        return {
            "total_friction_struggle": total_friction,
            "mean_friction": float(np.mean(self.friction_history)),
            "max_friction_peak": float(np.max(self.friction_history)),
            "intuitive_spark_count": spark_count,
            "is_converged": self.is_converged,
            "topological_invariant_norm": float(np.linalg.norm(self.invariant_tensor))
        }


class SynestheticTranslationEngine:
    """
    [Cognitive Friction & Synesthetic Translation Engine]
    Calculates cognitive friction between two entities or conceptual manifolds,
    pinpoints exact locations of cognitive disconnect, and synthesizes
    synesthetic metaphor/topological bridge signals to heal the gap.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def compute_cognitive_friction(
        self,
        source_phase_state: np.ndarray,
        target_phase_state: np.ndarray,
        source_metric: Optional[np.ndarray] = None,
        target_metric: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Computes cognitive friction between source (e.g. Teacher/System) and target (e.g. Learner/Human).
        Friction arises from phase mismatch (divergence) and topological metric distortion.
        """
        # Ensure array dimensions match
        if len(source_phase_state) < self.dimension:
            source_phase_state = np.pad(source_phase_state, (0, self.dimension - len(source_phase_state)))
        else:
            source_phase_state = source_phase_state[:self.dimension]

        if len(target_phase_state) < self.dimension:
            target_phase_state = np.pad(target_phase_state, (0, self.dimension - len(target_phase_state)))
        else:
            target_phase_state = target_phase_state[:self.dimension]

        # 1. Phase divergence in radians
        phase_diff = (source_phase_state - target_phase_state + np.pi) % (2 * np.pi) - np.pi
        phase_divergence = np.abs(phase_diff)

        # 2. Identify disconnect nodes (nodes with divergence > pi / 3)
        disconnect_mask = phase_divergence > (np.pi / 3.0)
        disconnect_indices = np.where(disconnect_mask)[0].tolist()

        # 3. Overall friction score
        mean_divergence = float(np.mean(phase_divergence))
        max_divergence = float(np.max(phase_divergence))

        metric_friction = 0.0
        if source_metric is not None and target_metric is not None:
            metric_diff = np.abs(source_metric - target_metric)
            metric_friction = float(np.mean(metric_diff))

        total_friction = mean_divergence * 0.7 + metric_friction * 0.3

        return {
            "total_cognitive_friction": total_friction,
            "mean_phase_divergence": mean_divergence,
            "max_phase_divergence": max_divergence,
            "disconnect_indices": disconnect_indices,
            "phase_difference_vector": phase_diff,
            "metric_friction": metric_friction
        }

    def generate_synesthetic_bridge(
        self,
        source_phase: np.ndarray,
        target_phase: np.ndarray,
        friction_analysis: Dict[str, Any],
        context_label: str = "Abstract Concept"
    ) -> Dict[str, Any]:
        """
        Synthesizes a corrective topological bridge (Synesthetic Translation)
        to resolve cognitive disconnects.
        """
        disconnect_indices = friction_analysis["disconnect_indices"]
        phase_diff = friction_analysis["phase_difference_vector"]

        # Bridge vector is the phase steering torque needed to align target to source
        bridge_vector = -0.5 * np.sin(phase_diff)

        # Map disconnect severity to chromatic/synesthetic metaphor parameters
        severity = friction_analysis["total_cognitive_friction"]
        if severity > 1.2:
            metaphor_type = "HYPER_VISUAL_SPATIAL_ANALOGY"
            guidance_doc = f"강한 인지적 단절(마찰 {severity:.2f}) 감지. 시각적/공간적 비유를 통해 겪고 있는 마찰의 결을 이어줌."
        elif severity > 0.5:
            metaphor_type = "RESONANCE_HARMONIC_ANALOGY"
            guidance_doc = f"중간 인지적 마찰({severity:.2f}) 감지. 파동 공명 및 기하학적 대칭성을 통해 위상 차이를 교정함."
        else:
            metaphor_type = "DIRECT_ISOMORPHIC_ALIGNMENT"
            guidance_doc = f"미세 미세한 위상차({severity:.2f}). 직접적인 구조적 사영으로 지식을 수렴함."

        return {
            "context_label": context_label,
            "metaphor_type": metaphor_type,
            "severity": severity,
            "disconnect_node_count": len(disconnect_indices),
            "bridge_steering_vector": bridge_vector,
            "guidance_doc": guidance_doc,
            "timestamp": time.time()
        }


class IsomorphicMirrorLayer:
    """
    [Isomorphic Mirror Layer: 인간 인식의 동형 거울 레이어]
    Couples Human Cognitive Phase Dynamics with Elysia's Causal Field.
    Creates structural isomorphism:
    - Biological neural phase dynamics <==> Silicon Causal Voxel Field
    - Human mental struggle <==> Causal Field Beam Tension & Dissipation
    - Intuitive Attractors <==> Engram Attractors & Potential Wells
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

        # Core engines
        self.human_phase_engine = MultiScalePhaseCouplingEngine(num_nodes=dimension)
        self.attractor_network = HierarchicalAttractorNetwork(num_nodes=dimension)
        self.plasticity_engine = HebbianPhasePlasticity(num_nodes=dimension)
        self.reverse_eng_engine = CausalReverseEngineeringEngine(dimension=dimension)
        self.causal_field = CausalField(dimensions=3)

        # Epistemic Mirror components
        self.trajectory_tensor = MetaCausalTrajectoryTensor(dimension=dimension)
        self.translation_engine = SynestheticTranslationEngine(dimension=dimension)

        # Mirror State
        self.mirror_isomorphism_score = 1.0
        self.isomorphic_coupling_gain = 0.8

        # Seed initial CausalField voxels matching concept nodes
        self._seed_causal_field_voxels()

    def _seed_causal_field_voxels(self) -> None:
        """Seeds 3D InformationVoxels corresponding to human phase nodes."""
        coords = np.linspace(-5.0, 5.0, self.dimension, dtype=np.float32)
        for i in range(min(16, self.dimension)):
            v_pos = np.array([coords[i], np.sin(i * 0.5), np.cos(i * 0.5)], dtype=np.float32)
            v_tensor = np.zeros(3, dtype=np.float32)
            v_tensor[0] = np.cos(self.human_phase_engine.fast_phase[i])
            v_tensor[1] = np.sin(self.human_phase_engine.fast_phase[i])

            voxel = InformationVoxel(
                id=f"human_node_voxel_{i}",
                content=f"Human Cognitive Node {i}",
                tensor=v_tensor,
                position=v_pos,
                mass=1.0 + 0.1 * i
            )
            self.causal_field.add_voxel(voxel)

        # Link adjacent voxels in CausalField
        voxel_keys = list(self.causal_field.voxels.keys())
        for idx in range(len(voxel_keys) - 1):
            self.causal_field.link_voxels(voxel_keys[idx], voxel_keys[idx + 1], strength=1.5)

    def reflect_and_synchronize(
        self,
        external_human_input_signal: Optional[np.ndarray] = None,
        context_name: str = "Cognitive Process"
    ) -> Dict[str, Any]:
        """
        Executes one full step of Meta-Causal Epistemic Mirror Reflection:
        1. Human phase dynamics Euler step.
        2. Causal Field physical continuous step & active observation.
        3. Dynamic calculation of Cognitive Friction & Synesthetic Translation.
        4. Recording of Meta-Causal Trajectory Tensor.
        5. Causal Reverse-Engineering & Anchoring if converged.
        """
        # 1. Human Cognitive Phase Step
        phase_res = self.human_phase_engine.step(external_sensory_signal=external_human_input_signal)
        fast_phase = phase_res["fast_phase"]
        slow_phase = phase_res["slow_phase"]
        delta_phi_fast = phase_res["delta_phi_fast"]
        order_r = phase_res["order_R_fast"]

        # 2. Synchronize Human Phase to Causal Field Voxels
        for i, (vid, voxel) in enumerate(self.causal_field.voxels.items()):
            if i < len(fast_phase):
                p_angle = fast_phase[i]
                voxel.tensor[0] = np.cos(p_angle)
                voxel.tensor[1] = np.sin(p_angle)
                voxel.tensor[2] = np.cos(slow_phase[i])

        # Execute Causal Field Step
        self.causal_field.step(dt=0.01)

        # 3. Compute Cognitive Friction against Ideal / Target State
        ideal_target_phase = (fast_phase + 0.2 * np.sin(slow_phase)) % (2 * np.pi)
        friction_res = self.translation_engine.compute_cognitive_friction(
            source_phase_state=ideal_target_phase,
            target_phase_state=fast_phase,
            source_metric=self.human_phase_engine.metric_dist
        )

        current_friction = friction_res["total_cognitive_friction"]

        # 4. Record Meta-Causal Trajectory Tensor Step
        error_vec = (ideal_target_phase - fast_phase)
        self.trajectory_tensor.record_step(
            phase_state=fast_phase,
            friction=current_friction,
            resonance=order_r,
            error_vector=error_vec
        )

        traj_metrics = self.trajectory_tensor.compute_trajectory_metrics()

        # 5. Synesthetic Translation Bridge
        synesthetic_bridge = self.translation_engine.generate_synesthetic_bridge(
            source_phase=ideal_target_phase,
            target_phase=fast_phase,
            friction_analysis=friction_res,
            context_label=context_name
        )

        # Apply bridge torque feedback to human phase engine
        bridge_steering = synesthetic_bridge["bridge_steering_vector"]
        self.human_phase_engine.fast_phase = (
            self.human_phase_engine.fast_phase + 0.1 * bridge_steering
        ) % (2 * np.pi)

        # 6. Check for Causal Reverse-Engineering Anchoring if converged
        anchoring_result = None
        if traj_metrics["is_converged"]:
            anchoring_result = self.reverse_eng_engine.execute_self_explanation_loop(
                target_name=context_name,
                output_payload=fast_phase.tolist(),
                context_description=f"Converged trajectory with total struggle {traj_metrics['total_friction_struggle']:.3f}"
            )

        # 7. Update Hebbian Plasticity
        self.human_phase_engine.metric_dist = self.plasticity_engine.update_metric(
            current_metric=self.human_phase_engine.metric_dist,
            slow_phase=slow_phase,
            fast_phase=fast_phase
        )

        # Compute Isomorphism Score (Symmetry between human phase order and CausalField energy state)
        cf_topology = self.causal_field.get_topology()
        cf_potentials = [v["potential"] for v in cf_topology["voxels"].values()]
        mean_cf_potential = float(np.mean(cf_potentials)) if cf_potentials else 0.0

        self.mirror_isomorphism_score = float(
            max(0.0, 1.0 - abs(delta_phi_fast - mean_cf_potential))
        )

        return {
            "context_name": context_name,
            "order_parameter_R": order_r,
            "delta_phi_fast": delta_phi_fast,
            "cognitive_friction": friction_res,
            "trajectory_metrics": traj_metrics,
            "synesthetic_bridge": synesthetic_bridge,
            "anchoring_result": anchoring_result,
            "isomorphism_score": self.mirror_isomorphism_score,
            "causal_field_summary": {
                "num_voxels": len(self.causal_field.voxels),
                "num_beams": len(self.causal_field.beams),
                "total_dissipated_energy": self.causal_field.total_dissipated_energy
            }
        }


class MetaCausalEpistemicMirror:
    """
    [MetaCausalEpistemicMirror: 최상위 인지 거울 통제 엔진]
    Orchestrates the entire Meta-Causal Epistemic Mirror framework,
    binding human cognitive phase dynamics, causal reverse engineering,
    meta subjectivity, and the causal field into a unified living mirror runtime.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.mirror_layer = IsomorphicMirrorLayer(dimension=dimension)
        self.creation_timestamp = time.time()
        self.reflection_counter = 0

    def process_epistemic_reflection(
        self,
        sensory_input: Optional[np.ndarray] = None,
        concept_label: str = "Meta Causal Discovery"
    ) -> Dict[str, Any]:
        """Runs one full cycle of epistemic mirror reflection."""
        self.reflection_counter += 1
        return self.mirror_layer.reflect_and_synchronize(
            external_human_input_signal=sensory_input,
            context_name=f"{concept_label} (Cycle #{self.reflection_counter})"
        )

    def get_epistemic_status(self) -> Dict[str, Any]:
        """Returns complete status of the epistemic mirror architecture."""
        return {
            "reflection_counter": self.reflection_counter,
            "dimension": self.dimension,
            "isomorphism_score": self.mirror_layer.mirror_isomorphism_score,
            "internal_territory_radius": self.mirror_layer.reverse_eng_engine.internal_territory_radius,
            "growth_rings_count": len(self.mirror_layer.reverse_eng_engine.growth_rings),
            "trajectory_history_len": len(self.mirror_layer.trajectory_tensor.phase_trajectory_history),
            "is_trajectory_converged": self.mirror_layer.trajectory_tensor.is_converged
        }
