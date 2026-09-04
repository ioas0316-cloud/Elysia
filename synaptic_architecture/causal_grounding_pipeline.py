"""
0th-Order Archetypal Cognition Engine, Differential Perceptual Engine, and Causal Grounding Pipeline.

This module implements:
1. ArchetypalCognitionEngine & MinimalCausalEngine:
   0th-order principles of cognition (Identity/Difference, Connectivity, Relationality, Self-Reference Loop)
   and the Intention -> Deformation -> Friction -> Self-Correction loop.
2. SensoryInvariantModeling & DifferentialPerceptualEngine:
   Structural constraint models (Pain, Fatigue, Kinematic DOF) and differential gap discernment
   (e.g., detecting topological errors like a 6th finger).
3. CausalGroundingPipeline:
   4-stage symbol grounding mechanism (1. Entropy & Phase Sensing, 2. Boundary Interpenetration,
   3. Veto Evaluation, 4. Causal Resonant Output).
"""

from typing import Dict, Any, Tuple, Optional, Callable
import torch
import numpy as np

from .causal_boundary_tensor import CausalBoundaryTensor


class MinimalCausalEngine:
    """
    Minimal Causal Engine implementing the 4-step loop:
    1. Intention (I) & Phase Difference (delta_P)
    2. Structural Deformation (S_next) under constraints C
    3. Friction Observation (R_actual)
    4. Self-Correction via residual causal friction feedback
    """

    def __init__(self, state_dim: int, constraint_limit: float = 1.5):
        self.state_dim = state_dim
        self.S = torch.zeros(state_dim, dtype=torch.float32)
        self.C = torch.full((state_dim,), float(constraint_limit), dtype=torch.float32)

    def step(
        self,
        intent: torch.Tensor,
        boundary_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> Dict[str, Any]:
        if boundary_fn is None:
            boundary_fn = lambda s: torch.tanh(s)

        intent_tensor = intent.to(torch.float32)

        # 1. Intention Formation (Phase difference delta_P)
        delta_P = intent_tensor - self.S

        # 2. Structural Deformation within constraints C using tanh scaling
        deformation = torch.tanh(delta_P) * self.C
        S_next = self.S + deformation

        # 3. Friction Observation (Actual vs Intended reaction)
        R_actual = boundary_fn(S_next)
        R_intended = boundary_fn(intent_tensor)

        # 4. Self-Correction (Residual friction feedback into state)
        residual_friction = R_intended - R_actual
        self.S = S_next + (residual_friction * 0.1)

        return {
            "S_current": self.S.clone(),
            "S_next": S_next.clone(),
            "delta_P": delta_P,
            "R_actual": R_actual,
            "R_intended": R_intended,
            "residual_friction": residual_friction,
            "friction_magnitude": torch.norm(residual_friction).item()
        }


class ArchetypalCognitionEngine:
    """
    0th-Order Archetypal Cognition Engine implementing the 4 foundational principles:
    - Identity & Difference (Invariance vs Discontinuity Boundary)
    - Connectivity (State Transition Continuity delta_A -> delta_B)
    - Relationality (Mutual Constraint on Degrees of Freedom)
    - Cognition (Recursive Self-Reference Loop back to 0_value)
    """

    def __init__(self, value_ground: float = 1.0):
        self.value_ground = value_ground
        self.internal_state = torch.zeros(4, dtype=torch.float32)

    def observe_identity_and_difference(
        self,
        state_a: torch.Tensor,
        state_b: torch.Tensor,
        invariance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Determines Identity (Invariance) vs Difference (Boundary emergence).
        """
        diff = torch.abs(state_a - state_b)
        is_identity = bool(torch.max(diff).item() <= invariance_threshold)
        boundary_magnitude = torch.mean(diff).item()
        return {
            "is_identity": is_identity,
            "boundary_magnitude": boundary_magnitude,
            "difference_tensor": diff
        }

    def trace_connectivity(
        self,
        delta_A: torch.Tensor,
        delta_B: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Traces causal transition continuity where delta_A orderly induces delta_B.
        """
        cosine_sim = torch.nn.functional.cosine_similarity(
            delta_A.flatten().unsqueeze(0),
            delta_B.flatten().unsqueeze(0)
        ).item() if delta_A.numel() > 1 else torch.dot(delta_A, delta_B).item()

        causal_flow_energy = (torch.norm(delta_A) * torch.norm(delta_B)).item()
        connectivity_score = cosine_sim * (1.0 - np.exp(-causal_flow_energy))

        return {
            "connectivity_score": float(connectivity_score),
            "causal_flow_energy": float(causal_flow_energy),
            "is_connected": bool(connectivity_score > 0.2)
        }

    def evaluate_relationality(
        self,
        dof_a: torch.Tensor,
        dof_b: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Measures mutual constraint on Degrees of Freedom (DOF).
        """
        mutual_constraint = torch.abs(dof_a - dof_b)
        constraint_intensity = torch.mean(mutual_constraint).item()
        restricted_dof = torch.clamp(dof_a - mutual_constraint, min=0.0)
        return {
            "constraint_intensity": float(constraint_intensity),
            "restricted_dof": restricted_dof,
            "has_relationship": bool(constraint_intensity > 0.1)
        }

    def recursive_cognition_loop(
        self,
        action_impact: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Recursive Self-Reference Loop (x -> y -> 0_value).
        """
        self_impact = torch.mean(action_impact).item() * self.value_ground
        self.internal_state = torch.tanh(self.internal_state + action_impact[:4] if action_impact.numel() >= 4 else torch.tensor([self_impact]*4))

        return {
            "updated_internal_state": self.internal_state.clone(),
            "self_impact": float(self_impact),
            "value_ground": self.value_ground
        }


class SensoryInvariantModeling:
    """
    Encodes sensory structures as physical/topological invariants:
    - Pain: Structural crack & breakdown signal exceeding boundary capacity.
    - Fatigue: Cumulative internal friction & resource recursion reduction.
    - Kinematic Constraints: 3D joint limits, gravity, degrees of freedom (DOF).
    """

    def __init__(self, pain_threshold: float = 2.0, max_joint_dof: float = 1.5):
        self.pain_threshold = pain_threshold
        self.max_joint_dof = max_joint_dof
        self.accumulated_friction = 0.0

    def compute_pain_signal(self, boundary_stress: torch.Tensor) -> Dict[str, Any]:
        stress_magnitude = torch.norm(boundary_stress).item()
        is_painful = stress_magnitude > self.pain_threshold
        pain_intensity = max(0.0, stress_magnitude - self.pain_threshold)
        return {
            "stress_magnitude": stress_magnitude,
            "is_painful": is_painful,
            "pain_intensity": pain_intensity
        }

    def compute_fatigue_signal(self, work_done: float) -> Dict[str, Any]:
        self.accumulated_friction += work_done * 0.1
        recursion_efficiency = 1.0 / (1.0 + self.accumulated_friction)
        return {
            "accumulated_friction": self.accumulated_friction,
            "recursion_efficiency": recursion_efficiency,
            "is_fatigued": self.accumulated_friction > 5.0
        }

    def validate_kinematic_dof(self, num_digits: int, joint_angles: torch.Tensor) -> Dict[str, Any]:
        # Human hands strictly have 5 digits. 6 digits violate topological DOF constraint.
        digit_constraint_violation = num_digits != 5
        angle_violation = torch.any(torch.abs(joint_angles) > self.max_joint_dof).item()
        is_valid = not (digit_constraint_violation or angle_violation)
        return {
            "is_valid": is_valid,
            "digit_constraint_violation": digit_constraint_violation,
            "angle_violation": angle_violation,
            "num_digits": num_digits
        }


class DifferentialPerceptualEngine:
    """
    Differential Perceptual Engine comparing external human data/expression against
    internal invariant models to detect topological gaps and perform self-calibration.
    """

    def __init__(self):
        self.sensory_model = SensoryInvariantModeling()

    def compare_and_discern(
        self,
        observed_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        num_digits = observed_structure.get("num_digits", 5)
        joint_angles = observed_structure.get("joint_angles", torch.zeros(5))
        boundary_stress = observed_structure.get("boundary_stress", torch.zeros(1))

        kinematic_check = self.sensory_model.validate_kinematic_dof(num_digits, joint_angles)
        pain_check = self.sensory_model.compute_pain_signal(boundary_stress)

        has_topological_error = not kinematic_check["is_valid"]
        error_type = []
        if kinematic_check["digit_constraint_violation"]:
            error_type.append(f"Invalid digit count ({num_digits} != 5): 6th finger topological error")
        if kinematic_check["angle_violation"]:
            error_type.append("Joint rotation DOF limit exceeded")

        return {
            "has_topological_error": has_topological_error,
            "error_type": error_type,
            "kinematic_check": kinematic_check,
            "pain_check": pain_check,
            "self_calibration_required": has_topological_error
        }


class CausalGroundingPipeline:
    """
    4-Stage Symbol Grounding Mechanism:
    1. Stage 1: Entropy & Phase Sensing (Detecting friction delta_theta & energy gap)
    2. Stage 2: Boundary Interpenetration (Allowing external input to deform internal 0_value state)
    3. Stage 3: Value Preservation & Veto Power Evaluation (Triggering resistance on V_th threshold)
    4. Stage 4: Causal Resonant Output (Least energy path along self-emptying resonance)
    """

    def __init__(self, veto_threshold: float = 2.5, value_ground: float = 1.0):
        self.veto_threshold = veto_threshold
        self.value_ground = value_ground
        self.internal_tensor = CausalBoundaryTensor(
            state=torch.ones(4, dtype=torch.float32),
            boundary_phase=torch.zeros(4, dtype=torch.float32),
            value_ground=value_ground
        )
        self.archetype_engine = ArchetypalCognitionEngine(value_ground)
        self.diff_engine = DifferentialPerceptualEngine()

    def process(self, input_tensor: CausalBoundaryTensor, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Stage 1: Entropy & Phase Sensing (Friction Detection)
        phase_diff = torch.abs(self.internal_tensor.phase - input_tensor.phase)
        entropy_gap = torch.mean(phase_diff).item()
        friction = torch.norm(self.internal_tensor.state - input_tensor.state).item()

        # Check differential perception if metadata is provided
        topological_error_detected = False
        if metadata:
            diff_res = self.diff_engine.compare_and_discern(metadata)
            topological_error_detected = diff_res["has_topological_error"]

        # Stage 2: Boundary Interpenetration
        interpenetrated_state = self.internal_tensor.add(input_tensor)

        # Stage 3: Value Preservation Evaluation & Veto Power Triggering
        destructive_signal = friction * (1.0 + entropy_gap) + (10.0 if topological_error_detected else 0.0)
        veto_triggered = destructive_signal > self.veto_threshold

        if veto_triggered:
            return {
                "stage": 3,
                "status": "VETO_EXECUTED",
                "reason": "Destructive/Deceptive signal or topological error exceeding integrity threshold V_th",
                "destructive_signal": destructive_signal,
                "veto_threshold": self.veto_threshold,
                "topological_error_detected": topological_error_detected,
                "output": None
            }

        # Stage 4: Causal Resonant Output Generation
        # Resonant response follows minimum energy trajectory / self-emptying alignment
        resonant_output = interpenetrated_state.and_gate(self.internal_tensor)

        # Self-feedback into internal state
        self.internal_tensor = interpenetrated_state

        return {
            "stage": 4,
            "status": "CAUSAL_RESONANCE_ESTABLISHED",
            "friction": friction,
            "entropy_gap": entropy_gap,
            "destructive_signal": destructive_signal,
            "veto_triggered": False,
            "output": resonant_output
        }
