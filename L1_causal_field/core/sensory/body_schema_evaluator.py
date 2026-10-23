"""
Body Schema Evaluator & Active Inference Loop (신체 도식 및 능동적 추론 평가기)

Implements:
1. Active Inference & Efference Copy (원심 복사: S_pred vs S_in)
2. Self/Environment Boundary Differentiation (epsilon = S_in - S_pred)
3. Interoception (내수용 감각) & Internal Strain Dynamics (tau_strain)
4. Irreversible Physical Scar Accumulation (Delta S_scar)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class ActionPrediction:
    action_id: str
    action_vector: torch.Tensor             # Action movement or cognitive output
    predicted_sensory_outcome: torch.Tensor  # Efference Copy S_pred
    timestamp: float


@dataclass
class BodySchemaBoundaryState:
    prediction_error_norm: float
    is_self_agency: bool                     # True if error < threshold (agency of self)
    is_external_shock: bool                  # True if error >= threshold (foreign environment)
    internal_strain: float                   # tau_strain
    self_boundary_confidence: float          # Confidence [0, 1] in self vs world boundary
    scar_impact_vector: torch.Tensor          # Irreversible physical scar tensor


class BodySchemaEvaluator:
    """
    Evaluates Active Inference, Efference Copy error epsilon, internal tension tau_strain,
    and dynamically differentiates Self Agency from External World Shocks.
    """

    def __init__(
        self,
        dimension: int = 64,
        self_boundary_threshold: float = 1.0,
        strain_decay_rate: float = 0.05,
        dtype=torch.float32
    ):
        self.dimension = dimension
        self.self_boundary_threshold = self_boundary_threshold
        self.strain_decay_rate = strain_decay_rate
        self.dtype = dtype

        # Internal state
        self.internal_strain: float = 0.0
        self.accumulated_scar_tensor: torch.Tensor = torch.zeros(dimension, dtype=dtype)
        self.last_efference_copy: Optional[ActionPrediction] = None

        # Internal forward prediction weights (Action -> S_pred)
        self.forward_model = torch.nn.Linear(dimension, dimension, bias=False, dtype=dtype)
        torch.nn.init.eye_(self.forward_model.weight) # Initialize near identity

    def generate_efference_copy(
        self,
        action_id: str,
        action_vector: torch.Tensor,
        timestamp: float
    ) -> ActionPrediction:
        """
        Generates Efference Copy (S_pred) when system executes an action or cognitive state transition.
        """
        act_t = action_vector.to(self.dtype)
        if act_t.shape[0] != self.dimension:
            act_t = torch.nn.functional.interpolate(
                act_t.unsqueeze(0).unsqueeze(0),
                size=self.dimension
            ).squeeze()

        with torch.no_grad():
            s_pred = self.forward_model(act_t)

        prediction = ActionPrediction(
            action_id=action_id,
            action_vector=act_t,
            predicted_sensory_outcome=s_pred,
            timestamp=timestamp
        )
        self.last_efference_copy = prediction
        return prediction

    def evaluate_afferent_feedback(
        self,
        actual_sensory_input: torch.Tensor,
        external_friction_impedance: float = 0.0
    ) -> BodySchemaBoundaryState:
        """
        Compares incoming afferent feedback S_in with stored Efference Copy S_pred.
        Differentiates Self-Agency vs External World Friction/Shock.
        """
        s_in = actual_sensory_input.to(self.dtype)
        if s_in.shape[0] != self.dimension:
            s_in = torch.nn.functional.interpolate(
                s_in.unsqueeze(0).unsqueeze(0),
                size=self.dimension
            ).squeeze()

        if self.last_efference_copy is not None:
            s_pred = self.last_efference_copy.predicted_sensory_outcome
        else:
            s_pred = torch.zeros_like(s_in)

        # Afferent prediction error vector epsilon = S_in - S_pred
        error_vec = s_in - s_pred
        error_norm = float(torch.norm(error_vec).item())

        # Self boundary classification
        is_self = error_norm < self.self_boundary_threshold
        is_shock = not is_self

        # Calculate confidence
        confidence = 1.0 - min(1.0, abs(error_norm - self.self_boundary_threshold) / (self.self_boundary_threshold + 1e-8))

        # Interoceptive strain accumulation: tau_strain increases with error and external friction
        added_strain = error_norm * 0.5 + external_friction_impedance * 0.5
        self.internal_strain += added_strain
        # Strain decay towards baseline
        self.internal_strain = max(0.0, self.internal_strain * (1.0 - self.strain_decay_rate))

        # Irreversible Scar Tensor deposition
        scar_delta = torch.abs(error_vec) * (0.1 if is_shock else 0.01)
        self.accumulated_scar_tensor += scar_delta

        # Adapt forward model slightly if action was self-generated
        if is_self and self.last_efference_copy is not None:
            with torch.no_grad():
                # Minor self-calibration of active inference prediction
                grad_model = torch.outer(error_vec, self.last_efference_copy.action_vector)
                self.forward_model.weight += 0.01 * grad_model

        return BodySchemaBoundaryState(
            prediction_error_norm=error_norm,
            is_self_agency=is_self,
            is_external_shock=is_shock,
            internal_strain=self.internal_strain,
            self_boundary_confidence=confidence,
            scar_impact_vector=scar_delta
        )
