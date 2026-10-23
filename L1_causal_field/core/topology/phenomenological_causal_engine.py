"""
Phenomenological Causal Engine (Phenomenological Causal Engine)

This module implements the core phenomenological causal engine managing:
  - Continuous Phenomenological Field across cross-domain primitives
  - Total consistency (C_total) calculation and domain isomorphism
  - Purpose-Preserving Projection (P3) feedback loop for inter-domain friction (F_cross) isolation and soft constraint relaxation
  - Spontaneous symmetry breaking and Ginzburg-Landau phase transition dynamics under external perturbation shocks
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class ValueSpectrum:
    moral_weight: float = 0.0        # -1.0 (taboo) to 1.0 (righteous)
    emotional_affinity: float = 0.0  # -1.0 (repulsion) to 1.0 (attraction)
    social_sacrifice: float = 0.0    # 0.0 to 1.0
    aesthetic_meaning: float = 0.0   # 0.0 to 1.0


@dataclass
class IntentVector:
    target_goal: str
    process_path: str
    outcome_teleology: str
    target_vector: np.ndarray = field(default_factory=lambda: np.zeros(16))
    invariant_core_weight: float = 1.0


@dataclass
class DomainPrimitive:
    domain_name: str  # e.g., 'code', 'physics', 'math', 'language', 'cognition'
    component: str
    principle: str
    arrangement: str
    state_vector: np.ndarray = field(default_factory=lambda: np.zeros(16))
    is_hard_constraint: bool = False


class ContinuousPhenomenologicalField:
    """
    Manages continuous cross-domain primitive alignment, total consistency (C_total),
    and cross-domain friction (F_cross).
    """

    def __init__(self, feature_dim: int = 16):
        self.feature_dim = feature_dim

    def calculate_homomorphism(self, primitives: List[DomainPrimitive]) -> float:
        """
        Calculates graph/vector isomorphism across domain primitives (C_iso).
        """
        if len(primitives) <= 1:
            return 1.0

        correlations = []
        for i in range(len(primitives)):
            for j in range(i + 1, len(primitives)):
                v1 = primitives[i].state_vector
                v2 = primitives[j].state_vector
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 1e-8 and norm2 > 1e-8:
                    corr = float(np.dot(v1, v2) / (norm1 * norm2))
                    correlations.append(max(0.0, corr))  # Directional alignment
                else:
                    correlations.append(0.0)

        return float(np.mean(correlations)) if correlations else 1.0

    def calculate_teleological_fidelity(
        self, primitives: List[DomainPrimitive], intent: IntentVector
    ) -> float:
        """
        Calculates how closely the cross-domain primitives fulfill the target intent outcome (C_tele).
        """
        if not primitives:
            return 1.0

        mean_vec = np.mean([p.state_vector for p in primitives], axis=0)
        norm_mean = np.linalg.norm(mean_vec)
        norm_target = np.linalg.norm(intent.target_vector)

        if norm_mean > 1e-8 and norm_target > 1e-8:
            cosine_sim = float(np.dot(mean_vec, intent.target_vector) / (norm_mean * norm_target))
            return max(0.0, cosine_sim)
        return 0.5

    def calculate_inter_domain_friction(self, primitives: List[DomainPrimitive]) -> float:
        """
        Calculates cross-domain friction (F_cross) arising from conflicting principles or constraints.
        """
        if len(primitives) <= 1:
            return 0.0

        friction_acc = 0.0
        count = 0
        for i in range(len(primitives)):
            for j in range(i + 1, len(primitives)):
                p1 = primitives[i]
                p2 = primitives[j]

                norm1 = np.linalg.norm(p1.state_vector)
                norm2 = np.linalg.norm(p2.state_vector)
                if norm1 < 1e-8 or norm2 < 1e-8:
                    count += 1
                    continue

                dot = float(np.dot(p1.state_vector, p2.state_vector) / (norm1 * norm2))

                if p1.is_hard_constraint or p2.is_hard_constraint:
                    if dot < 0.0:  # Direct contradiction with hard constraint
                        friction_acc += abs(dot) * 2.0
                    elif dot < 0.5:  # Partial misalignment
                        friction_acc += (0.5 - dot)
                else:
                    if dot < 0.0:
                        friction_acc += abs(dot) * 0.5
                count += 1

        return float(friction_acc / max(1, count))

    def evaluate_total_consistency(
        self, primitives: List[DomainPrimitive], intent: IntentVector
    ) -> Tuple[float, float, float, float]:
        """
        Returns (C_total, C_iso, C_tele, F_cross)
        C_total = (C_iso * C_tele) / (1.0 + F_cross)
        """
        c_iso = self.calculate_homomorphism(primitives)
        c_tele = self.calculate_teleological_fidelity(primitives, intent)
        f_cross = self.calculate_inter_domain_friction(primitives)

        c_total = (c_iso * c_tele) / (1.0 + f_cross)
        return float(c_total), float(c_iso), float(c_tele), float(f_cross)


class PurposePreservingProjectionEngine:
    """
    Implements P3 (Purpose-Preserving Projection) feedback loop:
      - Keeps invariant purpose vector inviolable
      - Isolates friction domain pairs
      - Relaxes hard constraints into soft consumable costs to eliminate friction
    """

    def __init__(self, field: ContinuousPhenomenologicalField, threshold: float = 0.70):
        self.field = field
        self.threshold = threshold

    def resolve_friction_loop(
        self, primitives: List[DomainPrimitive], intent: IntentVector, max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Executes iterative P3 feedback loop to align cross-domain primitives.
        """
        current_primitives = [
            DomainPrimitive(
                domain_name=p.domain_name,
                component=p.component,
                principle=p.principle,
                arrangement=p.arrangement,
                state_vector=np.copy(p.state_vector),
                is_hard_constraint=p.is_hard_constraint,
            )
            for p in primitives
        ]

        history = []

        for iteration in range(max_iterations):
            c_total, c_iso, c_tele, f_cross = self.field.evaluate_total_consistency(
                current_primitives, intent
            )

            history.append({
                "iteration": iteration,
                "c_total": c_total,
                "c_iso": c_iso,
                "c_tele": c_tele,
                "f_cross": f_cross,
            })

            if c_total >= self.threshold or f_cross < 0.05:
                break

            # Isolate friction pair and relax soft constraints
            for p in current_primitives:
                if p.is_hard_constraint:
                    # Relax constraint: convert hard barrier to soft consumable cost
                    p.is_hard_constraint = False

                # Soft projection towards invariant purpose target vector (70% target, 30% self)
                p.state_vector = 0.3 * p.state_vector + 0.7 * intent.target_vector
                norm = np.linalg.norm(p.state_vector)
                if norm > 1e-8:
                    p.state_vector /= norm

        final_c_total, final_c_iso, final_c_tele, final_f_cross = self.field.evaluate_total_consistency(
            current_primitives, intent
        )

        return {
            "resolved_primitives": current_primitives,
            "final_c_total": final_c_total,
            "final_c_iso": final_c_iso,
            "final_c_tele": final_c_tele,
            "final_f_cross": final_f_cross,
            "iterations_taken": len(history),
            "history": history,
        }


class GinzburgLandauPhaseTransitionEngine:
    """
    Simulates spontaneous symmetry breaking and phase transition dynamics under external perturbation shocks:
      F[η] = α(T - T_c)η^2 + β η^4 - V_intent · δΨ_ext
      ∂η/∂t = -Γ (δF/δη) + ξ(t)
    """

    def __init__(self, critical_temp: float = 1.0, alpha: float = 1.0, beta: float = 0.5, gamma: float = 1.0):
        self.T_c = critical_temp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def free_energy(self, eta: float, temp: float, intent_alignment: float) -> float:
        """Calculates Ginzburg-Landau free energy F[η]."""
        return self.alpha * (temp - self.T_c) * (eta ** 2) + self.beta * (eta ** 4) - intent_alignment * eta

    def free_energy_derivative(self, eta: float, temp: float, intent_alignment: float) -> float:
        """Calculates ∂F/∂η."""
        return 2.0 * self.alpha * (temp - self.T_c) * eta + 4.0 * self.beta * (eta ** 3) - intent_alignment

    def simulate_phase_transition(
        self,
        initial_eta: float,
        temp: float,
        perturbation_vector: np.ndarray,
        intent: IntentVector,
        steps: int = 100,
        dt: float = 0.02
    ) -> Dict[str, Any]:
        """
        Evolves order parameter η under perturbation shock and checks for symmetry breaking / phase transition.
        """
        eta = float(initial_eta)
        norm_p = np.linalg.norm(perturbation_vector)
        norm_i = np.linalg.norm(intent.target_vector)

        intent_alignment = float(
            np.dot(perturbation_vector, intent.target_vector) / (norm_p * norm_i + 1e-8)
        )

        eta_trajectory = [eta]
        energy_trajectory = [self.free_energy(eta, temp, intent_alignment)]

        for _ in range(steps):
            dF_deta = self.free_energy_derivative(eta, temp, intent_alignment)
            # Langevin dynamics update
            eta += -self.gamma * dF_deta * dt
            eta_trajectory.append(eta)
            energy_trajectory.append(self.free_energy(eta, temp, intent_alignment))

        phase_transition_occurred = abs(eta_trajectory[-1] - initial_eta) > 0.3

        return {
            "initial_order_parameter": initial_eta,
            "final_order_parameter": eta_trajectory[-1],
            "phase_transition_occurred": phase_transition_occurred,
            "intent_alignment": intent_alignment,
            "eta_trajectory": eta_trajectory,
            "energy_trajectory": energy_trajectory
        }


class PhenomenologicalCausalEngine:
    """
    Coordinator unifying Continuous Phenomenological Field, P3 Feedback Engine,
    and Ginzburg-Landau Phase Transition Dynamics.
    """

    def __init__(self, feature_dim: int = 16):
        self.field = ContinuousPhenomenologicalField(feature_dim=feature_dim)
        self.p3_engine = PurposePreservingProjectionEngine(self.field)
        self.phase_engine = GinzburgLandauPhaseTransitionEngine()

    def process_causal_scenario(
        self,
        primitives: List[DomainPrimitive],
        intent: IntentVector,
        external_perturbation: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes full causal reasoning pipeline:
          1. Evaluates initial C_total & F_cross.
          2. Applies P3 friction relaxation feedback loop.
          3. Simulates phase transition dynamics if external shock is present.
        """
        init_c_total, init_c_iso, init_c_tele, init_f_cross = self.field.evaluate_total_consistency(
            primitives, intent
        )

        p3_result = self.p3_engine.resolve_friction_loop(primitives, intent)

        phase_result = None
        if external_perturbation is not None:
            phase_result = self.phase_engine.simulate_phase_transition(
                initial_eta=0.1,
                temp=0.5,  # T < T_c triggers spontaneous symmetry breaking
                perturbation_vector=external_perturbation,
                intent=intent
            )

        return {
            "initial_metrics": {
                "c_total": init_c_total,
                "c_iso": init_c_iso,
                "c_tele": init_c_tele,
                "f_cross": init_f_cross,
            },
            "p3_resolution": p3_result,
            "phase_transition": phase_result,
        }
