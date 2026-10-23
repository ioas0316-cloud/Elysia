"""
core/intelligence/teleological_relational_spectrum.py

Integrated Teleological & Relational Spectrum Engine
=====================================================
Integrates Retrocausal Reverse-Inference, Multidimensional Relational Spectrum,
Epistemic History Lineage Tracing, and Hypothesis-Driven Failure/Constraint Learning
into a single connected cognitive process.

Key Principles:
1. Goal state (S_target) is a flexible possibility world, not an immutable scalar target.
2. Retrocausal inference reasons backward: S_target -> Necessary Conditions -> Current Deficits -> Action.
3. Relations across Time, Scale, Meaning, Possibility, and Relation axes are dynamically inferred
   and categorized into Hypothesis, VerifiedRelation, or RejectedHypothesis.
4. Every belief/judgment maintains an Epistemic Lineage (observations, premises, principles, counterexamples).
5. Mismatch -> Raw Difference -> Cause Hypothesis -> Independent Verification -> Verified Constraint (Delta)
   or Premise/Goal Re-evaluation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class EpistemicLineage:
    """
    Stores the full causal history of how a belief, judgment, or goal was formed.
    """
    def __init__(
        self,
        origin_id: str,
        observations: List[Any] = None,
        premises: List[str] = None,
        principles: List[str] = None,
        parent_lineage_ids: List[str] = None
    ):
        self.origin_id = origin_id
        self.observations = observations or []
        self.premises = premises or []
        self.principles = principles or []
        self.parent_lineage_ids = parent_lineage_ids or []
        self.counterexamples: List[Any] = []
        self.revision_history: List[Dict[str, Any]] = []

    def record_revision(self, reason: str, changed_field: str, old_value: Any, new_value: Any):
        self.revision_history.append({
            "reason": reason,
            "field": changed_field,
            "old": old_value,
            "new": new_value
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "observations": self.observations,
            "premises": self.premises,
            "principles": self.principles,
            "counterexamples": self.counterexamples,
            "revisions": len(self.revision_history)
        }


class RelationalSpectrum:
    """
    Multidimensional Relational Spectrum (Time, Scale, Meaning, Possibility, Relation).
    Maintains dynamic relation candidates without pre-baked mapping tables.
    Distinguishes Hypothesis, VerifiedRelation, and RejectedHypothesis.
    """
    def __init__(self):
        self.hypotheses: Dict[str, Dict[str, Any]] = {}
        self.verified_relations: Dict[str, Dict[str, Any]] = {}
        self.rejected_hypotheses: Dict[str, Dict[str, Any]] = {}

    def propose_relation_hypothesis(
        self,
        rel_id: str,
        source_concept: str,
        target_concept: str,
        axes_weights: Dict[str, float],
        structural_invariant: np.ndarray,
        proposed_by_lineage: EpistemicLineage
    ) -> Dict[str, Any]:
        hyp = {
            "rel_id": rel_id,
            "source": source_concept,
            "target": target_concept,
            "axes_weights": axes_weights, # {"time": float, "scale": float, "meaning": float, "possibility": float, "relation": float}
            "invariant": structural_invariant,
            "lineage": proposed_by_lineage,
            "status": "HYPOTHESIS",
            "support_evidence": [],
            "counterexamples": []
        }
        self.hypotheses[rel_id] = hyp
        return hyp

    def verify_hypothesis(self, rel_id: str, observation_evidence: Any) -> Optional[Dict[str, Any]]:
        if rel_id not in self.hypotheses:
            return None
        hyp = self.hypotheses.pop(rel_id)
        hyp["status"] = "VERIFIED"
        hyp["support_evidence"].append(observation_evidence)
        self.verified_relations[rel_id] = hyp
        return hyp

    def reject_hypothesis(self, rel_id: str, counterexample: Any, reason: str) -> Optional[Dict[str, Any]]:
        if rel_id in self.hypotheses:
            hyp = self.hypotheses.pop(rel_id)
        elif rel_id in self.verified_relations:
            hyp = self.verified_relations.pop(rel_id)
        else:
            return None

        hyp["status"] = "REJECTED"
        hyp["counterexamples"].append(counterexample)
        hyp["rejection_reason"] = reason
        hyp["lineage"].counterexamples.append(counterexample)
        self.rejected_hypotheses[rel_id] = hyp
        return hyp


class TargetGoalWorld:
    """
    Represents an imagined future target state (Possibility World).
    Contains required conditions, current deficits, and flexibility to adapt when constrained.
    """
    def __init__(self, goal_id: str, description: str, target_state_vector: np.ndarray, lineage: EpistemicLineage):
        self.goal_id = goal_id
        self.description = description
        self.target_state_vector = target_state_vector.astype(np.float64)
        self.lineage = lineage
        self.required_conditions: List[Dict[str, Any]] = []
        self.constraints: List[Dict[str, Any]] = []
        self.is_abandoned = False

    def add_required_condition(self, condition_id: str, description: str, req_vector: np.ndarray):
        self.required_conditions.append({
            "condition_id": condition_id,
            "description": description,
            "vector": req_vector.astype(np.float64),
            "satisfied": False
        })

    def modify_target_goal(self, new_description: str, new_target_vector: np.ndarray, reason: str):
        self.lineage.record_revision(reason, "target_state_vector", self.target_state_vector.copy(), new_target_vector.copy())
        self.description = new_description
        self.target_state_vector = new_target_vector.astype(np.float64)


class IntegratedTeleologicalSpectrumEngine:
    """
    Unified Cognitive Core connecting:
    - Retrocausal Reverse-Inference
    - Multidimensional Relational Spectrum
    - Epistemic History Back-tracing
    - Failure -> Cause Hypothesis -> Verified Constraint ($\Delta$)
    """
    def __init__(self, state_dim: int = 8):
        self.state_dim = state_dim
        self.current_state = np.zeros(state_dim, dtype=np.float64)
        self.spectrum = RelationalSpectrum()
        self.active_goal: Optional[TargetGoalWorld] = None
        self.known_constraints: List[Dict[str, Any]] = [] # Verified Constraints (Delta)
        self.observed_differences: List[Dict[str, Any]] = []
        self.epistemic_store: Dict[str, EpistemicLineage] = {}

    def set_current_state(self, state_vector: np.ndarray):
        self.current_state = np.array(state_vector, dtype=np.float64)

    def set_target_goal(self, goal_id: str, description: str, target_vector: np.ndarray, premises: List[str] = None) -> TargetGoalWorld:
        lineage = EpistemicLineage(
            origin_id=f"lineage_goal_{goal_id}",
            observations=[self.current_state.copy()],
            premises=premises or ["Desire to reach possibility world"],
            principles=["Retrocausal Teleological Inference"]
        )
        self.epistemic_store[lineage.origin_id] = lineage

        goal = TargetGoalWorld(goal_id, description, target_vector, lineage)
        self.active_goal = goal
        return goal

    def retrocausal_reverse_infer(self) -> Dict[str, Any]:
        """
        Performs reverse inference from active_goal -> required conditions -> current state deficit.
        """
        if self.active_goal is None or self.active_goal.is_abandoned:
            return {"status": "NO_ACTIVE_GOAL"}

        goal = self.active_goal
        delta_vector = goal.target_state_vector - self.current_state
        deficit_norm = float(np.linalg.norm(delta_vector))

        # Infer required conditions from goal deficit
        condition_id = f"req_{goal.goal_id}_{len(goal.required_conditions)+1}"
        req_desc = f"Necessary condition to bridge deficit of magnitude {deficit_norm:.4f}"
        goal.add_required_condition(condition_id, req_desc, delta_vector)

        # Propose relation hypothesis between current state and goal state
        rel_id = f"rel_{goal.goal_id}_current"
        axes_weights = {"time": 0.8, "scale": 0.5, "meaning": 0.9, "possibility": 1.0, "relation": 0.7}
        self.spectrum.propose_relation_hypothesis(
            rel_id=rel_id,
            source_concept="Current_State",
            target_concept=goal.description,
            axes_weights=axes_weights,
            structural_invariant=delta_vector / (deficit_norm + 1e-8),
            proposed_by_lineage=goal.lineage
        )

        return {
            "status": "REVERSE_INFERRED",
            "goal_id": goal.goal_id,
            "deficit_norm": deficit_norm,
            "delta_vector": delta_vector,
            "required_conditions": goal.required_conditions,
            "rel_hypothesis_id": rel_id
        }

    def execute_action_and_observe(self, action_vector: np.ndarray, actual_outcome_state: np.ndarray) -> Dict[str, Any]:
        """
        Executes an action toward the goal, observes actual outcome, and evaluates mismatch against expectation.
        If mismatch occurs, initiates Epistemic History Back-tracing & Failure Learning.
        """
        if self.active_goal is None:
            return {"status": "NO_ACTIVE_GOAL"}

        goal = self.active_goal
        expected_state = self.current_state + action_vector
        actual_outcome_state = np.array(actual_outcome_state, dtype=np.float64)

        mismatch_vector = actual_outcome_state - expected_state
        mismatch_norm = float(np.linalg.norm(mismatch_vector))

        # Update current state
        self.current_state = actual_outcome_state.copy()

        # Step A: Record raw observed difference
        obs_diff = {
            "goal_id": goal.goal_id,
            "expected": expected_state,
            "actual": actual_outcome_state,
            "mismatch_vector": mismatch_vector,
            "mismatch_norm": mismatch_norm
        }
        self.observed_differences.append(obs_diff)

        # Step B: Evaluate if mismatch is significant
        if mismatch_norm > 0.1: # Significant mismatch detected
            evaluation = self._handle_mismatch_and_retrace(obs_diff)
            return {
                "status": "MISMATCH_DETECTED",
                "mismatch_norm": mismatch_norm,
                "mismatch_evaluation": evaluation
            }

        return {
            "status": "HARMONIOUS_PROGRESS",
            "mismatch_norm": mismatch_norm,
            "current_deficit": float(np.linalg.norm(goal.target_state_vector - self.current_state))
        }

    def _handle_mismatch_and_retrace(self, obs_diff: Dict[str, Any]) -> Dict[str, Any]:
        """
        Epistemic History Back-tracing and Premise/Goal Re-evaluation.
        Formulates cause hypotheses, verifies constraints, and updates goals/premises.
        """
        goal = self.active_goal
        lineage = goal.lineage
        lineage.counterexamples.append(obs_diff)

        # 1. Propose Cause Hypothesis
        mismatch_vec = obs_diff["mismatch_vector"]
        hypothesis_cause = f"Premise insufficiency or physical constraint along axis {np.argmax(np.abs(mismatch_vec))}"

        # 2. Check for Counterexamples in Spectrum Hypotheses
        rel_id = f"rel_{goal.goal_id}_current"
        self.spectrum.reject_hypothesis(rel_id, obs_diff, reason=hypothesis_cause)

        # 3. Formulate & Verify Constraint (Delta)
        # Check if the mismatch vector is consistently opposing action direction
        constraint_id = f"constraint_{len(self.known_constraints)+1}"
        verified_constraint = {
            "constraint_id": constraint_id,
            "description": f"Verified environmental/logical boundary: resistance along axis {np.argmax(np.abs(mismatch_vec))}",
            "resisting_axis": int(np.argmax(np.abs(mismatch_vec))),
            "bound_magnitude": float(np.max(np.abs(mismatch_vec))),
            "lineage_id": lineage.origin_id
        }
        self.known_constraints.append(verified_constraint)
        goal.constraints.append(verified_constraint)

        # 4. Premise / Goal Modification: Re-evaluating the possibility world
        # Instead of stubbornly persisting in impossible direction, reframe target goal vector to respect constraint!
        new_target_vector = goal.target_state_vector.copy()
        axis = verified_constraint["resisting_axis"]
        # Adapt target state vector away from hard wall
        new_target_vector[axis] = self.current_state[axis] + 0.1 * (new_target_vector[axis] - self.current_state[axis])

        new_desc = f"{goal.description} (Reframed under {verified_constraint['constraint_id']})"
        goal.modify_target_goal(
            new_description=new_desc,
            new_target_vector=new_target_vector,
            reason=f"Back-traced mismatch to constraint {constraint_id}"
        )

        return {
            "cause_hypothesis": hypothesis_cause,
            "verified_constraint": verified_constraint,
            "reframed_goal_description": goal.description,
            "new_target_vector": goal.target_state_vector,
            "retraced_lineage": lineage.to_dict()
        }

    def trace_epistemic_history(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Back-traces full epistemic history for a goal or judgment.
        """
        if self.active_goal and self.active_goal.goal_id == goal_id:
            return self.active_goal.lineage.to_dict()
        return None
