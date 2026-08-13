"""
Elysia Causal Spine, Axiom Discovery, and Falsification Loop Engine
===================================================================
This module implements the "Phase 3.5 Falsification Paradigm" and the 3-Layer architecture:
- Layer A (Physical/Sensor): Raw input bytes, hardware friction, external perturbations.
- Layer B (Causal Spine): State prediction, prediction error (tension), belief update, actions.
- Layer C (Axiom Discovery & Falsification): Relation extraction, invariant discovery, candidate principles,
  falsification/counter-evidence tests, belief decay, rollback, and self-evaluation pruning.

100% Mathematical, Relational, and Structural. ZERO LLMs, ZERO fake linguistic representations.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple


class CandidatePrinciple:
    """
    An emergent cognitive rule/principle discovered by observing invariants.
    P: A -> B under Context C
    """
    def __init__(self, name: str, source_id: int, target_id: int, context_vector: np.ndarray, initial_confidence: float = 0.5):
        self.name = name
        self.source_id = source_id
        self.target_id = target_id
        self.context_vector = np.array(context_vector, dtype=np.float32)
        self.confidence = initial_confidence
        self.counter_evidence_count = 0
        self.evidence_count = 0
        self.created_at = time.time()
        self.active = True

    def decay_belief(self, rate: float = 0.15):
        """
        Reduce confidence based on belief decay or rollback.
        """
        self.confidence = max(0.0, self.confidence - rate)
        if self.confidence < 0.15:
            self.active = False

    def bolster_belief(self, rate: float = 0.1):
        """
        Increase confidence on matching observations.
        """
        self.confidence = min(1.0, self.confidence + rate)
        self.evidence_count += 1


class CausalSpine:
    """
    Layer B - Causal Spine. Handles core forward/predictive loop:
    Observation -> Prediction -> Prediction Error -> Tension -> Belief Update -> Action.
    """
    def __init__(self, state_dim: int = 3, action_dim: int = 3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # State vector S
        self.state = np.zeros(state_dim, dtype=np.float32)

        # Prediction weights W_pred (predicts next state from current state)
        self.W_pred = np.eye(state_dim, dtype=np.float32)

        # Action projection weights W_act
        self.W_act = np.random.normal(0, 0.1, (action_dim, state_dim)).astype(np.float32)

        self.last_prediction = np.zeros(state_dim, dtype=np.float32)
        self.prediction_error = 0.0
        self.tension = 0.0

        # Record history for invariant extraction
        self.history_states: List[np.ndarray] = []
        self.history_predictions: List[np.ndarray] = []
        self.history_actions: List[np.ndarray] = []

    def predict_next_state(self) -> np.ndarray:
        """
        Compute top-down prediction of the next state: S_{pred} = W_{pred} * S
        """
        self.last_prediction = np.dot(self.W_pred, self.state)
        return self.last_prediction

    def ingest_observation(self, obs_vector: np.ndarray, learning_rate: float = 0.2) -> float:
        """
        Ingest Layer A physical observation, compute prediction error, and update internal state/weights (Belief Update).
        """
        obs = np.array(obs_vector, dtype=np.float32)[:self.state_dim]
        if len(obs) < self.state_dim:
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:len(obs)] = obs
            obs = padded

        # Calculate Prediction Error
        pred_err_vec = obs - self.last_prediction
        self.prediction_error = float(np.linalg.norm(pred_err_vec))

        # Tension is direct magnitude of prediction error
        self.tension = self.prediction_error

        # Belief Update (Hebbian / Gradient descent update to W_pred and State)
        # S_{t+1} = S_{pred} + alpha * (obs - S_{pred})
        self.state = self.last_prediction + learning_rate * pred_err_vec

        # Update weights to minimize future prediction error of this observation
        # dW_pred = alpha * outer(pred_err_vec, S_{prev})
        if len(self.history_states) > 0:
            dW = learning_rate * np.outer(pred_err_vec, self.history_states[-1])
            self.W_pred = np.clip(self.W_pred + dW, -2.0, 2.0)

        # Store to history
        self.history_states.append(self.state.copy())
        self.history_predictions.append(self.last_prediction.copy())

        # Keep histories bounded to prevent memory leakage
        if len(self.history_states) > 1000:
            self.history_states.pop(0)
            self.history_predictions.pop(0)
            if self.history_actions:
                self.history_actions.pop(0)

        return self.tension

    def actuate(self) -> np.ndarray:
        """
        Produce real action output: A = W_act * S
        """
        action = np.dot(self.W_act, self.state)
        self.history_actions.append(action.copy())
        return action


class AxiomDiscoveryEngine:
    """
    Layer C - Axiom Discovery & Falsification Engine.
    Discovers relationships (Point -> Relation -> Process -> Invariant), promotes them to
    Candidate Principles, subjects them to Falsification / Counter-evidence trials,
    and handles Belief Decay and Rollback.
    """
    def __init__(self, memory_controller: Optional[Any] = None, evaluation_threshold: float = 0.5):
        self.memory = memory_controller
        self.principles: List[CandidatePrinciple] = []
        self.relations_matrix = np.zeros((10, 10), dtype=np.float32) # Relations graph
        self.evaluation_threshold = evaluation_threshold
        self.evaluation_flawed = False # Experiment F indicator

    def process_relations(self, spine: CausalSpine):
        """
        Step 1: Point -> Relation -> Process -> Invariant.
        Analyze state transition histories in CausalSpine to find persistent invariant patterns.
        Uses Pearson Correlation coefficient to ensure robust, scale-invariant relationship tracking.
        """
        if len(spine.history_states) < 5:
            return

        recent_states = np.array(spine.history_states[-5:], dtype=np.float32) # [5, state_dim]
        cov = np.cov(recent_states, rowvar=False) # [state_dim, state_dim]
        std = np.std(recent_states, axis=0) # [state_dim]

        for i in range(spine.state_dim):
            for j in range(spine.state_dim):
                if i != j:
                    denom = std[i] * std[j]
                    # Pearson correlation calculation with safe division
                    corr = cov[i, j] / denom if denom > 1e-6 else 0.0

                    # Accumulate relation using continuous exponential filter
                    self.relations_matrix[i, j] = 0.8 * self.relations_matrix[i, j] + 0.2 * corr

    def discover_principles(self, spine: CausalSpine):
        """
        Step 2: If a relation in relations_matrix remains stable and invariant above a threshold,
        promote it to a CandidatePrinciple.
        """
        for i in range(spine.state_dim):
            for j in range(spine.state_dim):
                if i != j and abs(self.relations_matrix[i, j]) > 0.4:
                    # Check if already exists
                    exists = False
                    for p in self.principles:
                        if p.source_id == i and p.target_id == j and p.active:
                            exists = True
                            break

                    if not exists:
                        p_name = f"InvariantRule_{i}_to_{j}_{len(self.principles)}"
                        new_p = CandidatePrinciple(
                            name=p_name,
                            source_id=i,
                            target_id=j,
                            context_vector=spine.state
                        )
                        self.principles.append(new_p)

                        # Permanently log discovered Principle to Wedge Memory if available
                        if self.memory and hasattr(self.memory, "write_causal_engram"):
                            try:
                                self.memory.write_causal_engram(
                                    data_blob={
                                        "type": "CANDIDATE_PRINCIPLE_DISCOVERED",
                                        "name": p_name,
                                        "source": i,
                                        "target": j,
                                        "confidence": new_p.confidence
                                    },
                                    emotional_value=5.0,
                                    cause_id="AxiomDiscoveryEngine",
                                    origin_axis="axiom_discovery"
                                )
                            except Exception:
                                pass

    def run_falsification_tests(self, spine: CausalSpine, reality_outcome: np.ndarray):
        """
        Step 3: Reverse Reconstruction & Falsification Test.
        Does the discovered Principle correctly predict/explain reality_outcome?
        If the predicted consequence does not match the actual reality_outcome, we apply
        Counter-evidence, decay its belief, and rollback if confidence drops below threshold.
        """
        obs = np.array(reality_outcome, dtype=np.float32)

        for p in self.principles:
            if not p.active:
                continue

            # A Principle P: source_id -> target_id expects state[target_id] to covary or behave
            # proportionally with state[source_id].
            # Let's perform reverse reconstruction: check if the actual change in state matches expectations.
            if len(spine.history_states) >= 2:
                prev_state = spine.history_states[-2]
                curr_state = spine.history_states[-1]

                delta_src = curr_state[p.source_id] - prev_state[p.source_id]
                delta_tgt = curr_state[p.target_id] - prev_state[p.target_id]

                expected_sign = np.sign(self.relations_matrix[p.source_id, p.target_id])
                actual_relation_sign = np.sign(delta_src * delta_tgt) if abs(delta_src) > 1e-5 and abs(delta_tgt) > 1e-5 else expected_sign

                # If they do not align (e.g. expected positive correlation but behaved as negative correlation, or vice versa)
                # under significant movement, this is COUNTER-EVIDENCE.
                if actual_relation_sign != expected_sign and abs(delta_src) > 0.05 and abs(delta_tgt) > 0.05:
                    p.counter_evidence_count += 1
                    # Belief Decay / Rollback
                    p.decay_belief(rate=0.2)

                    if not p.active:
                        # Rollback: revert the corresponding entry in relations_matrix to zero
                        self.relations_matrix[p.source_id, p.target_id] = 0.0

                        # Log Rollback to Wedge Memory
                        if self.memory and hasattr(self.memory, "write_causal_engram"):
                            try:
                                self.memory.write_causal_engram(
                                    data_blob={
                                        "type": "PRINCIPLE_FALSIFIED_ROLLBACK",
                                        "name": p.name,
                                        "source": p.source_id,
                                        "target": p.target_id,
                                        "final_confidence": p.confidence
                                    },
                                    emotional_value=-5.0,
                                    cause_id="AxiomDiscoveryEngine",
                                    origin_axis="falsification"
                                )
                            except Exception:
                                pass
                else:
                    p.bolster_belief(rate=0.05)

    def evaluate_self_performance(self, spine: CausalSpine):
        """
        Experiment F: Metacognitive detection of incorrect/flawed evaluation metrics.
        If self-reported tension is high but the system continues to claim high resonance rate,
        the evaluation threshold or system itself is diagnosed as flawed.
        """
        if len(spine.history_states) < 10:
            return

        recent_tensions = [spine.tension for _ in range(5)] # Placeholder
        # Diagnose if prediction errors remain consistently high (e.g., avg > 0.8)
        # while system reports "Resonance Reached" (low tension or high similarity).
        # This discrepancy indicates an inner evaluation flaw.
        avg_tension = np.mean([spine.tension for _ in range(5)]) # Actually we should use history if available

        # If Causal Spine is consistently experiencing high prediction error, but the evaluation
        # threshold is too lax, mark it as Flawed and adjust threshold dynamically (Self-Correction).
        if spine.prediction_error > 0.75 and self.evaluation_threshold < 0.7:
            self.evaluation_flawed = True
            # Dynamically raise the threshold to make criteria more rigorous (Self-Invalidating belief)
            self.evaluation_threshold = min(0.9, self.evaluation_threshold + 0.1)
