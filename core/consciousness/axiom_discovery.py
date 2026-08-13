"""
Axiom Discovery & Falsification Engine — Elysia Phase 3.5 Falsification Paradigm
=============================================================================
Implements the 3-Layer architecture (Layer A -> Layer B -> Layer C) with absolute
mathematical fidelity. Zero LLM or language-based framing. Uses continuous
multivariate tracking, Pearson correlation, and Active Inference to discover
invariants, promote them to Candidate Principles, test them under counter-evidence,
and perform belief decay and rollback when falsified.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class CandidatePrinciple:
    id: str
    name: str
    variables: Tuple[str, str]  # Variable pair under observation
    expected_correlation: float # Discovered baseline correlation
    confidence: float = 1.0     # Starts at 1.0, decays under counter-evidence
    history: List[float] = field(default_factory=list)
    created_at_cycle: int = 0
    stable_state_checkpoint: Dict[str, Any] = field(default_factory=dict) # State saved during high confidence

class AxiomDiscoveryEngine:
    """
    [Layer C — Axiom Discovery & Falsification]
    Tracks continuous variables, detects stable invariants, promotes them to
    Candidate Principles, subjects them to falsification tests, and triggers
    rollback when a principle is destroyed.
    """
    def __init__(self, correlation_threshold: float = 0.85, window_size: int = 10, falsification_threshold: float = 0.3):
        self.correlation_threshold = correlation_threshold
        self.window_size = window_size
        self.falsification_threshold = falsification_threshold

        # History of variable values: variable_name -> list of values
        self.history: Dict[str, List[float]] = {}
        self.principles: Dict[str, CandidatePrinciple] = {}
        self.falsified_principles: List[str] = []

        # Track if rollback was triggered in the last cycle
        self.rollback_triggered: bool = False
        self.last_rollback_checkpoint: Optional[Dict[str, Any]] = None

    def record_variables(self, variables: Dict[str, float]):
        """Records the current state of continuous variables."""
        for name, val in variables.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(float(val))
            # Limit history to 100 entries to prevent memory leak
            if len(self.history[name]) > 100:
                self.history[name].pop(0)

    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculates pure NumPy Pearson correlation coefficient."""
        if len(x) < 2:
            return 0.0
        mean_x, mean_y = np.mean(x), np.mean(y)
        std_x, std_y = np.std(x), np.std(y)
        if std_x == 0.0 or std_y == 0.0:
            return 0.0
        return float(np.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y))

    def evaluate_and_discover(self, current_cycle: int, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes historical trends to discover new invariants (Axiom Discovery)
        and evaluates active principles against incoming data (Falsification).
        """
        self.rollback_triggered = False
        logs = []

        # 1. Evaluate Active Principles (Falsification Test)
        decomposed_principles = []
        for pid, principle in list(self.principles.items()):
            v1, v2 = principle.variables
            if v1 in self.history and v2 in self.history and len(self.history[v1]) >= 2:
                # Get the latest instantaneous change direction/ratio
                dx = self.history[v1][-1] - self.history[v1][-2]
                dy = self.history[v2][-1] - self.history[v2][-2]

                # If they should be highly positively correlated, their changes should align.
                # If they are negatively correlated, they should oppose.
                # We test if the relation holds.
                expected_sign = np.sign(principle.expected_correlation)
                actual_sign = np.sign(dx * dy)

                # Check for counter-evidence: if changes are significant but sign is opposite
                if abs(dx) > 0.005 and abs(dy) > 0.005 and expected_sign != 0:
                    if expected_sign > 0 and dx * dy < 0:
                        # Positive correlation violated
                        violation_severity = 0.25 + abs(dx * dy) * 2.0
                        principle.confidence = max(0.0, principle.confidence - violation_severity)
                        logs.append({
                            "event": "COUNTER_EVIDENCE",
                            "principle": principle.name,
                            "details": f"Variables {v1} and {v2} moved in opposite directions. Confidence decayed to {principle.confidence:.4f}."
                        })
                    elif expected_sign < 0 and dx * dy > 0:
                        # Negative correlation violated
                        violation_severity = 0.25 + abs(dx * dy) * 2.0
                        principle.confidence = max(0.0, principle.confidence - violation_severity)
                        logs.append({
                            "event": "COUNTER_EVIDENCE",
                            "principle": principle.name,
                            "details": f"Variables {v1} and {v2} moved in same direction (expected negative correlation). Confidence decayed to {principle.confidence:.4f}."
                        })

                # Recover slightly if conforming
                if (expected_sign > 0 and dx * dy > 0) or (expected_sign < 0 and dx * dy < 0):
                    principle.confidence = min(1.0, principle.confidence + 0.02)

                # Falsification check
                if principle.confidence <= self.falsification_threshold:
                    decomposed_principles.append(pid)
                    self.falsified_principles.append(principle.name)
                    logs.append({
                        "event": "PRINCIPLE_FALSIFIED",
                        "principle": principle.name,
                        "details": f"Confidence dropped to {principle.confidence:.4f} (<= {self.falsification_threshold}). Falsification triggered."
                    })

        # Trigger Rollback if any active principle is falsified
        if decomposed_principles:
            self.rollback_triggered = True
            # Rollback to the stable state checkpoint of the first falsified principle
            first_falsified = self.principles[decomposed_principles[0]]
            self.last_rollback_checkpoint = first_falsified.stable_state_checkpoint

            # Evict falsified principles
            for pid in decomposed_principles:
                self.principles.pop(pid)

            logs.append({
                "event": "STATE_ROLLBACK",
                "details": "Falsification occurred. Rolling back internal states to the last stable checkpoint."
            })
            return logs

        # 2. Discover New Invariants from History
        variables_to_check = list(self.history.keys())
        n_vars = len(variables_to_check)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                v1 = variables_to_check[i]
                v2 = variables_to_check[j]

                # Skip if already a principle
                already_exists = False
                for p in self.principles.values():
                    if (p.variables == (v1, v2)) or (p.variables == (v2, v1)):
                        already_exists = True
                        break
                if already_exists:
                    continue

                # Check sliding window size
                if len(self.history[v1]) >= self.window_size:
                    vec1 = np.array(self.history[v1][-self.window_size:])
                    vec2 = np.array(self.history[v2][-self.window_size:])

                    corr = self._pearson_correlation(vec1, vec2)
                    if abs(corr) >= self.correlation_threshold:
                        # Discovered a stable invariant! Promote to Candidate Principle
                        pid = f"principle_{v1}_{v2}_{current_cycle}"
                        name = f"Invariant_{v1}_and_{v2}"

                        # Capture a clean copy of the stable state checkpoint
                        checkpoint = {k: float(v[-1]) for k, v in self.history.items()}
                        # Include external state values
                        checkpoint.update({k: v for k, v in current_state.items() if isinstance(v, (int, float))})

                        self.principles[pid] = CandidatePrinciple(
                            id=pid,
                            name=name,
                            variables=(v1, v2),
                            expected_correlation=corr,
                            created_at_cycle=current_cycle,
                            stable_state_checkpoint=checkpoint
                        )
                        logs.append({
                            "event": "INVARIANT_PROMOTED",
                            "principle": name,
                            "details": f"Discovered stable correlation ({corr:.4f}) between {v1} and {v2}. Promoted to Candidate Principle."
                        })

        return logs

class CausalSpine:
    """
    [Layer B — Causal Spine]
    Maintains a core physical-causal pathway:
    Observation -> Prediction -> Prediction Error (Tension) -> Belief Update -> Action -> Outcome
    """
    def __init__(self, dimensions: int = 3, learning_rate: float = 0.1):
        self.dimensions = dimensions
        self.learning_rate = learning_rate

        # State beliefs (S_self)
        self.belief_state = np.zeros(dimensions, dtype=np.float32)
        # Expected/predicted next state
        self.predicted_state = np.zeros(dimensions, dtype=np.float32)

        # Model parameters (linear state transition matrix)
        self.transition_matrix = np.eye(dimensions, dtype=np.float32)
        # Actuation matrix (maps state to action)
        self.actuation_matrix = np.eye(dimensions, dtype=np.float32) * 0.5

        # Latest calculated values
        self.latest_prediction_error: float = 0.0
        self.latest_action: np.ndarray = np.zeros(dimensions, dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Generates Top-Down expectation based on current beliefs."""
        self.predicted_state = np.dot(self.transition_matrix, self.belief_state)
        return self.predicted_state

    def compute_prediction_error(self, actual_observation: np.ndarray) -> float:
        """Calculates prediction error (tension) as Euclidean distance."""
        diff = actual_observation - self.predicted_state
        self.latest_prediction_error = float(np.linalg.norm(diff))
        return self.latest_prediction_error

    def update_belief(self, actual_observation: np.ndarray, neuromodulator_alpha: float = 1.0):
        """
        Updates belief/representation state using Hebbian/Gradient update (Belief Update).
        alpha is a dynamic modulation rate (e.g. Hebbian Learning Rate).
        """
        effective_lr = self.learning_rate * neuromodulator_alpha
        prediction_error_vec = actual_observation - self.predicted_state

        # Update state beliefs directly based on the prediction error (gradient update)
        self.belief_state += effective_lr * prediction_error_vec

        # Update transition matrix parameters (Hebbian learning on state transition)
        # S(t+1) ~ W * S(t). Outer product Hebbian correlation update.
        if np.linalg.norm(self.belief_state) > 1e-6:
            self.transition_matrix += 0.01 * np.outer(prediction_error_vec, self.belief_state)
            # Clip to prevent divergence
            self.transition_matrix = np.clip(self.transition_matrix, -2.0, 2.0)

    def actuate(self) -> np.ndarray:
        """Translates belief update to real action output vector."""
        self.latest_action = np.dot(self.actuation_matrix, self.belief_state)
        return self.latest_action

    def rollback_state(self, checkpoint: Dict[str, Any]):
        """Restores the belief state and model parameters to saved stable checkpoint values."""
        # Restore basic parameters
        if "belief_state_0" in checkpoint:
            self.belief_state = np.array([checkpoint.get(f"belief_state_{i}", 0.0) for i in range(self.dimensions)], dtype=np.float32)
        else:
            # Simple soft reset of belief state
            self.belief_state *= 0.1

        self.predicted_state = np.zeros(self.dimensions, dtype=np.float32)
        # Soft-reset transition matrix towards identity to clean up bad parameters
        self.transition_matrix = np.eye(self.dimensions, dtype=np.float32)
