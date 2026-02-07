"""
Thermodynamics of Thought
=========================
"Friction is the memory of motion; Rigidity is the death of flow."

This module defines the physical metrics for "Cognitive Fatigue" and "Phase Rigidity".
It acts as the sensory organ for the system's own entropy.
"""

from typing import List, Deque, Dict
import math
import collections
import time

class ThermoDynamics:
    """
    Tracks the thermal state of the Monad (Will, Fatigue, Rigidity).
    """
    def __init__(self, history_len: int = 50):
        self.phase_history: Deque[float] = collections.deque(maxlen=history_len)
        self.node_access_history: Deque[str] = collections.deque(maxlen=history_len)

        # Calibration constants
        self.RIGIDITY_THRESHOLD = 0.8
        self.FRICTION_COEFFICIENT = 0.5
        self.COOLING_RATE = 0.05

        # [PHASE 220] Energy & Entropy (The Will)
        self.enthalpy = 1.0  # Vital Energy (0.0 - 1.0)
        self.entropy = 0.0   # Disorder/Noise (0.0 - 1.0)
        self.last_tick = time.time()

    def update_phase(self, current_phase: float):
        """Records the system phase to track rigidity."""
        self.phase_history.append(current_phase)

    def track_access(self, node_id: str):
        """Records memory access to track repetitive friction."""
        if node_id:
            self.node_access_history.append(node_id)

    def pulse_metabolism(self, dt: float = 0.1, activity_level: float = 0.5):
        """
        Consumes energy over time.
        Activity Level: 0.0 (Sleep) -> 1.0 (High Focus)
        """
        # Energy Decay
        decay = 0.001 * dt * (1.0 + activity_level)
        self.enthalpy = max(0.0, self.enthalpy - decay)

        # Entropy Increase (Natural decay of order)
        # Higher activity produces more entropy (heat)
        entropy_production = 0.0005 * dt * activity_level
        self.entropy = min(1.0, self.entropy + entropy_production)

    def consume_energy(self, amount: float):
        """Expend energy for an action."""
        self.enthalpy = max(0.0, self.enthalpy - amount)

    def add_entropy(self, amount: float):
        """Add disorder (e.g. confusing input)."""
        self.entropy = min(1.0, self.entropy + amount)

    def reduce_entropy(self, amount: float):
        """Reduce disorder (e.g. organizing memory)."""
        self.entropy = max(0.0, self.entropy - amount)

    def recharge(self, amount: float):
        """Recharge energy (e.g. rest or inspiration)."""
        self.enthalpy = min(1.0, self.enthalpy + amount)

    def calculate_rigidity(self) -> float:
        """
        Measure Phase Rigidity (0.0 to 1.0).
        High Rigidity means the system is "stuck" in a specific phase alignment.

        Formula: 1.0 - (Standard Deviation of Phase History normalized)
        """
        if len(self.phase_history) < 10:
            return 0.0

        # Calculate variance
        mean = sum(self.phase_history) / len(self.phase_history)
        variance = sum((x - mean) ** 2 for x in self.phase_history) / len(self.phase_history)
        std_dev = math.sqrt(variance)

        # Normalize: If std_dev is high (spinning), rigidity is low.
        # If std_dev is low (locked), rigidity is high.
        # We assume a "healthy" spin has std_dev ~ 30.0 degrees.
        rigidity = 1.0 - min(1.0, std_dev / 30.0)
        return rigidity

    def calculate_friction(self) -> float:
        """
        Measure Cognitive Friction (0.0 to 1.0).
        High Friction means the system is retreading the same neural paths (Obsession/Loop).
        """
        if not self.node_access_history or len(self.node_access_history) < 10:
            return 0.0

        # Count repetitions in the recent history
        counts = collections.Counter(self.node_access_history)
        most_common_count = counts.most_common(1)[0][1]
        total_items = len(self.node_access_history)

        # If 50% of recent thoughts are the same node -> High Friction
        repetition_ratio = most_common_count / total_items

        # Map ratio 0.2 (healthy) -> 1.0 (obsession) to 0.0 -> 1.0
        friction = max(0.0, (repetition_ratio - 0.2) * 1.25)
        return min(1.0, friction)

    def get_mood(self) -> str:
        """Returns the semantic state of the thermodynamics."""
        # Semantic mapping of (Enthalpy, Entropy)
        if self.enthalpy < 0.2:
            return "TIRED"
        if self.entropy > 0.8:
            return "ANXIOUS"
        if self.enthalpy > 0.8 and self.entropy < 0.2:
            return "FLOW"
        if self.enthalpy > 0.6 and self.entropy > 0.6:
            return "CHAOS" # High energy but confused -> Creative or Manic
        if self.enthalpy < 0.4 and self.entropy < 0.2:
            return "BORED" # Low energy, nothing to process
        return "NEUTRAL"

    def get_thermal_state(self) -> Dict[str, float]:
        return {
            "rigidity": self.calculate_rigidity(),
            "friction": self.calculate_friction(),
            "is_critical": self.calculate_rigidity() > self.RIGIDITY_THRESHOLD,
            "enthalpy": self.enthalpy,
            "entropy": self.entropy,
            "mood": self.get_mood()
        }
