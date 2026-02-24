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

        # [PHASE Ω-1] Emergent States (Synced from Manifold)
        self.enthalpy = 1.0  # Vitality (read from ch 6)
        self.entropy = 0.0   # Disorder (read from ch 7)
        self.joy = 0.5       # Joy (read from ch 4)
        self.curiosity = 0.5 # Curiosity (read from ch 5)
        self.mood = "NEUTRAL"
        self.last_tick = time.time()

    def sync_with_manifold(self, report: dict):
        """
        [PHASE Ω-1: THE OBSERVER]
        Synchronizes the thermal metrics with the emergent states from the VortexField.
        Thermodynamics no longer 'calculates' state; it 'perceives' it.
        """
        self.enthalpy = report.get('enthalpy', self.enthalpy)
        self.entropy = report.get('entropy', self.entropy)
        self.joy = report.get('joy', self.joy)
        self.curiosity = report.get('curiosity', self.curiosity)
        self.mood = report.get('mood', self.mood)

    def update_phase(self, current_phase: float):
        """Records the system phase to track rigidity."""
        self.phase_history.append(current_phase)

    def track_access(self, node_id: str):
        """Records memory access to track repetitive friction."""
        if node_id:
            self.node_access_history.append(node_id)

    def pulse_metabolism(self, dt: float = 0.1, activity_level: float = 0.5):
        """
        [DEPRECATED STATE MATH] 
        Now only serves as a heartbeat for temporal history updates.
        Manifold state is synced via sync_with_manifold().
        """
        pass

    def consume_energy(self, amount: float):
        """[PROXY] External energy consumption should now be handled via Torque Injection."""
        pass

    def add_entropy(self, amount: float):
        """[PROXY] External entropy increase should now be handled via Torque Injection."""
        pass

    def recharge(self, amount: float):
        """[PROXY] Recharging now happens through Joy-Torque or Sleep-Cycles."""
        pass

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
