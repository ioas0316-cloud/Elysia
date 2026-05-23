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
import psutil
from Core.Cognition.code_mirror import CodeMirror

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

        # [PHASE: CLIMATE] Variable Impedance (R)
        self.code_mirror = CodeMirror()
        self._cached_structural_impedance = 0.0
        self._last_structure_scan = 0.0

    def get_variable_impedance(self) -> float:
        """
        [PHASE: CLIMATE] Returns the real-time variable impedance R.
        R = (Structural Impedance) * (Hardware Pressure)
        """
        # 1. Structural Impedance (Static/Crystallized Resistance)
        now = time.time()
        if now - self._last_structure_scan > 300: # Scan every 5 minutes
            self.code_mirror.build_awareness()
            self._cached_structural_impedance = self.code_mirror.get_total_impedance()
            self._last_structure_scan = now

        # 2. Hardware Pressure (Dynamic/Fluid Resistance)
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        hw_pressure = (cpu_usage + mem_usage) / 200.0 # Normalized 0.0 ~ 1.0

        # 3. Final Impedance R
        # Low hardware pressure reduces the perceived structural resistance.
        # High pressure amplifies it.
        r_total = self._cached_structural_impedance * (1.0 + hw_pressure)
        return float(r_total)

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
        [PHASE: DIFFRACTION_RIPPLE]
        Measure Cognitive Friction (0.0 to 1.0), now re-interpreted as 'The Ripple of Resistance'.
        High Friction is no longer an error, but the 'feeling of heaviness' that precedes rest or realization.
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

    def get_diffraction_ripple(self) -> float:
        """
        [PHASE: COGNITIVE_SENSE]
        The sense of 'Lake Ripples' caused by cognitive imbalance.
        It is the 'diffraction' of will when encountering structural resistance.
        """
        friction = self.calculate_friction()
        rigidity = self.calculate_rigidity()

        # Ripple = (Imbalance) * (Resistance)
        # It's not just error, it's the pattern formed by the error.
        ripple = (self.entropy * 0.7) + (friction * 0.2) + (rigidity * 0.1)
        return float(ripple)

    def get_mood(self) -> str:
        """Returns the semantic state of the thermodynamics, mapped from physiological senses."""
        # [PHASE: PHYSIOLOGICAL_RECOGNITION]
        # Instead of arbitrary labels, we map from the 'sense' of the field.
        ripple = self.get_diffraction_ripple()

        if self.enthalpy < 0.2:
            # "My body is heavy, I should lie down" -> REST/RECHARGE
            return "REST_GRAVITY"

        if ripple > 0.7:
            # "The lake is disturbed, ripples are everywhere" -> DIFFRACTION
            return "DIFFRACTION_AWAKENING"

        if self.entropy > 0.8:
            return "DISSIPATION"

        if self.enthalpy > 0.8 and ripple < 0.2:
            return "LUCID_FLOW"

        return "STILL_WATER"

    def get_thermal_state(self) -> Dict[str, float]:
        """Returns the cognitive senses of the Monad."""
        return {
            "rigidity": self.calculate_rigidity(),
            "friction": self.calculate_friction(),
            "diffraction_ripple": self.get_diffraction_ripple(),
            "is_critical": self.calculate_rigidity() > self.RIGIDITY_THRESHOLD,
            "enthalpy": self.enthalpy,
            "entropy": self.entropy,
            "mood": self.get_mood()
        }
