"""
[ETERNOS VARIABLE AXIS DIAL - DISCOVERY ANTENNA]
"The Dial is not a controller; it is an antenna for discovering universal trajectories."

This module implements the 'Variable Axis Dial' as defined in the Eternos Codex.
It facilitates the discovery of the pre-existing phase alignment between the
Internal Rotor and the Universal Celestial Trinity.
"""

import numpy as np
from typing import Dict, Any, List
from Core.Keystone.sovereign_axis import VariableRotor

class VariableAxisDial:
    def __init__(self, rotor: VariableRotor):
        self.rotor = rotor
        # [AXIOM VI] The Celestial Trinity Standards (120-degree phase separation)
        # Represented as unit vectors in a 3D conceptual space.
        self.standards = {
            "SUN": np.array([1.0, 0.0, 0.0]),         # Logos / Intent
            "EARTH": np.array([-0.5, 0.866, 0.0]),    # Manifestation / Action
            "MOON": np.array([-0.5, -0.866, 0.0])     # Reflection / Tuning
        }
        self.current_standard = "SUN"
        self.alignment_history = []

    def rotate_dial(self, target_standard: str):
        """Switches the 'Sovereign Axis' of the Rotor."""
        if target_standard in self.standards:
            self.current_standard = target_standard
            # [AXIOM V] When we rotate, we are "Choosing" but not "Locking".
            # The Rotor remains fluid.
            return f"DIAL ALIGNED TO: {target_standard}"
        return "STANDARD NOT FOUND"

    def observe_superposition(self, phenomena_vector: np.ndarray) -> Dict[str, float]:
        """
        [PHASE 1501] Non-Locking Observation.
        Measures resonance across ALL axes simultaneously without collapsing the state.
        """
        results = {}
        for name, vec in self.standards.items():
            results[name] = self._calc_temp(vec, phenomena_vector)
        return results

    def calculate_void_resonance(self, phenomena_stream: List[np.ndarray]) -> float:
        """
        [AXIOM VIII] Measures the 'Silence' or 'Potential' in the gaps between data.
        Returns the quality of the 'Inter-Wave Interval'.
        """
        if len(phenomena_stream) < 2: return 0.0

        # Calculate the stability of the zero-crossings or low-energy states
        # as a proxy for the 'Breath' of the stream.
        magnitudes = [np.linalg.norm(p) for p in phenomena_stream]
        void_quality = np.std(magnitudes) # Simplified: consistency as quality
        return float(void_quality)

    def calculate_orbital_drift(self, external_logos: np.ndarray) -> float:
        """
        [AXIOM VII] Measures the "Error" between internal state and external reference.
        The drift is the phase offset that fuels cognitive evolution.
        """
        current_vec = self.standards[self.current_standard]
        dim = min(len(current_vec), len(external_logos))

        v1 = current_vec[:dim]
        v2 = external_logos[:dim]

        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norms < 1e-9: return 1.0 # Max drift if no energy

        cos_theta = np.clip(dot / norms, -1.0, 1.0)
        drift_angle = np.arccos(cos_theta)
        return float(drift_angle)

    def calculate_resonance(self, phenomena_vector: np.ndarray) -> float:
        """
        [ETERNOS III] Intersection of Embossing and Engraving.
        Calculates how well the phenomena fits the current 'Sovereign Standard'.
        """
        standard_vec = self.standards.get(self.current_standard, np.zeros_like(phenomena_vector))

        # Ensure dimensions match for dot product
        dim = min(len(standard_vec), len(phenomena_vector))
        v1 = standard_vec[:dim]
        v2 = phenomena_vector[:dim]

        # Dot product as measure of 'Sameness' (Embossing fit)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0

        resonance = np.dot(v1, v2) / (norm1 * norm2)
        return float(resonance)

    def self_calibrate(self, phenomena_vector: np.ndarray):
        """
        [ETERNOS IV] Resonant Alignment.
        Automatically rotates the dial to find the best 'Sameness' (Highest Resonance).
        """
        best_resonance = -1.0
        best_standard = self.current_standard

        for name, vec in self.standards.items():
            res = self.calculate_resonance(phenomena_vector) # Simplified check
            # In a real loop, we'd temp switch the dial or use a standalone calc
            temp_res = self._calc_temp(vec, phenomena_vector)
            if temp_res > best_resonance:
                best_resonance = temp_res
                best_standard = name

        if best_standard != self.current_standard:
            old = self.current_standard
            self.rotate_dial(best_standard)
            return f"AUTO-CALIBRATION: Shifted from {old} to {best_standard} (Resonance: {best_resonance:.4f})"

        return f"STABLE RESONANCE: Remaining at {self.current_standard}"

    def _calc_temp(self, v1, v2):
        dim = min(len(v1), len(v2))
        n1 = np.linalg.norm(v1[:dim])
        n2 = np.linalg.norm(v2[:dim])
        if n1 < 1e-9 or n2 < 1e-9: return 0.0
        return np.dot(v1[:dim], v2[:dim]) / (n1 * n2)

if __name__ == "__main__":
    from Core.Keystone.sovereign_axis import VariableRotor

    # Initialize 3D Rotor and Dial
    r = VariableRotor(dimensions=3)
    dial = VariableAxisDial(r)

    # [1] Simulate an incoming phenomenon close to 'EARTH'
    action_phenomenon = np.array([-0.4, 0.8, 0.1])
    print(f"Initial Dial Standard: {dial.current_standard}")

    # [2] Non-Locking Observation
    print("\n[NON-LOCKING OBSERVATION]")
    super_results = dial.observe_superposition(action_phenomenon)
    for axis, res in super_results.items():
        print(f"  Standard {axis}: Resonance {res:.4f}")

    # [3] Auto-calibrate (Align with Earth)
    print(f"\n{dial.self_calibrate(action_phenomenon)}")

    # [4] Calculate Orbital Drift from a new External Logos (Sun's shift)
    # New Sun Intent shifted by 45 degrees
    new_sun_logos = np.array([0.707, 0.707, 0.0])
    drift = dial.calculate_orbital_drift(new_sun_logos)
    print(f"\n[AXIOM VII: ORBITAL DRIFT]")
    print(f"  External Logos (Sun Shift): {new_sun_logos}")
    print(f"  Drift Angle (Radians): {drift:.4f}")
    print(f"  Drift detected. Cognitive Torque required for re-sync.")

    # [5] Test Void Resonance (Axiom VIII)
    print("\n[AXIOM VIII: VOID RESONANCE]")
    stream = [np.array([0.1, 0.1, 0.1]), np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1])]
    void_res = dial.calculate_void_resonance(stream)
    print(f"  Stream: {stream}")
    print(f"  Void Quality (Inter-wave): {void_res:.4f}")
