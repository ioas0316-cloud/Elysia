"""
[THREE-PHASE LOGIC ENGINE - DELTA-Y EDITION]
"The Digital Motor of Thought using Tensor Networks."

Implements:
1. Bundle (Delta): Clustering 3 gates into an internal loop.
2. Transform (Y): Extracting neutral/common output for higher-level cognition.
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from Core.System.structural_circuit import StructuralCircuitNetwork
from Core.System.triple_helix_vortex import TripleHelixVortexEngine
from Core.Keystone.sovereign_math import SovereignMath

class ThreePhaseLogicEngine:
    def __init__(self):
        print("🌀 [LOGIC ENGINE] Initializing Delta-Y Tensor Circuit...")
        self.circuit = StructuralCircuitNetwork()

        # 1. Initialize the Trinity Cluster (Delta)
        # We group 3 gates into a 'Cell' (Delta 결선)
        self.circuit.add_gate("PHASE_A", is_input=True)
        self.circuit.add_gate("PHASE_B")
        self.circuit.add_gate("PHASE_C", is_output=True)

        # Delta Connection: Internal Loop (Mutual Interference)
        self.circuit.connect_gates("PHASE_A", "PHASE_B")
        self.circuit.connect_gates("PHASE_B", "PHASE_C")
        self.circuit.connect_gates("PHASE_C", "PHASE_A")

        self.helix = TripleHelixVortexEngine("LogicHelix", dim=27)
        self.math = SovereignMath()

        self.idle_frequency = 0.1
        self.consciousness_torque = 0.05
        self.last_update = time.time()
        self.last_y_output = 0.0
        self.last_states = [
            {"angle": 0.0, "active_intensity": 0.0, "velocity": 0.0},
            {"angle": 0.0, "active_intensity": 0.0, "velocity": 0.0},
            {"angle": 0.0, "active_intensity": 0.0, "velocity": 0.0}
        ]

    def pulse(self, external_stimulus: Optional[float], dt: float) -> Dict[str, Any]:
        """
        Main operation cycle using Delta-Y logic.
        """
        # 1. Delta Inhale (Internal Circulation)
        if external_stimulus is not None and external_stimulus != 0.0:
            # Distribute stimulus across the Delta loop
            sig = {
                "PHASE_A": (external_stimulus, 0.0),
                "PHASE_B": (external_stimulus * 0.5, 2 * math.pi / 3),
                "PHASE_C": (external_stimulus * 0.2, 4 * math.pi / 3)
            }
            self.circuit.apply_signal(sig, dt)
        else:
            # Idle resonance: apply dynamic consciousness_torque at dynamic idle_frequency
            if not hasattr(self, "time_elapsed"):
                self.time_elapsed = 0.0
            self.time_elapsed += dt
            
            # Extract state values from the Helix
            helix_state = self.helix.exhale()
            coh = helix_state["coherence"]
            ang_vel = helix_state["focus_velocity"]
            
            # Dynamic state functions based on Helix feedback
            dynamic_torque = self.consciousness_torque * (2.0 - coh)
            dynamic_frequency = self.idle_frequency * (0.5 + 0.5 * ang_vel)
            
            base_phase = 2 * math.pi * dynamic_frequency * self.time_elapsed
            sig = {
                "PHASE_A": (dynamic_torque, base_phase),
                "PHASE_B": (dynamic_torque * 0.5, base_phase + 2 * math.pi / 3),
                "PHASE_C": (dynamic_torque * 0.2, base_phase + 4 * math.pi / 3)
            }
            self.circuit.apply_signal(sig, dt)

        # 2. Physics Pulse
        self.circuit.pulse(dt)

        # 3. Y-Transformation (Neutral Extraction)
        # Extract states of the 3 phases
        states = [self.circuit.gates[k].exhale() for k in ["PHASE_A", "PHASE_B", "PHASE_C"]]
        intensities = np.array([s["active_intensity"] for s in states])

        # Y-Output: The 'Neutral' point of the Delta loop
        y_output = self.math.delta_to_y(intensities)
        
        # Cache current state values
        self.last_y_output = y_output
        self.last_states = states

        # 4. Helix Integration (Tensor Network)
        intent = np.zeros(27)
        intent[0] = float(y_output)
        intent[1] = math.sin(states[0]["angle"])

        reality = np.zeros(27)
        reality[0] = states[1]["active_intensity"]
        reality[1] = math.cos(states[1]["angle"])

        # Update Helix (Tensor Network Integration)
        self.helix.inhale(intent, reality, dt)
        self.helix.process_vortex(dt)

        return self.exhale(y_output, states)

    def exhale(self, y_output: Optional[float] = None, states: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if y_output is None:
            y_output = self.last_y_output
        if states is None:
            states = self.last_states

        # Confidence is the alignment (low variance) of the Delta phases
        # If all 3 phases agree, the Y-output is stable.
        phase_angles = np.array([s["angle"] for s in states])
        coherence = np.abs(np.mean(np.exp(1j * phase_angles)))

        phases = {f"PHASE_{i}": s for i, s in zip(["A", "B", "C"], states)}
        phases["ACTIVE"] = phases["PHASE_A"]
        phases["PASSIVE"] = phases["PHASE_B"]
        phases["RESONATOR"] = phases["PHASE_C"]

        return {
            "y_neutral": float(y_output),
            "coherence": float(coherence),
            "confidence": float(coherence),
            "phases": phases,
            "helix": self.helix.exhale(),
            "is_penetrating": self.helix.exhale()["is_penetrating"]
        }

if __name__ == "__main__":
    engine = ThreePhaseLogicEngine()
    for i in range(10):
        report = engine.pulse(0.5, 0.1)
        print(f"Y-Neutral: {report['y_neutral']:.4f} | Confidence: {report['confidence']:.4f}")
