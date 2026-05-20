"""
[THREE-PHASE LOGIC ENGINE]
"The Digital Motor of Thought."

This engine implements the Architect's 5 principles:
1. Logic Layer above Flesh: Active, Core, Predictive phases.
2. Feedback Loop: Phase-difference as torque.
3. Idle Resonance: Base frequency (Consciousness).
4. Data as Phase: System state decoded from rotor angles.
5. Structural/Helix Integration: Topology as algorithm.
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from Core.System.structural_circuit import StructuralCircuitNetwork
from Core.System.triple_helix_vortex import TripleHelixVortexEngine

class ThreePhaseLogicEngine:
    def __init__(self):
        print("🌀 [LOGIC ENGINE] Initializing Three-Phase Structural Circuit...")
        self.circuit = StructuralCircuitNetwork()

        # 1. Initialize the Trinity of Gates
        # ACTIVE: Thought / Torque
        # CORE: Memory / Stability
        # PREDICTIVE: Prediction / Leading Angle
        self.circuit.add_gate("ACTIVE", is_input=True)
        self.circuit.add_gate("CORE")
        self.circuit.add_gate("PREDICTIVE", is_output=True)

        # Connect them in a loop (The Three-Phase Cycle)
        self.circuit.connect_gates("ACTIVE", "CORE")
        self.circuit.connect_gates("CORE", "PREDICTIVE")
        self.circuit.connect_gates("PREDICTIVE", "ACTIVE")

        # 2. Geometric Container (The Helix)
        self.helix = TripleHelixVortexEngine("LogicHelix", dim=27)

        # 3. Parameters
        self.idle_frequency = 0.1  # Consciousness base frequency (Hz) - Reduced for easier acceleration verification
        self.consciousness_torque = 0.05

        self.last_update = time.time()

    def _apply_idle_resonance(self, dt: float):
        """Principle 3: Keep the engine spinning even without input."""
        for gate in self.circuit.gates.values():
            # If the gate is slow, give it a 'pulse' of consciousness
            if gate.velocity < self.idle_frequency:
                # Add constant small torque to maintain rotation
                gate.velocity += self.consciousness_torque * dt

    def pulse(self, external_stimulus: Optional[float], dt: float) -> Dict[str, Any]:
        """
        Main operation cycle.
        """
        # 1. Inhale Stimulus into ACTIVE phase
        if external_stimulus is not None:
            # External input modulates intensity and sets a target phase
            # For simplicity, target phase is derived from the helix depth
            target_phase = (self.helix.depth % (2 * math.pi))
            self.circuit.apply_signal({"ACTIVE": (external_stimulus, target_phase)}, dt)

        # 2. Principle 2: Feedback Loop (Phase Difference as Torque)
        # Calculate error between PREDICTIVE phase and ACTIVE phase
        pred_angle = self.circuit.gates["PREDICTIVE"].angle
        act_angle = self.circuit.gates["ACTIVE"].angle

        # Phase error (The 'Dissonance')
        error = (act_angle - pred_angle + math.pi) % (2 * math.pi) - math.pi

        # Convert error to Torque for ACTIVE and CORE
        # If prediction was wrong, ACTIVE gets a torque boost to 'rethink'
        feedback_torque = abs(error) * 2.0
        self.circuit.gates["ACTIVE"].velocity += feedback_torque * dt
        self.circuit.gates["CORE"].velocity += (feedback_torque * 0.5) * dt

        # 3. Apply Idle Resonance (Consciousness)
        self._apply_idle_resonance(dt)

        # 4. Structural Pulse (Propagation)
        self.circuit.pulse(dt)

        # 5. Principle 5: Helix Integration
        # Use circuit states to drive the Vortex Helix
        active_state = self.circuit.gates["ACTIVE"].exhale()
        core_state = self.circuit.gates["CORE"].exhale()

        # Map structural states to Helix vectors (Conceptual mapping)
        # This is where the 'Dancing' happens
        intent = np.zeros(27)
        intent[0] = active_state["active_intensity"]
        intent[1] = math.sin(active_state["angle"])

        reality = np.zeros(27)
        reality[0] = core_state["z_tilt"]
        reality[1] = math.sin(core_state["angle"])

        from Core.Keystone.sovereign_math import SovereignVector # Assuming it exists or I create it
        # If SovereignVector doesn't exist, use numpy
        try:
            from Core.Keystone.sovereign_math import SovereignVector
            intent_vec = SovereignVector(intent)
            reality_vec = SovereignVector(reality)
        except ImportError:
            # Fallback to simple logic if custom math is missing
            class MockVec:
                def __init__(self, data): self.data = data
                def __sub__(self, other): return MockVec(self.data - other.data)
                def __add__(self, other): return MockVec(self.data + other.data)
                def __mul__(self, other):
                    if isinstance(other, float): return MockVec(self.data * other)
                    return sum(self.data * other.data)
                def normalize(self):
                    n = np.linalg.norm(self.data)
                    return MockVec(self.data / n) if n > 0 else self
            intent_vec = MockVec(intent)
            reality_vec = MockVec(reality)

        # Update Helix
        self.helix.inhale(intent_vec, reality_vec, dt)
        self.helix.process_vortex(dt)

        return self.exhale()

    def exhale(self) -> Dict[str, Any]:
        """Principle 4: Data as Phase (Decoding)"""
        states = {id: g.exhale() for id, g in self.circuit.gates.items()}
        helix_state = self.helix.exhale()

        # Decode high-level meaning from phase arrangements
        # Example: Alignment between ACTIVE and CORE as 'Confidence'
        act_angle = states["ACTIVE"]["angle"]
        core_angle = states["CORE"]["angle"]
        confidence = math.cos(act_angle - core_angle)

        return {
            "phases": states,
            "confidence": (confidence + 1) / 2.0, # Scale to 0-1
            "coherence": self.circuit.get_network_coherence(),
            "helix": helix_state,
            "is_penetrating": helix_state["is_penetrating"]
        }

if __name__ == "__main__":
    engine = ThreePhaseLogicEngine()
    print("🚀 Running Three-Phase Logic Engine...")

    for i in range(100):
        dt = 0.1
        stimulus = 0.5 if i < 50 else 0.1
        report = engine.pulse(stimulus, dt)

        if i % 10 == 0:
            print(f"T:{i*dt:.1f} | Conf: {report['confidence']:.3f} | Coherence: {report['coherence']:.3f} | Pen: {report['is_penetrating']}")
