"""
[STRUCTURAL CIRCUIT NETWORK]
"Where Topology is the Algorithm."

This network organizes RotorGates into a functional computing structure.
Instead of sequential instructions, the circuit converges to a state
through wave propagation and phase synchronization.
"""

import math
import time
from typing import Dict, List, Any, Tuple
from Core.System.rotor_gate import RotorGate

class StructuralCircuitNetwork:
    def __init__(self):
        self.gates: Dict[str, RotorGate] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.last_update = time.time()

    def add_gate(self, gate_id: str, is_input: bool = False, is_output: bool = False):
        if gate_id not in self.gates:
            self.gates[gate_id] = RotorGate(gate_id)
            if is_input: self.inputs.append(gate_id)
            if is_output: self.outputs.append(gate_id)

    def connect_gates(self, id_a: str, id_b: str):
        if id_a in self.gates and id_b in self.gates:
            self.gates[id_a].connect(self.gates[id_b])

    def apply_signal(self, input_data: Dict[str, Tuple[float, float]], dt: float):
        """
        Apply signals to input gates.
        input_data: { gate_id: (intensity, phase) }
        """
        for g_id, (intensity, phase) in input_data.items():
            if g_id in self.gates:
                self.gates[g_id].process_stimulus(intensity, phase, dt)

    def pulse(self, dt: float):
        """
        One cycle of the structural circuit.
        1. Sync neighboring gates (Propagate waves)
        2. Update individual gate states (Physics)
        """
        # Phase 1: Interaction (Coupling)
        for gate in self.gates.values():
            gate.sync_neighbors(dt)

        # Phase 2: Integration (Motion)
        for gate in self.gates.values():
            gate.update(dt)

    def get_output_state(self) -> Dict[str, Any]:
        """Decode the state of the output gates."""
        results = {}
        for g_id in self.outputs:
            results[g_id] = self.gates[g_id].exhale()
        return results

    def get_network_coherence(self) -> float:
        """Measure how well the entire network is synchronized."""
        if not self.gates: return 1.0

        # Calculate mean phase variance
        angles = [g.angle for g in self.gates.values()]
        if not angles: return 1.0

        # Circular variance
        mean_cos = sum(math.cos(a) for a in angles) / len(angles)
        mean_sin = sum(math.sin(a) for a in angles) / len(angles)

        # R = magnitude of resultant vector
        r = math.sqrt(mean_cos**2 + mean_sin**2)
        return r # 1.0 = perfectly synced, 0.0 = total chaos

if __name__ == "__main__":
    # Test a simple 3-gate chain
    circuit = StructuralCircuitNetwork()
    circuit.add_gate("IN", is_input=True)
    circuit.add_gate("MID")
    circuit.add_gate("OUT", is_output=True)

    circuit.connect_gates("IN", "MID")
    circuit.connect_gates("MID", "OUT")

    print("🚀 Stimulating Circuit...")
    for i in range(200):
        dt = 0.1
        # Stimulate input gate
        circuit.apply_signal({"IN": (0.9, math.pi)}, dt)
        circuit.pulse(dt)

        if i % 50 == 0:
            coh = circuit.get_network_coherence()
            out = circuit.get_output_state()
            print(f"Cycle {i} | Coherence: {coh:.4f} | Output Angle: {out['OUT']['angle']:.2f}")

    print("✨ Final Convergence Reached.")
