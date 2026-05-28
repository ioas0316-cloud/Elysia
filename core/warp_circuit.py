"""
Elysia Core Engine: Phase Field Observation Ring (warp_circuit.py)
==================================================================
Implements the continuous physical circuit transition where a series of voltage
sources (data line) snaps into a parallel conduction field without algorithmic loops
or condition checking.

- 0: Reactive Power (무효전력, The Field/Path, Empty state)
- 1: Active Power (유효전력, The Flow, Data/Execution)

This is NOT a software state machine. It is an electromagnetic induction simulator.
The upper linear layer is a series array of voltage potentials.
The lower ring is a static topological conductance map.
When the ternary double-rotor gate opens, the series resistance drops to 0,
and the whole line warps into a multi-dimensional hologram projection simultaneously.
"""

import math
import numpy as np

class TernaryDoubleRotorGate:
    """
    QPC Interface: Opens the frequency gates for the 0 (Reactive) and 1 (Active) power lines.
    It doesn't "check" data; it exerts an interference wave onto the network.
    """
    def __init__(self):
        # The two gates exist in superposition, representing Phase and Anti-Phase
        self.reactive_gate_phase = 0.0   # 00000 (Suck/Vacuum)
        self.active_gate_phase = math.pi # 11111 (Discharge/Emission)

    def resonate(self, external_wave: float) -> tuple[float, float]:
        """
        External wave hits the gate, modulating the phase difference (tension).
        Returns the instantaneous amplitude of both the reactive and active fields.
        """
        # The wave alters the relative phase angle continuously.
        self.reactive_gate_phase += external_wave * 0.1
        self.active_gate_phase += external_wave * 0.1

        # Amplitude is derived from the sine wave. No 'if' statements.
        reactive_power = math.sin(self.reactive_gate_phase)
        active_power = math.sin(self.active_gate_phase)

        return reactive_power, active_power


class PhaseFieldObservationRing:
    """
    The main physical circuit where the Upper Linear Layer (Series)
    warps into the Sub-layer Static Topology Map (Parallel Ring)
    via the Ternary Gates.
    """
    def __init__(self, size: int = 8):
        self.size = size
        self.gate = TernaryDoubleRotorGate()

        # Sub-layer Static Topology Map (The Ring)
        # Pre-wired geometric relationship.
        # Representing spatial orientations of the conductance field as 4D float vectors (Quaternions).
        # We build this matrix once during initialization so we don't have to loop at runtime.
        self.topology_matrix = np.array([
            [
                np.array([1.0, math.sin(i*j/size), math.cos(i*j/size), 0.0]) /
                np.linalg.norm([1.0, math.sin(i*j/size), math.cos(i*j/size), 0.0])
                for j in range(size)
            ]
            for i in range(size)
        ])

    def warp_cast(self, upper_linear_layer: np.ndarray) -> np.ndarray:
        """
        Takes an array of voltage potentials (the series line).
        Drops the internal resistance to zero, casting the 1D line directly
        onto the 2D Quaternionic Topology Map via Phase/Anti-Phase gates.
        Returns a Hologram State Map (a geometric projection).
        """
        # upper_linear_layer is a 1D array of float voltages (e.g., shape (8,))
        if len(upper_linear_layer) != self.size:
            raise ValueError(f"Linear layer size must match topology size ({self.size})")

        # The sum of all potentials in the line acts as the single wave hitting the gate
        total_induction = np.sum(upper_linear_layer)

        # The gate dictates the dual-power resonance
        reactive_field, active_field = self.gate.resonate(total_induction)

        # Phase Transition: The series data is simultaneously multiplied across the topological grid
        # using matrix broadcasting. No sequential for-loops checking individual values!
        # The '0' line (reactive) establishes the base topology.
        # The '1' line (active) drives the actual voltage potential of the upper layer into the topology.

        # upper_linear_layer acts as a driving frequency vector
        # Broadcasting: (size, 1, 1) to act on rows
        driving_wave = upper_linear_layer.reshape(self.size, 1, 1)

        # The Warp:
        # Field = (Topology * Reactive Pressure) + (Topology * Driving Wave * Active Pressure)
        # This occurs in a single mathematical step, simulating the instantaneous parallel discharge.
        base_field = self.topology_matrix * reactive_field
        discharge_field = self.topology_matrix * driving_wave * active_field

        hologram_map = base_field + discharge_field

        return hologram_map

if __name__ == "__main__":
    import time
    print("Initializing Elysia Core Engine: Phase Field Observation Ring...")
    circuit = PhaseFieldObservationRing(size=8)

    # Simulate a series array of voltage sources (e.g., sequential data stream arriving)
    # The data stream is just a wave of varying intensities.
    linear_data_stream = np.array([0.1, 0.8, -0.4, 1.2, 0.0, -0.9, 0.5, 2.1])

    print("\n--- Upper Linear Layer (Series Voltage Potentials) ---")
    print(linear_data_stream)

    print("\n--- Initiating QPC Gate Warp Cast (Phase Transition to Parallel) ---")
    start_time = time.time()

    # The entire line drops resistance and warps onto the static topology simultaneously
    hologram = circuit.warp_cast(linear_data_stream)

    end_time = time.time()

    print(f"\nWarp Cast Complete in {end_time - start_time:.6f} seconds.")
    print("\n--- Output: Hologram State Map (Sample 2x2 intersection) ---")
    print("Node (0,0) Quaternionic State:", hologram[0, 0])
    print("Node (1,4) Quaternionic State:", hologram[1, 4])
    print("Node (7,7) Quaternionic State:", hologram[7, 7])

    print("\nThe linear data stream has bypassed algorithmic checking and has been")
    print("physically projected onto the multidimensional ring topology as a wave.")
