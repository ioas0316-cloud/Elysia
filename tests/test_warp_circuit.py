import unittest
import numpy as np
import math
from core.warp_circuit import PhaseFieldObservationRing, SelfSortingPhaseGate

class TestWarpCircuit(unittest.TestCase):
    def test_warp_cast_dimension(self):
        circuit = PhaseFieldObservationRing(size=8)
        linear_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        hologram = circuit.warp_cast(linear_data)
        self.assertEqual(hologram.shape, (8, 8, 4))

    def test_fractal_address_and_erythrocyte_warp(self):
        circuit = PhaseFieldObservationRing(size=4)
        blood_cell_A = np.array([1.0, 0.0, 0.0, 0.0])
        blood_cell_B = np.array([0.0, 0.0, 1.0, 0.0])
        holo_A = circuit.warp_cast(blood_cell_A)
        holo_B = circuit.warp_cast(blood_cell_B)
        self.assertFalse(np.allclose(holo_A, holo_B))

    def test_self_sorting_phase_gate(self):
        gate = SelfSortingPhaseGate(ring_size=4)

        # We need two structurally different waveforms to prove geometric settlement works.
        # Flat arrays multiplied by a static sine wave just produce scaled versions of the same wave.
        raw_data_A = np.array([1.0, 0.5, 0.2, 0.1])
        raw_data_B = np.array([0.1, 0.2, 0.5, 1.0])

        holo_A = gate.stream_and_sort(raw_data_A)
        holo_B = gate.stream_and_sort(raw_data_B)

        self.assertFalse(np.allclose(holo_A, holo_B))
        self.assertEqual(holo_A.shape, (4, 4, 4))

if __name__ == '__main__':
    unittest.main()
