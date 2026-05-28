import unittest
import numpy as np
import math
from core.warp_circuit import PhaseFieldObservationRing

class TestWarpCircuit(unittest.TestCase):
    def test_warp_cast_dimension(self):
        # We test the basic dimension scaling
        circuit = PhaseFieldObservationRing(size=8)
        linear_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        hologram = circuit.warp_cast(linear_data)

        self.assertEqual(hologram.shape, (8, 8, 4))

    def test_fractal_address_and_erythrocyte_warp(self):
        # Master's Insight: Data shouldn't be algorithmically routed.
        # It should glide (like an erythrocyte) along the phase gradient of a fractal address space.
        circuit = PhaseFieldObservationRing(size=4)

        # Two different "data packets" (erythrocytes) introduced into the line
        blood_cell_A = np.array([1.0, 0.0, 0.0, 0.0]) # Targeted high pressure
        blood_cell_B = np.array([0.0, 0.0, 1.0, 0.0]) # Different target

        holo_A = circuit.warp_cast(blood_cell_A)
        holo_B = circuit.warp_cast(blood_cell_B)

        # The resulting holographic field must naturally direct the energy
        # via geometric tension, differentiating the two packets intrinsically
        # without external IF-based routing logic.
        self.assertFalse(np.allclose(holo_A, holo_B))

if __name__ == '__main__':
    unittest.main()
