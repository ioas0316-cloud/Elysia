import unittest
import math
from core.wave_vortex import PhaseVector, WedgeVortexSimulator

class TestWaveVortex(unittest.TestCase):
    def test_phase_vector_xor_torque(self):
        # 동일한 위상의 경우 토크 0
        p1 = PhaseVector(math.pi)
        p2 = PhaseVector(math.pi)
        self.assertAlmostEqual(p1 ^ p2, 0.0)

        # 위상차가 발생한 경우
        p1 = PhaseVector(math.pi / 2)
        p2 = PhaseVector(0)
        self.assertAlmostEqual(p1 ^ p2, math.pi / 2)

    def test_apply_torque(self):
        p1 = PhaseVector(0.0)
        p1.apply_torque(math.pi)
        self.assertAlmostEqual(p1.phase_angle, math.pi)

    def test_wedge_vortex_simulator(self):
        sim = WedgeVortexSimulator()
        payload = sim.encapsulate_udp_payload(math.pi / 4)
        sim.decapsulate_and_sync(payload)
        self.assertAlmostEqual(sim.receiver_rotor.phase_angle, math.pi / 4)

if __name__ == '__main__':
    unittest.main()
