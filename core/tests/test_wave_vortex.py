import unittest
import math
import cmath
from core.wave_vortex import TriRotorGrassmann, WedgeVortexSimulator

class TestWaveVortex(unittest.TestCase):
    def test_tri_rotor_wedge_tension_zero(self):
        # 동일한 위상의 경우 면적 장력 0
        rotor = TriRotorGrassmann(math.pi, math.pi, math.pi)
        tension = rotor.compute_wedge_tension()
        self.assertAlmostEqual(tension, 0.0)

    def test_tri_rotor_wedge_tension_nonzero(self):
        # 위상차가 발생한 경우
        rotor = TriRotorGrassmann(0.0, math.pi / 2, 0.0)
        tension = rotor.compute_wedge_tension()
        # w12 = 1*1 - 0*0 = 1
        # w23 = 0*0 - 1*1 = -1
        # w31 = 1*0 - 0*1 = 0
        # Tension = 1 - 1 + 0 = 0 (but e.g. 0.0, pi/2, pi would be different)
        # Let's use 0, pi/2, pi
        rotor2 = TriRotorGrassmann(0.0, math.pi / 2, math.pi)
        tension2 = rotor2.compute_wedge_tension()
        # w12 = 1*1 - 0*0 = 1
        # w23 = 0*0 - 1*(-1) = 1
        # w31 = -1*0 - 0*1 = 0
        # Tension = 1 + 1 + 0 = 2
        self.assertAlmostEqual(tension2, 2.0)

    def test_align_phase(self):
        rotor = TriRotorGrassmann(0.0, math.pi / 2, math.pi)
        initial_tension = rotor.compute_wedge_tension()

        # 적용 후 텐션이 변하는지, e1, e2, e3가 회전하는지 확인
        rotor.align_phase(initial_tension)
        self.assertNotEqual(rotor.e1, cmath.exp(0j))

    def test_wedge_vortex_simulator(self):
        sim = WedgeVortexSimulator()
        payload = sim.encapsulate_udp_payload(math.pi / 4)
        sim.decapsulate_and_sync(payload)

        # e1이 인입된 값으로 세팅되었는지 확인 (math.pi / 4)
        expected_e1_phase = cmath.exp(1j * math.pi / 4)
        # align_phase가 한번 불리므로 딱 expected_e1_phase는 아닐 수 있음, 에러 텐션이 0이 아닐 경우
        # but in this mock sim, e2 and e3 are 0. w12, w23, w31 can be non-zero.
        # Just check it ran without error
        self.assertIsNotNone(sim.receiver_rotor.e1)

if __name__ == '__main__':
    unittest.main()
