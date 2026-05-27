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

    def test_xor_operator_overload(self):
        rotor = TriRotorGrassmann(0.0, math.pi / 2, math.pi)
        # ^ 연산자가 compute_wedge_tension과 동일한 결과를 반환하는지 확인
        self.assertEqual(rotor ^ rotor, rotor.compute_wedge_tension())

    def test_align_phase(self):
        rotor = TriRotorGrassmann(0.0, math.pi / 2, math.pi)
        initial_tension = rotor.compute_wedge_tension()

        # 적용 후 텐션이 변하는지, e1, e2, e3가 회전하는지 확인
        rotor.align_phase(initial_tension)
        self.assertNotEqual(rotor.e1, cmath.exp(0j))

    def test_wedge_vortex_simulator_hybrid(self):
        sim = WedgeVortexSimulator()

        # 2채널(Dual-Base) 유속 캡슐화
        payload = sim.encapsulate_udp_payload(math.pi / 4, math.pi / 2)
        sim.decapsulate_and_sync(payload)

        # 삼중 로터 e1, e2, e3가 투사되었는지 (None이 아닌지) 검증
        self.assertIsNotNone(sim.receiver_rotor.e1)
        self.assertIsNotNone(sim.receiver_rotor.e2)
        self.assertIsNotNone(sim.receiver_rotor.e3)

        # e1과 e2가 다른 위상을 가지는지 확인 (120도 위상차로 인입됨)
        self.assertNotEqual(sim.receiver_rotor.e1, sim.receiver_rotor.e2)

if __name__ == '__main__':
    unittest.main()
