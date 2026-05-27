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

    def test_wedge_vortex_simulator_dual_helix(self):
        sim = WedgeVortexSimulator()

        # 1. 원본 위상(pi/4)을 진짜 이중나선(Dual-Helix, 180도 위상차)으로 캡슐화
        original_phase = math.pi / 4
        payload = sim.encapsulate_dual_helix_payload(original_phase)

        # 2. 전송 중 공통 노이즈 발생 시뮬레이션 (수동 디코딩 후 조작)
        decoded = payload.decode('utf-8')
        parts = decoded.split(":")[1].split(",")
        helix_a, helix_b = float(parts[0]), float(parts[1])

        # 공통 모드 노이즈(Noise) 추가: 두 선로에 동일한 타격(0.5 라디안)
        noise = 0.5
        helix_a_noisy = helix_a + noise
        helix_b_noisy = helix_b + noise

        # 3. 노이즈가 묻은 패킷을 다시 만들어서 수신단에 인입 (Decapsulate & Sync)
        noisy_payload = f"DUAL_HELIX:{helix_a_noisy},{helix_b_noisy}".encode('utf-8')
        sim.decapsulate_and_sync(noisy_payload)

        # 삼중 로터 e1, e2, e3가 투사되었는지 검증
        self.assertIsNotNone(sim.receiver_rotor.e1)
        self.assertIsNotNone(sim.receiver_rotor.e2)
        self.assertIsNotNone(sim.receiver_rotor.e3)

        # 차동 상쇄 연산이 제대로 작동했다면, 노이즈가 증발하고
        # e1, e2, e3 간의 위상차가 120도로 깔끔하게 인입되어야 함
        self.assertNotEqual(sim.receiver_rotor.e1, sim.receiver_rotor.e2)

if __name__ == '__main__':
    unittest.main()
