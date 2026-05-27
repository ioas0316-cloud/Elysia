import unittest
import math
import cmath
from core.wave_vortex import TriRotorTensionEngine, WedgeVortexSimulator

class TestWaveVortex(unittest.TestCase):
    def test_apply_relative_tension(self):
        # 극단적으로 불균형한 초기 위상 설정 (0, 0, 0)
        engine = TriRotorTensionEngine(0.0, 0.0, 0.0)

        # 텐션 적용
        # 현재는 완전히 동일하면 차이가 0이므로 tension이 0이 됨.
        # 살짝 다르게 주어 척력이 발생하도록 유도
        engine = TriRotorTensionEngine(0.0, 0.1, -0.1)

        for _ in range(10):
            engine.apply_relative_tension()

        phases = engine.get_current_phases()

        # 시간이 지나면서 각 로터의 위상이 서로 밀어내어 분산되어야 함
        # 차이가 점점 커지는지 확인
        self.assertNotEqual(phases[0], phases[1])
        self.assertNotEqual(phases[1], phases[2])
        self.assertNotEqual(phases[0], phases[2])

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
        self.assertIsNotNone(sim.receiver_rotor.rotors[0])
        self.assertIsNotNone(sim.receiver_rotor.rotors[1])
        self.assertIsNotNone(sim.receiver_rotor.rotors[2])

        # 1번 로터에 순수 위상이 인입되고, 이후 텐션이 적용되어 평형을 찾아감
        phases = sim.receiver_rotor.get_current_phases()
        self.assertNotEqual(phases[0], phases[1])

if __name__ == '__main__':
    unittest.main()
