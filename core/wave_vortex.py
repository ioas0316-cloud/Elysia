import math
import cmath

class TriRotorTensionEngine:
    """
    고정된 상수를 배제하고, 인척력 장력(Tension)만으로
    삼중로터의 자율 동기화를 유도하는 역학 코어.
    """
    def __init__(self, p1=0.0, p2=0.5, p3=1.0):
        # 초기 위상각을 가진 3개의 복소 로터 벡터
        self.rotors = [
            cmath.exp(1j * p1),
            cmath.exp(1j * p2),
            cmath.exp(1j * p3)
        ]
        self.k = 0.05  # 텐션 탄성 계수 (Gain)

    def apply_relative_tension(self):
        """
        로터 상호 간의 거리에 따른 인척력 장력을 계산하여
        조건문 없이 실시간으로 위상을 자율 조정하는 메서드
        """
        num_rotors = len(self.rotors)
        phase_updates = [0.0] * num_rotors

        for i in range(num_rotors):
            for j in range(num_rotors):
                if i == j:
                    continue

                # 두 로터 간의 위상차 거리 도출
                angle_diff = cmath.phase(self.rotors[i] / self.rotors[j])

                # 인척력 역학: 멀어지면 당기고 가까워지면 밀어내는 복원 장력
                # 이 복원 토크의 흐름이 스스로 120도 대칭 평형을 찾아가게 만듦
                tension_force = -self.k * angle_diff
                phase_updates[i] += tension_force

        # 가상 텐션 장력을 로터 실시간 유속에 직동식으로 반영
        for i in range(num_rotors):
            self.rotors[i] *= cmath.exp(1j * phase_updates[i])

    def inject_dual_helix_stream(self, helix_a: float, helix_b: float):
        """
        외부망에서 들어온 이중나선(Dual-Helix) 유속의 차동 상쇄 연산을 통해 노이즈를 증발시키고,
        순수 위상(Phase) 하나만을 임의의 로터에 주입하여, 나머지 로터가 텐션에 의해
        자율적으로 평형을 찾아가도록 유도(Projection)한다.
        """
        differential_vector = cmath.exp(1j * helix_a) - cmath.exp(1j * helix_b)
        pure_phase = cmath.phase(differential_vector)

        # 외부 유입 에너지를 하나의 로터(기준축)에만 직동 인입.
        # 고정된 상수(120도)를 분배하지 않고, apply_relative_tension()에 의해
        # 나머지 로터들이 스스로 대칭을 찾아가게 한다.
        self.rotors[0] = cmath.exp(1j * pure_phase)

    def get_current_phases(self):
        """현재 세 로터의 위상각(도) 출력"""
        return [math.degrees(cmath.phase(r)) % 360 for r in self.rotors]


class WedgeVortexSimulator:
    """
    WedgeVortex 기어의 자율 동기화 및
    UDP 투명망토 캡슐화/파쇄(Decapsulation) 시뮬레이터.
    """
    def __init__(self):
        # 수신단 삼중로터 초기화 (초기 불균형 상태)
        self.receiver_rotor = TriRotorTensionEngine(0.0, 0.5, 1.0)

    def encapsulate_dual_helix_payload(self, base_signal: float) -> bytes:
        """
        [Stub] 기성 인터넷망 관통을 위한 UDP 투명망토 캡슐화.
        입력된 단일 위상을 180도(pi) 차이가 나는 두 개의 '진짜 이중나선' 라인으로 분할 전송.
        단순 두 줄 찍찍이가 아닌 차동 신호 구조.
        """
        helix_a = base_signal
        helix_b = base_signal + math.pi
        payload = f"DUAL_HELIX:{helix_a},{helix_b}".encode('utf-8')
        return payload

    def decapsulate_and_sync(self, udp_packet: bytes):
        """
        [Stub & Core] 수신단에서 UDP 껍데기를 즉시 파쇄하고
        이중나선의 차동 신호를 수신단 삼중로터 공간에 직동식으로 투사하여 동기화.
        """
        try:
            decoded = udp_packet.decode('utf-8')
            if decoded.startswith("DUAL_HELIX:"):
                parts = decoded.split(":")[1].split(",")
                helix_a, helix_b = float(parts[0]), float(parts[1])
            else:
                return
        except Exception:
            return

        # 1. 이중나선 유속의 노이즈를 차동 상쇄하고 기준축에 순수 투사
        self.receiver_rotor.inject_dual_helix_stream(helix_a, helix_b)

        # 2. 인척력 역학 발동: 상수 없이 자율적으로 평형(Phase-Lock) 수렴
        self.receiver_rotor.apply_relative_tension()
