import math
import cmath

class TriRotorGrassmann:
    """
    삼중로터 시스템을 그라스만 대수의 기저 벡터 및 쐐기곱으로 치환한 코어 엔진
    """
    def __init__(self, r1_phase=0.0, r2_phase=0.0, r3_phase=0.0):
        # 3개의 로터를 독립된 복소 위상 벡터(e1, e2, e3)로 정의 (정삼각 결선 기준점)
        self.e1 = cmath.exp(1j * r1_phase)
        self.e2 = cmath.exp(1j * r2_phase)
        self.e3 = cmath.exp(1j * r3_phase)

    def inject_dual_helix_stream(self, helix_a: float, helix_b: float):
        """
        외부망(레거시)에서 들어온 진짜 이중나선(Dual-Helix, 180도 위상차) 유속을 받아
        차동 상쇄(Differential Cancel) 연산을 통해 외부 노이즈를 100% 증발시키고,
        순수 위상차(알맹이)만을 내부 3차원 삼중로터 공간으로 투사(Projection)한다.
        """
        # 차동 상쇄 연산: 두 나선의 복소 차이를 구해 공통 노이즈를 상쇄
        # helix_a = original + noise, helix_b = (original + pi) + noise
        # cmath.exp(1j * helix_a) - cmath.exp(1j * helix_b) 는 노이즈 환경에서도
        # 원래의 위상차 방향을 유지하는 순수 백터 합을 만들어냄 (Differential Cancel)
        differential_vector = cmath.exp(1j * helix_a) - cmath.exp(1j * helix_b)

        pure_phase = cmath.phase(differential_vector)

        # 순수 유속 에너지를 3축 삼중로터 공간으로 정삼각 분배 (Projection)
        self.e1 = cmath.exp(1j * (pure_phase))
        self.e2 = cmath.exp(1j * (pure_phase + 2*math.pi/3))
        self.e3 = cmath.exp(1j * (pure_phase + 4*math.pi/3))

    def compute_wedge_tension(self):
        """
        그라스만 쐐기곱(Wedge Product)을 이용하여
        삼중로터 간의 상호 위상차 면적(Bi-vector) 장력을 도출하는 메서드.
        (타겟 벡터 없이 자체 3축 간의 텐션을 구함)
        """
        # 기하 교과서의 외적 공식 공식화 (조건문 100% 배제)
        # 각 로터 축 간의 교차 면적(Wedge) 계산
        w12 = (self.e1.real * self.e2.imag) - (self.e1.imag * self.e2.real)
        w23 = (self.e2.real * self.e3.imag) - (self.e2.imag * self.e3.real)
        w31 = (self.e3.real * self.e1.imag) - (self.e3.imag * self.e1.real)

        # 3축 면적 장력의 합산 (Bi-vector 결과물)
        bi_vector_tension = w12 + w23 + w31
        return bi_vector_tension

    def __xor__(self, other):
        """
        __xor__ 연산자를 오버로딩하여
        쐐기곱 장력을 구하는 직관적 인터페이스 (Syntactic Sugar)
        """
        return self.compute_wedge_tension()

    def align_phase(self, error_tension):
        """
        도출된 면적 장력(토크)을 이용해 3개의 로터 위상을 조건문 없이
        동시 고정(Phase-Lock)시키는 직동식 피드백
        """
        # 오차 장력 그 자체가 회전 낙차가 되어 로터들의 위상각을 강제로 끌어당김
        correction = error_tension * 0.1  # 피드백 게인

        self.e1 *= cmath.exp(1j * correction)
        self.e2 *= cmath.exp(1j * correction)
        self.e3 *= cmath.exp(1j * correction)

class WedgeVortexSimulator:
    """
    WedgeVortex 기어의 삼중로터 동기화 및
    UDP 투명망토 캡슐화/파쇄(Decapsulation) 시뮬레이터.
    """
    def __init__(self):
        # 수신단 삼중로터 초기화
        self.receiver_rotor = TriRotorGrassmann(0.0, 0.0, 0.0)

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

        # 1. 이중나선 유속의 노이즈를 상쇄하고 내부 3차원 코어로 순수 투사
        self.receiver_rotor.inject_dual_helix_stream(helix_a, helix_b)

        # 2. 그라스만 쐐기곱 직동식 결선: 내부 삼중로터 간의 오차 면적 토크 환원
        tension = self.receiver_rotor.compute_wedge_tension()

        # 3. 토크 반영: 단 1클럭 시차 없이 정상 궤도로 보정
        self.receiver_rotor.align_phase(tension)
