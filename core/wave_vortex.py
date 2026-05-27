import math

class PhaseVector:
    """
    시간축 위상 주파수를 나타내는 벡터 클래스.
    그라스만 쐐기곱(^, XOR 오버로딩)을 통해 위상차(Torque)를 계산하여,
    if/else 조건문 없이 실시간으로 동기화를 구현한다.
    """
    def __init__(self, phase_angle: float):
        # phase_angle: 위상각 (라디안 또는 정규화된 값)
        self.phase_angle = phase_angle % (2 * math.pi)

    def __xor__(self, other):
        """
        그라스만 쐐기곱(Wedge Product) 오버로딩.
        두 위상 벡터가 같으면 0(동기화 상태, 병목 없음)으로 수렴하며,
        다르면 그 차이만큼의 면적 전압(Bi-vector) 즉, 복원 토크(Torque)를 반환한다.
        """
        if not isinstance(other, PhaseVector):
            raise TypeError("쐐기곱은 PhaseVector 간에만 가능합니다.")

        # 위상차를 계산하되 가장 짧은 경로의 각도 차이(Torque)를 도출
        diff = (self.phase_angle - other.phase_angle) % (2 * math.pi)

        # -pi ~ pi 범위의 회전 토크로 정규화 (if문 없이 수학적 계산만으로)
        # diff가 pi보다 크면 반대 방향(-방향)으로 당기는 것이 더 빠름
        torque = diff - (2 * math.pi) * (diff > math.pi)

        # 완전한 동기화(오차 0)인 경우 0.0을 반환
        return torque

    def apply_torque(self, torque: float):
        """
        발생한 회전 장력(Torque)을 가변축 로터에 직동식으로 반영하여
        즉시 위상을 보정(동기화)한다.
        """
        self.phase_angle = (self.phase_angle + torque) % (2 * math.pi)


class WedgeVortexSimulator:
    """
    WedgeVortex 기어의 삼중로터 동기화 및
    UDP 투명망토 캡슐화/파쇄(Decapsulation) 시뮬레이터.
    """
    def __init__(self):
        # 수신단 로터의 초기 위상 설정
        self.receiver_rotor = PhaseVector(0.0)

    def encapsulate_udp_payload(self, phase_signal: float) -> bytes:
        """
        [Stub] 기성 인터넷망 관통을 위한 UDP 투명망토 캡슐화.
        삼중로터의 가변 위상 비트 스트림을 일반 UDP 패킷 상자에 밀어 넣음.
        """
        # 실제 네트워크 전송 시 직렬화 로직 스텁
        payload = f"PHASE_PAYLOAD:{phase_signal}".encode('utf-8')
        return payload

    def decapsulate_and_sync(self, udp_packet: bytes):
        """
        [Stub & Core] 수신단에서 UDP 껍데기를 즉시 파쇄하고
        알맹이(위상 신호)를 수신단 로터에 직동식으로 결선하여 동기화.
        """
        # 1. 상자 파쇄 (디코딩 스텁)
        try:
            decoded = udp_packet.decode('utf-8')
            if decoded.startswith("PHASE_PAYLOAD:"):
                received_phase = float(decoded.split(":")[1])
            else:
                return  # 유효하지 않은 패킷 버림
        except Exception:
            return

        incoming_vector = PhaseVector(received_phase)

        # 2. 그라스만 쐐기곱(^) 직동식 결선: 토크 환원
        # if문 없이 (incoming ^ receiver) 연산만으로 오차 면적(Torque) 추출
        torque = incoming_vector ^ self.receiver_rotor

        # 3. 토크 반영: 단 1클럭 시차 없이 정상 궤도로 보정
        self.receiver_rotor.apply_torque(torque)

        # 동기화 후에는 incoming_vector.phase_angle == self.receiver_rotor.phase_angle 상태가 됨
