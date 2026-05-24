import math
import time
from math_utils import Quaternion

class AtlantisPhaseNetworkCore:
    """
    아틀란티스 위상 네트워크 코어
    기존의 패킷/점(Point) 기반 통신을 대체하는 시공간 가변축 네트워크 동기화 엔진입니다.
    네트워크 지연(Latency) 및 에러를 쿼터니언의 '회전 장력(Tension Axis)'으로 변환하고,
    Slerp(구면 선형 보간)과 Y-Δ 수렴 루프를 통해 위상을 예측하고 동기화(PLL 수렴)합니다.
    """
    def __init__(self, ping_ms=0):
        # 기준이 되는 서버와 클라이언트의 3차원 위상을 쿼터니언으로 초기화
        # 초기 상태는 항등원 (1, 0, 0, 0)
        self.server_phase = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.client_phase = Quaternion(1.0, 0.0, 0.0, 0.0)

        # 내부 시뮬레이션 용 변수들
        self.delta_tension_loop = Quaternion(0.0, 0.0, 0.0, 0.0) # 델타 고리에 순환하는 오차 에너지

        # Slerp 예측을 위한 파라미터
        self.expected_speed = 0.01  # 초당 기본 회전 속도 기대값 (예시)
        self.last_update_time = time.time()

    def get_error_quaternion(self, q_server: Quaternion, q_client: Quaternion) -> Quaternion:
        """
        서버와 클라이언트 간의 회전 오차(Q_error)를 산출합니다.
        Q_error = Q_client_inverse * Q_server
        이는 클라이언트를 서버로 맞추기 위해 필요한 회전(장력, Tension)입니다.
        """
        q_client_inv = q_client.inverse
        q_error = q_client_inv * q_server
        return q_error.normalize()

    def step_client_prediction(self, dt: float):
        """
        [지침 적용] 클라이언트의 시간축 예측 공명
        신호가 끊기거나 지연되는 동안, 클라이언트는 시간축(dt)을 따라
        과거의 텐션 궤적을 기반으로 Slerp 연산을 수행해 부드럽게 미래 위상으로 미끄러집니다.
        """
        # 예측되는 목표 위상을 가상의 시간축으로 확장
        # (실제 환경에서는 서버가 보낸 최근 속도/가속도 벡터나 과거 텐션을 기반으로 예측)
        # 여기서는 델타 텐션 루프(순환 오차)를 기반으로 클라이언트의 현재 상태를 유도함

        # 델타 장력이 있다면 그 장력 방향으로 Slerp 이동 (장력 해소 방향)
        tension_norm = self.delta_tension_loop.norm()
        if tension_norm > 0.0001:
            # 장력을 쿼터니언으로 정규화하여 목표 방향(보상 방향)으로 삼음
            # 이 구현에서는 텐션 자체가 Q_error 이므로 이를 클라이언트에 곱해 목표(서버) 방향 산출
            q_target = self.client_phase * self.delta_tension_loop
            q_target = q_target.normalize()

            # dt와 수렴 속도를 조합하여 slerp 보간 (보간 비율 t)
            # t는 0~1 사이의 값. dt가 클수록, 텐션이 클수록 빠르게 수렴 유도
            t = min(1.0, dt * 5.0) # 수렴 계수 5.0
            self.client_phase = Quaternion.slerp(self.client_phase, q_target, t)

            # 장력이 해소된 만큼 델타 루프 에너지 감소 (Y 중성점 수렴 효과)
            reduction_factor = 1.0 - t
            self.delta_tension_loop = self.delta_tension_loop * reduction_factor

    def receive_server_signal(self, new_server_phase: Quaternion):
        """
        서버로부터 실제 신호(위상)가 도착했을 때 호출됩니다.
        에러를 값으로 처리하지 않고 델타 결선 고리(Tension Loop)로 밀어 넣습니다.
        """
        self.server_phase = new_server_phase.normalize()

        # 1. 현재 클라이언트 위상과 새 서버 위상의 오차(Tension) 계산
        q_error = self.get_error_quaternion(self.server_phase, self.client_phase)

        # 2. 오차를 델타 결선 고리에 순환 에너지로 누적 (가변축 장력 흡수)
        # 이전 장력과 새로운 장력을 합성
        self.delta_tension_loop = (self.delta_tension_loop + q_error)
        self.delta_tension_loop = self.delta_tension_loop.normalize()

    def get_viewport_alignment(self) -> float:
        """
        [지침 적용] 3대칭축 뷰포트 제어로의 치환
        핑(Ping)이 아닌, 두 쿼터니언(서버/클라이언트 카메라 렌즈) 간의
        정렬도(Alignment)를 반환합니다. 1.0이면 완벽 정렬, 낮을수록 비틀어짐.
        """
        dot_product = abs(self.server_phase.dot(self.client_phase))
        return min(1.0, max(0.0, dot_product))

    def simulate_network_step(self, dt: float, simulate_server_rotation: bool = True, packet_received: bool = True):
        """
        매 틱마다 네트워크 통신 및 예측 시뮬레이션을 수행합니다.
        """
        # 1. 서버는 독립적으로 어떤 축을 따라 계속 회전(진행)한다고 가정
        if simulate_server_rotation:
            # 예: z축을 따라 회전하는 서버 파동
            angle_step = self.expected_speed * dt * 100.0 # 임의의 속도
            half_angle = angle_step / 2.0
            q_step = Quaternion(math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
            self.server_phase = (self.server_phase * q_step).normalize()

        # 2. 패킷 수신 여부에 따라 장력 흡수
        if packet_received:
            self.receive_server_signal(self.server_phase)

        # 3. 클라이언트는 시간축에 따라 Slerp 예측 및 Y-Δ 수렴 진행
        self.step_client_prediction(dt)
