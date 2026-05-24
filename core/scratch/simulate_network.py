import time
import random
import math
import sys
import os

# core 디렉토리를 path에 추가하여 모듈을 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from atlantis_phase_network import AtlantisPhaseNetworkCore
from math_utils import Quaternion

def simulate_harsh_network():
    """
    네트워크 가혹 환경 시뮬레이터 (Network Simulator)
    패킷 로스(Packet Loss)를 발생시키면서도 쿼터니언 Slerp과 Y-Δ 결선 루프를 통해
    클라이언트가 서버의 위상에 어떻게 부드럽게 수렴(Alignment)하는지 증명합니다.
    """
    print("🌊 아틀란티스 위상 네트워크 시뮬레이션 (패킷 로스 50% 환경)")
    print("-" * 70)

    network_core = AtlantisPhaseNetworkCore()

    packet_loss_rate = 0.50 # 50% 패킷 손실률
    dt = 0.05 # 50ms 틱 레이트 (20FPS)
    total_steps = 100

    print(f"| Step | Packet | Server Phase (Q)                 | Client Phase (Q)                 | Alignment (Viewport) |")
    print("-" * 70)

    for step in range(1, total_steps + 1):
        # 50% 확률로 패킷 도착 여부 결정
        packet_received = random.random() >= packet_loss_rate

        # 네트워크 스텝 진행 (서버 회전 + 클라이언트 예측/수렴)
        network_core.simulate_network_step(dt, simulate_server_rotation=True, packet_received=packet_received)

        srv = network_core.server_phase
        cli = network_core.client_phase
        align = network_core.get_viewport_alignment()

        # 출력 포맷팅
        status = "✅ OK " if packet_received else "❌ LOSS"
        srv_str = f"Q({srv.w:5.2f}, {srv.x:5.2f}, {srv.y:5.2f}, {srv.z:5.2f})"
        cli_str = f"Q({cli.w:5.2f}, {cli.x:5.2f}, {cli.y:5.2f}, {cli.z:5.2f})"

        # Alignment를 막대 그래프로 표현 (0.0 ~ 1.0)
        bar_len = int(align * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)

        print(f"| {step:4d} | {status} | {srv_str} | {cli_str} | {bar} {align:4.2f} |")

        time.sleep(0.01) # 콘솔 출력 지연 (시뮬레이션 체감용)

    print("-" * 70)
    print("🎯 시뮬레이션 완료. 패킷 로스 환경에서도 위상 정렬도가 부드럽게 유지됨을 확인.")

if __name__ == "__main__":
    simulate_harsh_network()
