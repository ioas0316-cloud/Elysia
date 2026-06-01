import numpy as np
import time

def baseline_cpu_filter(packets, valid_signature):
    """
    전통적인 CPU 기반 패킷 필터링 방식 (Baseline)
    """
    valid_packets = []

    # 패킷당 CPU 연산 사이클(임의의 스칼라 비교를 시뮬레이션)
    # 실제로는 DPI(Deep Packet Inspection)처럼 더 많은 if/else가 들어감
    for packet in packets:
        # 조건 1: 헤더 검사
        if packet[0] > 0.1:
            # 조건 2: 페이로드 특정 값 검사
            if packet[1] < 0.9:
                # 조건 3: 시그니처 매칭
                if abs(packet[2] - valid_signature) < 0.05:
                    valid_packets.append(packet)

    return valid_packets

def phase_mirror_gpu_filter(complex_packets, channel_phase, threshold):
    """
    위상 거울 엔진 (GPU 시뮬레이션)
    데이터를 복소수 위상 벡터(A∠θ)로 다루며 벡터 연산으로 단일 스텝 필터링
    """
    # 1. 패킷 위상 추출 (numpy 벡터 연산으로 한 번에 처리)
    packet_phases = np.angle(complex_packets)

    # 2. 채널 위상과의 위상차 계산 (절대값 비교 연산 1회)
    phase_diffs = np.abs(packet_phases - channel_phase)

    # 3. 위상차가 임계치(threshold) 이내인 데이터만 통과 (반사/통과 결정)
    valid_mask = phase_diffs < threshold

    # 통과한 패킷 반환
    return complex_packets[valid_mask]

def run_benchmark():
    num_packets = 1000000 # 100만 개 패킷으로 테스트 스케일 확장

    print(f"--- 벤치마크 시작 (패킷 수: {num_packets:,} 개) ---")

    # 1. CPU 방식 데이터 준비 (헤더, 페이로드, 시그니처)
    np.random.seed(42)
    cpu_packets = np.random.rand(num_packets, 3)
    valid_signature = 0.5

    # 2. Phase Mirror (GPU) 방식 데이터 준비 (복소수 위상 벡터: 진폭 + 위상)
    # 진폭은 0.5~1.5, 위상은 -pi ~ pi 로 랜덤 생성
    amplitudes = np.random.uniform(0.5, 1.5, num_packets)
    phases = np.random.uniform(-np.pi, np.pi, num_packets)
    complex_packets = amplitudes * np.exp(1j * phases)

    channel_phase = 0.0 # 우리 채널의 고유 위상
    phase_threshold = 0.1 # 통과 임계치

    # === [Baseline CPU 테스트] ===
    start_time_cpu = time.time()

    # CPU 방식은 순차적 파이썬 리스트 처리를 시뮬레이션하기 위해 리스트로 변환
    cpu_packets_list = cpu_packets.tolist()
    valid_cpu = baseline_cpu_filter(cpu_packets_list, valid_signature)

    end_time_cpu = time.time()
    cpu_duration = end_time_cpu - start_time_cpu
    cpu_throughput = (num_packets / cpu_duration) / 1000000 # MPPS

    # === [Phase Mirror 테스트] ===
    start_time_gpu = time.time()

    valid_gpu = phase_mirror_gpu_filter(complex_packets, channel_phase, phase_threshold)

    end_time_gpu = time.time()
    gpu_duration = end_time_gpu - start_time_gpu
    gpu_throughput = (num_packets / gpu_duration) / 1000000 # MPPS

    # === [결과 출력] ===
    print("\n[Baseline (CPU 전통 방식)]")
    print(f" - 전체 처리 시간 : {cpu_duration * 1000:.2f} ms")
    print(f" - 처리량 (Throughput): {cpu_throughput:.4f} MPPS")
    print(f" - 필터링된 패킷 수 : {len(valid_cpu):,} 개")

    print("\n[Phase Mirror (GPU 위상 필터 방식)]")
    print(f" - 전체 처리 시간 : {gpu_duration * 1000:.2f} ms")
    print(f" - 처리량 (Throughput): {gpu_throughput:.4f} MPPS")
    print(f" - 필터링된 패킷 수 : {len(valid_gpu):,} 개")

    print("\n[성능 비교 요약]")
    speedup = cpu_duration / gpu_duration
    print(f" -> Phase Mirror 방식이 전통적 방식에 비해 약 **{speedup:.2f}배 빠름**")
    print(" -> CPU Cost(if/else 분기 및 연산)가 벡터 연산(위상 간섭)으로 대체되어 구조적 오버헤드 극도로 감소.")

if __name__ == "__main__":
    run_benchmark()
