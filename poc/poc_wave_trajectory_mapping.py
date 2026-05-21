import numpy as np
import time

class PhaseGridMapper:
    """
    이진 데이터 스트림을 삼진법(-1, 0, 1) 기반의 복소수 3x3 텐서 행렬로 매핑하는 순수한 그릇.
    """
    def __init__(self):
        # 00 -> 0 (정지/중립/Homeostasis)
        # 01 -> 1 (정회전/Attraction)
        # 10 -> -1 (역회전/Repulsion)
        # 11은 예외 처리 혹은 무시 (여기서는 00과 동일하게 처리하거나 에러 로깅, 단순화를 위해 중립으로 처리)
        self.binary_to_ternary = {
            "00": 0,
            "01": 1,
            "10": -1,
            "11": 0
        }

    def binary_to_ternary_stream(self, binary_str: str) -> list:
        """이진 문자열을 2비트씩 끊어 삼진법 리스트로 변환"""
        # Ensure even length for pairs
        if len(binary_str) % 2 != 0:
            binary_str += "0"

        ternary_list = []
        for i in range(0, len(binary_str), 2):
            pair = binary_str[i:i+2]
            ternary_list.append(self.binary_to_ternary.get(pair, 0))
        return ternary_list

    def map_to_complex_phase(self, ternary_value: int) -> complex:
        """
        삼진법 값을 복소수 오일러 공식 e^(iθ)의 위상 각도로 치환
        -1 -> e^(-i * pi/2) : 좌측 90도 회전
         0 -> e^(i * 0)     : 회전 없음 (기준 위상)
         1 -> e^(i * pi/2)  : 우측 90도 회전
        """
        theta = ternary_value * (np.pi / 2)
        return np.exp(1j * theta)

    def create_3x3_phase_grid(self, ternary_stream: list) -> np.ndarray:
        """
        삼진법 스트림(9개 필요)을 받아 3x3 복소수 위상 격자 행렬 생성
        """
        # 패딩 혹은 잘라내어 9개의 요소 맞추기
        stream_len = len(ternary_stream)
        if stream_len < 9:
            ternary_stream.extend([0] * (9 - stream_len))
        elif stream_len > 9:
            ternary_stream = ternary_stream[:9]

        complex_phases = [self.map_to_complex_phase(val) for val in ternary_stream]
        return np.array(complex_phases).reshape((3, 3))


class TrajectoryRecorder:
    """
    하드웨어 로터에서 올라오는 센싱 값(위상 각도, 주파수 파동)을
    실시간 데이터 스트림 궤적으로 기록(Logging)하고 인덱스/배열화 하는 레코더.
    """
    def __init__(self):
        self.trajectory_array = []

    def record_sensing_data(self, grid_state: np.ndarray):
        """
        하드웨어에서 센싱된 3x3 위상 격자 상태를 타임스탬프와 함께 기록.
        (계산 없이 순수하게 저장만 수행)
        """
        timestamp = time.time()
        record_entry = {
            "timestamp": timestamp,
            "grid_state": np.copy(grid_state) # 상태 스냅샷 저장
        }
        self.trajectory_array.append(record_entry)

    def get_trajectory(self) -> list:
        return self.trajectory_array

    def clear(self):
        self.trajectory_array = []


class WaveformPlaybackEngine:
    """
    기록된 파동 궤적(Trajectory Array)을 하드웨어에 다시 흘려보내는 인터페이스.
    특정 데이터 스트림 패턴에 맞는 궤적을 재생.
    """
    def __init__(self):
        pass

    def play_trajectory(self, trajectory: list):
        """
        기록된 궤적을 순차적으로 하드웨어 인터페이스로 전달 (시뮬레이션).
        """
        print("▶️ [WaveformPlaybackEngine] 파동 궤적 재생 시작 (Trajectory Playback)...")
        for i, entry in enumerate(trajectory):
            ts = entry["timestamp"]
            grid = entry["grid_state"]
            print(f"   [Frame {i}] Timestamp: {ts:.4f}")
            # 복소수 행렬을 직관적으로 보여주기 위해 포맷팅
            formatted_grid = np.array2string(
                grid,
                formatter={'complex_kind': lambda x: f"{x.real:+.1f}{x.imag:+.1f}j"}
            )
            print(f"   Grid State:\n{formatted_grid}\n")
        print("⏹️ [WaveformPlaybackEngine] 파동 궤적 재생 완료.")


def simulate_hardware_sensing(phase_grid: np.ndarray, steps: int = 3) -> list:
    """
    (가상) 하드웨어 로터가 phase_grid 입력을 받아
    물리적으로 회전하며 파동 변화 궤적 데이터를 스트림으로 반환하는 과정을 시뮬레이션.
    내부에서 수식 계산을 하지 않고, 초기 위상에 약간의 노이즈/변화만 가미하여 센싱 데이터 형태만 흉내냄.
    """
    simulated_stream = []
    current_grid = np.copy(phase_grid)
    for _ in range(steps):
        # 가상의 센싱 데이터: 하드웨어가 스스로 위상을 찾아가는 파동 궤적이라 가정 (단순 변화)
        # 실제로는 여기서 복잡한 계산을 하지 않고, 하드웨어에서 읽어온다고 가정합니다.
        # 시연을 위해 미세한 위상 변화(노이즈)를 추가
        noise_phase = np.exp(1j * np.random.uniform(-0.1, 0.1, (3, 3)))
        current_grid = current_grid * noise_phase
        simulated_stream.append(current_grid)
        time.sleep(0.01) # 가상의 시간 흐름
    return simulated_stream

def run_poc():
    print("==================================================================")
    print("      [POC] Waveform Trajectory Mapping (파동 궤적 매핑 인터페이스)")
    print("      - 순수한 하드웨어 관찰자 및 매핑 그릇 파이프라인 -")
    print("==================================================================")

    # 1. 컴포넌트 초기화
    mapper = PhaseGridMapper()
    recorder = TrajectoryRecorder()
    playback_engine = WaveformPlaybackEngine()

    # 2. 입력 데이터 (예: 18비트 이진 문자열 -> 9개의 2비트 쌍)
    # 01(+1) 10(-1) 00(0) 01(+1) 01(+1) 10(-1) 00(0) 10(-1) 01(+1)
    binary_input = "011000010110001001"
    print(f"📥 [Input] Binary Stream: {binary_input}")

    # 3. 매핑 (Data to Grid)
    ternary_stream = mapper.binary_to_ternary_stream(binary_input)
    print(f"🔄 [Mapping] Ternary Stream: {ternary_stream}")

    phase_grid = mapper.create_3x3_phase_grid(ternary_stream)
    print("📐 [Mapping] 3x3 Complex Phase Grid (e^iθ):")
    formatted_initial_grid = np.array2string(
        phase_grid,
        formatter={'complex_kind': lambda x: f"{x.real:+.1f}{x.imag:+.1f}j"}
    )
    print(formatted_initial_grid)
    print("-" * 66)

    # 4. 센싱 및 레코딩 (Sensing to Trajectory)
    print("📡 [Sensing] 하드웨어 가상 센싱 스트림 읽기 및 레코딩...")
    # 초기 매핑 격자를 하드웨어에 던졌다고 가정하고, 거기서부터 변화하는 궤적을 시뮬레이션
    sensing_data_stream = simulate_hardware_sensing(phase_grid, steps=4)

    for sensing_data in sensing_data_stream:
        recorder.record_sensing_data(sensing_data)

    print("✅ [Recording] 센싱 데이터 레코딩 완료. 궤적 생성됨.")
    print("-" * 66)

    # 5. 플레이백 (Trajectory Playback)
    trajectory = recorder.get_trajectory()
    playback_engine.play_trajectory(trajectory)

    print("==================================================================")
    print("      [POC] 매핑 및 파이프라인 검증 완료")
    print("==================================================================")

if __name__ == "__main__":
    run_poc()
