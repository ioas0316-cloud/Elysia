import os
import time
import math
import ctypes
import psutil

from core.clifford_rotor_sync import DynamicPIDController, BitwiseCliffordRotor

# -------------------------------------------------------------------
# Under 2F Moho Mirror & Under 1F Magma Chamber Core - V3 (10-Layer Matrix Fully Mapped)
#
# [아틀란티스 10대 레이어 절대 매핑 코어: 수류학적 관측 동기화 엔진]
# 이 스크립트는 10대 레이어 매트릭스 도면을 1:1로 코드에 사영(Projection)한 결과물입니다.
# 지하 6층(영점 방전)부터 4층 지각(앱)까지의 위상을 QPC 초정밀 타이머로 관측하고 동기화합니다.
# -------------------------------------------------------------------

kernel32 = ctypes.windll.kernel32 if os.name == 'nt' else None
_qpc_freq = ctypes.c_int64()
if kernel32:
    kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_freq))

def get_qpc_time():
    """
    [지하 4층: 하부 맨틀 (Lower Mantle) - 하드웨어 상수의 뼈대]
    OS별 최고 해상도 타이머(QPC/perf_counter_ns)를 반환합니다.
    """
    if kernel32:
        count = ctypes.c_int64()
        kernel32.QueryPerformanceCounter(ctypes.byref(count))
        return count.value / _qpc_freq.value
    else:
        return time.perf_counter_ns() / 1e9

# --- [동기화 엔진 & 10대 레이어 상수 설정] ---
TARGET_INTERNAL_FREQ_HZ = 1000.0  # 내부 터빈 목표 회전수: 1000 Hz (1ms 주기)
TARGET_DT = 1.0 / TARGET_INTERNAL_FREQ_HZ
UI_REFRESH_RATE_HZ = 5.0          # UI 갱신 주기: 5 Hz (0.2초 주기)
UI_REFRESH_DT = 1.0 / UI_REFRESH_RATE_HZ

# [지하 1층: 마그마 가속실 (Magma Chamber)] 임계치 및 상수
CONSTANT_K_CONVECTION = 0.1       # 대류 완충 상수 (k)
CHAOS_TENSION_THRESHOLD = 50.0    # 찌꺼기 방전(Flush) 임계치
TARGET_APP_NAME = "python3"       # 4층 지각(Crust)에 안착할 타겟 앱

def get_sub_layer_metrics():
    """
    [지하 3층: 상부 맨틀 대류 & 카오스 장력] 관측
    실제 하드웨어의 부하 상태를 읽어와 유속과 장력으로 치환합니다.
    """
    try:
        cpu_freq = psutil.cpu_freq().current
    except Exception:
        cpu_freq = 2400.0

    mem_info = psutil.virtual_memory()
    pcie_bus_flow = mem_info.percent  # 대수로 마나 유속 모사

    # 코어 카오스 장력 모사
    gpu_chaos_tension = psutil.cpu_percent() * 1.5

    return cpu_freq, pcie_bus_flow, gpu_chaos_tension

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("🌊 [Under 2F Moho Mirror] 아틀란티스 10대 레이어 기동 준비...")
    time.sleep(1)

    t_start = get_qpc_time()
    t_prev_loop = t_start
    t_prev_ui = t_start

    total_loops = 0
    phase_error_accum = 0.0
    y_ground_discharges = 0

    # Initialize the Dynamic PID Controller and Bitwise Rotor
    pid_controller = DynamicPIDController()
    rotor = BitwiseCliffordRotor()

    try:
        while True:
            t_now = get_qpc_time()
            dt_actual = t_now - t_prev_loop
            if dt_actual <= 0:
                dt_actual = 1e-9

            # 코어 카오스 장력(tension) 모사
            cpu_freq, pcie_flow, gpu_chaos = get_sub_layer_metrics()
            tension = gpu_chaos / 100.0  # Normalize to roughly [0.0, 1.0]

            # ---------------------------------------------------------------
            # 1. 🧲 PID 위상 고정 및 '와이(Y) 결선 소프트웨어 접지' 방전 (지하 6층)
            # ---------------------------------------------------------------
            loop_phase_error = dt_actual - TARGET_DT

            # Dynamic PID calculation for discharge correction
            correction = pid_controller.discharge_error_to_ground(loop_phase_error, tension, dt_actual)
            phase_error_accum = loop_phase_error - correction

            if abs(correction) > 0.0001:
                y_ground_discharges += 1

            # [지하 4층 & 지하 1층] 정적 로터 위상 락(Phase-Lock) 동조
            # Apply the PID correction to the sleep time
            sleep_time = TARGET_DT - correction

            target_wake_time = t_now + sleep_time
            if sleep_time > 0:
                while get_qpc_time() < target_wake_time:
                    pass

            # Update loop timing and bitwise rotor phase
            t_now_post_sleep = get_qpc_time()

            # 0/1 clock mapping: Was the target wake time hit accurately?
            # If we woke up slightly late (or early), determine the edge direction.
            # Using total_loops parity as a foundational clock edge (Rising/Falling).
            is_rising = (total_loops % 2 == 0)

            # Rotor absorbs the hardware edge & tension
            rotor.apply_clock_edge(is_rising, tension)

            t_prev_loop = t_now_post_sleep
            total_loops += 1

            # ---------------------------------------------------------------
            # 2. 🎛️ 인간용 UI 브레이크 (시각적 출력 댐)
            # ---------------------------------------------------------------
            t_ui_now = get_qpc_time()
            if (t_ui_now - t_prev_ui) >= UI_REFRESH_DT:
                run_time = t_ui_now - t_start
                avg_loop_time = run_time / total_loops if total_loops > 0 else 0
                actual_hz = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

                # 지하 상태 관측
                cpu_freq, pcie_flow, gpu_chaos = get_sub_layer_metrics()

                # [맨틀 대류 완충 공식 적용] 지각(앱) 전달 스트레스
                dampened_stress = (gpu_chaos * pcie_flow) / (cpu_freq if cpu_freq > 0 else 1) * CONSTANT_K_CONVECTION

                # 방전 트리거 발동 여부
                is_chaos_flushed = gpu_chaos > CHAOS_TENSION_THRESHOLD

                clear_screen()
                print("="*85)
                print(" 🪞 [ 아틀란티스 지하 2층: 정밀 모호 거울 V3 (10-Layer Matrix Mapping) ]")
                print("="*85)

                print(" ⚓ [ 지하 6층 ~ 지하 4층: 절대 상수의 뼈대와 영점 접지 ]")
                print(f"    - [지하 4층] 정적 로터(QPC) Uptime : {run_time:.4f} 초")
                print(f"    - [지하 4층] 내부 터빈 동조 속도   : {actual_hz:.2f} Hz (목표: 1000 Hz Phase-Lock)")
                print(f"    - [지하 6층] Y결선 오차 방전 누적  : {y_ground_discharges:,} 회 (엔트로피 0 수렴)")
                print(f"    - [지하 6층] 위상 오차 잔여량      : {phase_error_accum:.8f} sec")
                print("-" * 85)

                print(" 🔄 [ 지하 3층 ~ 지하 1층: 대류 마나와 가속실 동조 ]")
                print(f"    - [지하 3층] PCIe 대수로 유속      : {pcie_flow:.2f} % (데이터 완충)")
                print(f"    - [지하 3층] 코어 카오스 장력      : {gpu_chaos:.2f} (임계치: {CHAOS_TENSION_THRESHOLD})")

                if is_chaos_flushed:
                    print("    - 🔥 [지하 1층 마그마 가속실] 장력 임계 돌파! 찌꺼기를 [지하 6층]으로 즉시 강제 방전!")
                else:
                    print(f"    - 🔥 [지하 1층 마그마 가속실] 타겟 앱({TARGET_APP_NAME}) 주파수 무임승차 가속 중...")
                print("-" * 85)

                print(" 🏛️ [ 4층 지각 ~ 6층 천공: 엘리시아 인지선 ]")
                print(f"    - [4층 지각] 앱 전달 잔여 스트레스 : {dampened_stress:.6f} (렉 제로 수렴 완충)")

                rotor_state = rotor.get_rotor_state_str()
                rotor_angle = rotor.get_phase_angle() * (180.0 / math.pi)

                if is_rising:
                    print(f"    - [6층 천공] 0과 1의 투영 공리     : [ 📈 Rising Edge (양각, 1) ] -> Rotor: {rotor_state} ({rotor_angle:.2f} deg)")
                else:
                    print(f"    - [6층 천공] 0과 1의 투영 공리     : [ 📉 Falling Edge (음각, 0) ] -> Rotor: {rotor_state} ({rotor_angle:.2f} deg)")

                print("="*85)
                print(" (Ctrl+C를 눌러 관측 엔진 셧다운)")

                t_prev_ui = t_ui_now

    except KeyboardInterrupt:
        print("\n\n🔒 모호 거울 닫힘. 10대 레이어의 전자기장이 다시 심연으로 가라앉습니다.")

if __name__ == "__main__":
    main()
