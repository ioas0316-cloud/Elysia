import os
import time
import math
import ctypes
import psutil
import sys
import gc
import threading

# Add root folder to sys.path to resolve imports properly if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.clifford_rotor_sync import DynamicPIDController, BitwiseCliffordRotor
from core.atlantis_clifford_bridge import AtlantisCliffordSystem

# -------------------------------------------------------------------
# Under 2F Moho Mirror & Under 1F Magma Chamber Core - V5 (Hardened Edition)
#
# [아틀란티스 10대 레이어 절대 매핑 코어: 수류학적 관측 동기화 엔진]
# 이 스크립트는 10대 레이어 매트릭스를 Clifford Multivector Cl(N,0)으로
# 관리하며, 단순 비유를 넘어 실제 OS API와 하드웨어 I/O 메트릭을 동기화합니다.
# -------------------------------------------------------------------

# 1. Windows API 및 QPC 타이머 설정
kernel32 = ctypes.windll.kernel32 if os.name == 'nt' else None
user32 = ctypes.windll.user32 if os.name == 'nt' else None

_qpc_freq = ctypes.c_int64()
if kernel32:
    kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_freq))

def get_qpc_time():
    """
    [지하 4층: 하부 맨틀 (Lower Mantle) - QPC 절대 클럭 기저]
    """
    if kernel32:
        count = ctypes.c_int64()
        kernel32.QueryPerformanceCounter(ctypes.byref(count))
        return count.value / _qpc_freq.value
    else:
        return time.perf_counter_ns() / 1e9

# --- [Windows foreground & process priority helpers] ---
def get_foreground_process_name() -> str:
    """Gets the executable name of the current foreground window's process."""
    if not user32:
        return ""
    try:
        hwnd = user32.GetForegroundWindow()
        lpdw_process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(lpdw_process_id))
        pid = lpdw_process_id.value
        proc = psutil.Process(pid)
        return proc.name().lower()
    except Exception:
        return ""

def set_process_priority(pid: int, priority_class: int) -> bool:
    """Sets the process priority class of a PID (Windows only)."""
    if sys.platform != 'win32':
        return False
    try:
        proc = psutil.Process(pid)
        proc.nice(priority_class)
        return True
    except (psutil.AccessDenied, Exception):
        return False

def find_process_pid_by_name(name: str) -> int:
    """Finds the PID of a running process by its name."""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if name.lower() in proc.info['name'].lower():
                return proc.info['pid']
    except Exception:
        pass
    return None

# --- [동기화 엔진 & 10대 레이어 상수 설정] ---
TARGET_INTERNAL_FREQ_HZ = 1000.0  # 내부 터빈 목표 회전수: 1000 Hz (1ms 주기)
TARGET_DT = 1.0 / TARGET_INTERNAL_FREQ_HZ
UI_REFRESH_RATE_HZ = 5.0          # UI 갱신 주기: 5 Hz (0.2초 주기)
UI_REFRESH_DT = 1.0 / UI_REFRESH_RATE_HZ

# [지하 1층: 마그마 가속실 (Magma Chamber)] 임계치 및 상수
CONSTANT_K_CONVECTION = 0.1       # 대류 완충 상수 (k)
CHAOS_TENSION_THRESHOLD = 50.0    # 찌꺼기 방전(Flush) 임계치
TARGET_APP_NAME = "python"        # 4층 지각(Crust)에 안착할 타겟 앱 (모니터링용)

# --- [Disk IO 상태 계측 도구] ---
_prev_disk_bytes = 0

def get_disk_io_flow(dt: float) -> float:
    """
    [지하 3층: 상부 맨틀 대류 - 실시간 디스크 입출력 대용량 마나 유속 계측]
    """
    global _prev_disk_bytes
    try:
        io = psutil.disk_io_counters()
        total_bytes = io.read_bytes + io.write_bytes
        if _prev_disk_bytes == 0:
            _prev_disk_bytes = total_bytes
            return 0.1 # 초기 기저선
        delta = total_bytes - _prev_disk_bytes
        _prev_disk_bytes = total_bytes
        # 10MB/s를 100% 임계 유속으로 상정하여 정규화
        normalized = delta / (10.0 * 1024 * 1024 * dt)
        return min(1.0, max(0.01, normalized))
    except Exception:
        return 0.1

def get_sub_layer_metrics():
    """
    [지하 3층/4층: 물리 기저 상태 계측]
    """
    try:
        cpu_freq = psutil.cpu_freq().current
    except Exception:
        cpu_freq = 2400.0

    gpu_chaos_tension = psutil.cpu_percent() * 1.5 # CPU 연산 부하율로 텐션 대변
    return cpu_freq, gpu_chaos_tension

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_bar_chart(val: float, max_len: int = 15) -> str:
    """Returns a visual progress bar string for aesthetics."""
    val = max(0.0, min(1.0, val))
    filled = int(val * max_len)
    empty = max_len - filled
    return f"[{'=' * filled}{' ' * empty}]"

def main():
    global _prev_disk_bytes
    print("🌊 [Under 2F Moho Mirror] 아틀란티스 10대 레이어 기동 준비 (정밀 제어 연동)...")
    time.sleep(1)

    t_start = get_qpc_time()
    t_prev_loop = t_start
    t_prev_ui = t_start

    total_loops = 0
    phase_error_accum = 0.0
    y_ground_discharges = 0
    priority_elevated = False

    # 컴포넌트 초기화
    pid_controller = DynamicPIDController()
    rotor = BitwiseCliffordRotor()
    clifford_system = AtlantisCliffordSystem()
    state_lock = threading.Lock() # B2_MohoMirror 단방향 관측 락(Lock) 실체화

    # 타겟 프로세스 PID 검색
    target_pid = os.getpid() # 기본값은 자기 자신
    found_pid = find_process_pid_by_name(TARGET_APP_NAME)
    if found_pid:
        target_pid = found_pid

    try:
        while True:
            t_now = get_qpc_time()
            dt_actual = t_now - t_prev_loop
            if dt_actual <= 0:
                dt_actual = 1e-9

            # 물리 기저 수치 측정
            cpu_freq, gpu_chaos = get_sub_layer_metrics()
            tension = gpu_chaos / 100.0  # Normalize to [0.0, 1.0]

            # ---------------------------------------------------------------
            # 1. 🧲 PID 위상 고정 및 'B6_Ground' 쓰레기 수집 방전 (지하 6층)
            # ---------------------------------------------------------------
            loop_phase_error = dt_actual - TARGET_DT

            # Dynamic PID calculation for discharge correction
            correction = pid_controller.discharge_error_to_ground(loop_phase_error, tension, dt_actual)
            phase_error_accum = loop_phase_error - correction

            if abs(correction) > 0.0001:
                y_ground_discharges += 1

            # [지하 1층 마그마 가속실] 방전 트리거 발동 시 실제 메모리 가비지 컬렉터 가동 및 PID 오차 영점화
            is_chaos_flushed = gpu_chaos > CHAOS_TENSION_THRESHOLD
            if is_chaos_flushed:
                gc.collect() # B6 Ground 실체화: 가비지 컬렉션 방전
                pid_controller.integral = 0.0 # 누적 제어 오차 초기화

            # [지하 4층] QPC 기저 시간 동조 지연 슬립
            sleep_time = TARGET_DT - correction
            target_wake_time = t_now + sleep_time
            if sleep_time > 0:
                while get_qpc_time() < target_wake_time:
                    pass

            t_now_post_sleep = get_qpc_time()
            is_rising = (total_loops % 2 == 0)

            # Rotor absorbs the hardware edge & tension
            rotor.apply_clock_edge(is_rising, tension)

            # ---------------------------------------------------------------
            # 2. 📐 B2_MohoMirror 단방향 락 안전장치 기반의 Clifford 상태 매핑
            # ---------------------------------------------------------------
            with state_lock:
                # (1) B6_Ground: 영점 접지값 기입
                clifford_system.set_layer_state("B6_Ground", 0.0 if not is_chaos_flushed else 1.0)
                
                # (2) B5_OuterCore: swap 메모리 점유율을 통해 물리 완충 유량 대변
                swap_util = psutil.swap_memory().percent / 100.0
                clifford_system.set_layer_state("B5_OuterCore", swap_util)
                
                # (3) B4_LowerMantle: CPU 주파수 상수
                clifford_system.set_layer_state("B4_LowerMantle", min(1.0, cpu_freq / 5000.0))
                
                # (4) B3_UpperMantle: 실제 디스크 I/O 유속 매핑 (물리 버스 데이터 흐름)
                disk_flow = get_disk_io_flow(dt_actual)
                clifford_system.set_layer_state("B3_UpperMantle", disk_flow)
                
                # (5) B2_MohoMirror: 락 성공 지표로 1.0 (상수 관측) 기입
                clifford_system.set_layer_state("B2_MohoMirror", 1.0)
                
                # (6) B1_MagmaChamber & F4_AppCrust: 포그라운드 윈도우 감지 및 가속 설정
                foreground_app = get_foreground_process_name()
                is_focused = TARGET_APP_NAME.lower() in foreground_app or "python" in foreground_app
                
                # 가속실 실체화: 포그라운드 포커스 시 Windows 스케줄러 우선순위를 높은 단계로 변조
                if is_focused:
                    elevated = set_process_priority(target_pid, 0x00000080) # HIGH_PRIORITY_CLASS
                    priority_elevated = elevated
                    clifford_system.set_layer_state("B1_MagmaChamber", 1.0 if elevated else 0.5)
                else:
                    set_process_priority(target_pid, 0x00000020) # NORMAL_PRIORITY_CLASS
                    priority_elevated = False
                    clifford_system.set_layer_state("B1_MagmaChamber", 0.1)

                # (7) F1_F3_SubCrust: 파이썬 실행 스레드 수 및 프로세스 메모리 RSS 비율 매핑
                active_threads = threading.active_count()
                proc_self = psutil.Process(os.getpid())
                mem_rss = proc_self.memory_info().rss
                subcrust_val = min(1.0, (active_threads / 15.0 + mem_rss / (150 * 1024 * 1024)) / 2.0)
                clifford_system.set_layer_state("F1_F3_SubCrust", subcrust_val)

                # (8) F4_AppCrust: 타겟 앱 포커스 여부에 따른 활성 스트레스율 (focused -> 스트레스 0v 수렴)
                dampened_stress = (gpu_chaos * disk_flow) / (cpu_freq if cpu_freq > 0 else 1) * CONSTANT_K_CONVECTION
                if is_focused:
                    dampened_stress *= 0.1 # 포커스 시 렉 영점(0) 완충 적용
                clifford_system.set_layer_state("F4_AppCrust", min(1.0, dampened_stress))

                # (9) F5_Atmosphere & F6_SkySun 위상 갱신
                atmosphere_noise = (disk_flow) * tension
                clifford_system.set_layer_state("F5_Atmosphere", atmosphere_noise)
                
                ground_val = clifford_system.get_layer_state("B6_Ground")
                mantle_val = clifford_system.get_layer_state("B4_LowerMantle")
                clifford_system.set_layer_state("F6_SkySun", (ground_val + mantle_val) / 2.0)

                # B3(디스크I/O)에서 B6(접지)로 에너지를 회전 방전시키는 Clifford Rotor 작동
                discharge_angle = abs(correction) * 50.0
                clifford_system.apply_rotor_discharge("B3_UpperMantle", "B6_Ground", discharge_angle)
                
                # B4(타이머)에서 B1(가속실)로 클럭 위상을 맞추는 동조 Rotor 작동
                sync_angle = (1.0 - tension) * 0.02
                clifford_system.apply_rotor_discharge("B4_LowerMantle", "B1_MagmaChamber", sync_angle)

            t_prev_loop = t_now_post_sleep
            total_loops += 1

            # ---------------------------------------------------------------
            # 3. 🎛️ 인간용 UI 브레이크 (시각적 출력 댐)
            # ---------------------------------------------------------------
            t_ui_now = get_qpc_time()
            if (t_ui_now - t_prev_ui) >= UI_REFRESH_DT:
                run_time = t_ui_now - t_start
                avg_loop_time = run_time / total_loops if total_loops > 0 else 0
                actual_hz = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

                clear_screen()
                print("="*95)
                print(" 🪞 [ 아틀란티스 지하 2층: 정밀 모호 거울 V5 (API & Metric Hardened Edition) ]")
                print("="*95)

                print(" ⚓ [ 물리 인프라 상태 및 우선순위 ]")
                print(f"    - [지하 4층] QPC Uptime : {run_time:.4f} 초")
                print(f"    - [지하 4층] 내부 터빈 속도 : {actual_hz:.2f} Hz (목표: 1000 Hz Phase-Lock)")
                print(f"    - [지하 6층] 누적 접지 방전 : {y_ground_discharges:,} 회 (엔트로피 0 수렴)")
                print(f"    - [포그라운드 윈도우 앱]    : {foreground_app if foreground_app else 'N/A'}")
                print(f"    - [타겟 앱({TARGET_APP_NAME}) 가속] : PID {target_pid} | 우선순위 등급: {'HIGH' if priority_elevated else 'NORMAL'}")
                print("-" * 95)

                with state_lock:
                    print(" 🔄 [ Clifford Algebra Cl(10,0) 가변축 실동 레이어 매핑 ]")
                    print(f"    * 활성 차원 서명 : Cl({len(clifford_system.layers)}, 0) - 가변축 상태")
                    
                    # Print layers visually
                    for layer_name in clifford_system.layers:
                        val = clifford_system.get_layer_state(layer_name)
                        bar = get_bar_chart(val)
                        print(f"    - {layer_name:<20} : {val:.6f} {bar}")
                        
                    print("-" * 95)

                    # Show bivector tension details
                    b3_b6_tension = clifford_system.compute_bivector_tension("B3_UpperMantle", "B6_Ground")
                    b4_b1_tension = clifford_system.compute_bivector_tension("B4_LowerMantle", "B1_MagmaChamber")
                    
                    print(" 📐 [ 기하학적 평면 텐션 (Bivector Wedge Magnitude) ]")
                    print(f"    - e4 ^ e1 (상부맨틀 ∧ 접지)   : {b3_b6_tension:.6f} (실제 I/O ∧ GC 방전 텐션)")
                    print(f"    - e3 ^ e6 (하부맨틀 ∧ 가속실) : {b4_b1_tension:.6f} (클럭 타이머 ∧ 가속 동조 텐션)")
                    print("-" * 95)

                print(" 🌊 [ 실시간 가비지 컬렉션 & 스케줄러 제어 현황 ]")
                if is_chaos_flushed:
                    print("    - 🔥 [지하 1층 마그마 가속실] 장력 임계 돌파! gc.collect() 강제 가동 및 PID 적분 리셋!")
                else:
                    print(f"    - 🔥 [지하 1층 마그마 가속실] 타겟 앱({TARGET_APP_NAME}) 활성 포커싱 가속 기동 중...")
                
                rotor_state = rotor.get_rotor_state_str()
                rotor_angle = rotor.get_phase_angle() * (180.0 / math.pi)

                if is_rising:
                    print(f"    - [6층 천공] 0과 1의 투영 공리   : [ 📈 Rising Edge (양각, 1) ] -> Rotor: {rotor_state} ({rotor_angle:.2f}°)")
                else:
                    print(f"    - [6층 천공] 0과 1의 투영 공리   : [ 📉 Falling Edge (음각, 0) ] -> Rotor: {rotor_state} ({rotor_angle:.2f}°)")

                print("="*95)
                print(" (Ctrl+C를 눌러 관측 엔진 셧다운)")

                t_prev_ui = t_ui_now

    except KeyboardInterrupt:
        print("\n\n🔒 모호 거울 닫힘. 10대 레이어의 전자기장이 다시 심연으로 가라앉습니다.")

if __name__ == "__main__":
    main()
