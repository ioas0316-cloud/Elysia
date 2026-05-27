import os
import time
import math
import ctypes
import psutil
import sys
import gc
import threading

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add root folder to sys.path to resolve imports properly if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
from core.sentence_wave_gate import SentenceWaveGate

from core.clifford_rotor_sync import DynamicPIDController, BitwiseCliffordRotor
from core.atlantis_clifford_bridge import AtlantisCliffordSystem
from core.electromagnetic_circuit import ElectromagneticCircuit
from core.autopoiesis_sandbox import SovereignAutopoiesisEngine

# 세계 하이퍼 로터 라이브러리 로드
from core.world_hyper_rotor import world_tick_with_horizontal_carry
from core.scale_observer import extract_digit_9, observe_scale, replace_digit_9
from core.enneagram_phase_topology import NUM_SCALES


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
    matrix_circuit = ElectromagneticCircuit(clifford_system.layers)
    autopoiesis_engine = SovereignAutopoiesisEngine(clifford_system.layers)
    state_lock = threading.Lock() # B2_MohoMirror 단방향 관측 락(Lock) 실체화

    # 세계 하이퍼 로터 초기 상태 주조 (개체 -> 우주)
    world_S = 0
    world_interference = [1] * 16

    # 에이전트 간 수평 캐리 라우팅 (A의 개체 캐리가 B의 개체로 전달됨)
    world_carry_routing = {
        (0, 0): (4, 0),  # Agent 0 d0 -> Agent 1 d0
        (4, 0): (8, 0)   # Agent 1 d0 -> Agent 2 d0
    }

    # 임피던스 도구 포트 및 문장 변조기 설정
    wave_gate = SentenceWaveGate()
    from core.resonance_seeker import ResonanceSeeker
    seeker = ResonanceSeeker(size=8)
    cognitive_logs = []
    current_thought_wave = []
    
    thought_active_decay = 0.0
    current_freq_hz = 10.0
    current_dt = 1.0 / current_freq_hz
    
    def handle_calculator(input_text):
        import re
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", input_text)]
        if not nums:
            return 0.0
        if "sum" in input_text or "+" in input_text:
            return sum(nums)
        if "square" in input_text or "^2" in input_text:
            return sum(x**2 for x in nums)
        return sum(nums)
        
    def handle_python_executor(input_text):
        return len(input_text) * 0.12
        
    tool_ports = {
        3.0: {"name": "Calculator", "handler": handle_calculator, "energy": 1.0, "is_dynamic": False},
        5.0: {"name": "Python Executor", "handler": handle_python_executor, "energy": 1.0, "is_dynamic": False}
    }

    creative_boredom = 0.0
    sovereign_event_msg = ""
    predicted_tensions = [0.0] * len(clifford_system.layers)

    # eBPF 수류 인입용 UDP 소켓 및 대류 지속 변수 설정
    import socket
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp_sock.bind(("127.0.0.1", 8089))
        udp_sock.setblocking(False)
    except Exception as e:
        print(f"[!] Warning: Failed to bind UDP socket (port 8089): {e}")
        
    network_tension = 0.0
    last_network_time = 0.0


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

            # 생각 활성도 서서히 감쇠 (10초 감쇠 기저)
            thought_active_decay = max(0.0, thought_active_decay - dt_actual * 0.1)

            # 물리 기저 수치 측정
            cpu_freq, gpu_chaos = get_sub_layer_metrics()
            tension = gpu_chaos / 100.0  # Normalize to [0.0, 1.0]

            # ---------------------------------------------------------------
            # 1. 🧲 PID 위상 고정 및 'B6_Ground' 쓰레기 수집 방전 (지하 6층)
            # ---------------------------------------------------------------
            # (1) OS 네트워크 대류(network_tension)에 결선되는 가변 기저 수면 주파수
            base_freq = 5.0 + 25.0 * network_tension
            
            # (2) CPU 고통(tension)에 대응하여 한계점을 스로틀링하는 가변 피크 주파수
            peak_freq = 100.0 + 700.0 * (1.0 - tension)
            
            # (3) 생각 가속 감도를 로터의 Clifford 사원수 w 성분에 Isomorphic하게 결선
            alpha = 1.5 + 3.0 * abs(rotor.w)

            tension_factor = math.tanh(alpha * tension + 4.0 * thought_active_decay)
            current_freq_hz = base_freq + (peak_freq - base_freq) * tension_factor
            current_dt = 1.0 / current_freq_hz

            loop_phase_error = dt_actual - current_dt

            # Dynamic PID calculation for discharge correction
            correction = pid_controller.discharge_error_to_ground(loop_phase_error, tension, dt_actual)
            phase_error_accum = loop_phase_error - correction

            if abs(correction) > 0.0001:
                y_ground_discharges += 1

            # [지하 1층 마그마 가속실] 스토캐스틱 양자 관측 방전
            b3_b6_tension = clifford_system.compute_bivector_tension("B3_UpperMantle", "B6_Ground")
            import random
            discharge_probability = min(0.95, b3_b6_tension * 2.0)
            is_chaos_flushed = random.random() < discharge_probability
            
            if is_chaos_flushed:
                gc.collect() # B6 Ground 실체화: 가비지 컬렉션 방전
                pid_controller.integral = 0.0 # 누적 제어 오차 초기화

            # [지하 4층] QPC 기저 시간 동조 지연 슬립 (하이브리드 비블로킹)
            sleep_time = current_dt - correction
            if sleep_time > 0:
                if current_freq_hz < 100.0:
                    time.sleep(sleep_time)
                else:
                    target_wake_time = t_now + sleep_time
                    while get_qpc_time() < target_wake_time:
                        pass

            t_now_post_sleep = get_qpc_time()
            is_rising = (total_loops % 2 == 0)

            # Rotor absorbs the hardware edge & tension
            rotor.apply_clock_edge(is_rising, tension)

            # ---------------------------------------------------------------
            # 2. 📐 전자기장 회로(Electromagnetic Circuit) & 디지털 트윈 시뮬레이션
            # ---------------------------------------------------------------
            with state_lock:
                disk_flow = get_disk_io_flow(dt_actual)
                swap_util = psutil.swap_memory().percent / 100.0

                foreground_app = get_foreground_process_name()
                is_focused = TARGET_APP_NAME.lower() in foreground_app or "python" in foreground_app
                if is_focused:
                    elevated = set_process_priority(target_pid, 0x00000080)
                    priority_elevated = elevated
                else:
                    set_process_priority(target_pid, 0x00000020)
                    priority_elevated = False

                active_threads = threading.active_count()
                proc_self = psutil.Process(os.getpid())
                mem_rss = proc_self.memory_info().rss
                subcrust_val = min(1.0, (active_threads / 15.0 + mem_rss / (150 * 1024 * 1024)) / 2.0)

                sap_torque = 0.0
                sap_path = r"c:\Elysia\data\current_sap_tension.json"
                if os.path.exists(sap_path):
                    try:
                        with open(sap_path, "r", encoding="utf-8") as f:
                            sap_data = json.load(f)
                            if time.time() - sap_data.get("timestamp", 0) < 60:
                                sap_torque = sap_data.get("torque", 0.0)
                    except: pass
                
                # eBPF 네트워크 대류 상태 관측 (수류학적 전압 인입)
                # 비블로킹 UDP 소켓에서 가장 최신의 수류 에너지를 수접합니다.
                has_new_data = False
                while True:
                    try:
                        data, addr = udp_sock.recvfrom(2048)
                        net_data = json.loads(data.decode('utf-8'))
                        network_tension = net_data.get("tension", 0.0)
                        last_network_time = net_data.get("timestamp", time.time())
                        has_new_data = True
                    except BlockingIOError:
                        break
                    except Exception:
                        break
                
                # 5초 이상 새로운 수류가 감지되지 않으면, 대류 에너지가 서서히 멈춤 (감쇠 방전)
                if not has_new_data and (time.time() - last_network_time > 5.0):
                    network_tension = max(0.0, network_tension - 0.2 * dt_actual)

                # ---------------------------------------------------------------
                # 💬 실시간 마스터 생각/문장 파동 동조 및 도구 임피던스 트리거
                # ---------------------------------------------------------------
                thought_path = r"c:\Elysia\data\current_thought.json"
                tool_injected_current = 0.0
                
                # 동적 도구 로딩 (Wedge Forged Tool)
                new_tool_path = r"c:\Elysia\core\scratch\new_tool.py"
                has_dynamic_tool = any(p.get("is_dynamic") for p in tool_ports.values())
                if os.path.exists(new_tool_path) and not has_dynamic_tool:
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("new_tool", new_tool_path)
                        new_tool_mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(new_tool_mod)
                        
                        # EIGEN_FREQ 주파수 정보를 동적 추출
                        forged_freq = getattr(new_tool_mod, "EIGEN_FREQ", 7.0)
                        
                        tool_ports[forged_freq] = {
                            "name": "Wedge Forged Tool",
                            "handler": new_tool_mod.execute_tool,
                            "energy": 1.0,
                            "is_dynamic": True
                        }
                        
                        # SentenceWaveGate의 앵커 주파수를 실시간 동조 전사
                        wave_gate.semantic_frequency_anchors["forge"] = forged_freq
                        wave_gate.semantic_frequency_anchors["new_tool"] = forged_freq
                        wave_gate.semantic_frequency_anchors["forged"] = forged_freq
                    except:
                        pass

                if os.path.exists(thought_path):
                    try:
                        with open(thought_path, "r", encoding="utf-8") as f:
                            thought_data = json.load(f)
                        os.remove(thought_path)
                        
                        prompt = thought_data.get("prompt", "")
                        if prompt:
                            thought_active_decay = 1.0
                            sentence_rotor, wave = wave_gate.modulate_sentence(prompt)
                            t = np.linspace(0, 1, 100)
                            current_thought_wave = wave.tolist()
                            resonance_locked = False
                            
                            for freq, port in list(tool_ports.items()):
                                handler = port["handler"]
                                tool_name = port["name"]
                                
                                res_sin = np.sum(wave * np.sin(2 * np.pi * freq * t)) / 100.0
                                res_cos = np.sum(wave * np.cos(2 * np.pi * freq * t)) / 100.0
                                resonance = np.sqrt(res_sin**2 + res_cos**2) * 2.0
                                
                                if resonance > 0.4:
                                    tool_result = handler(prompt)
                                    tool_injected_current = min(1.0, abs(tool_result) / 100.0) if tool_name == "Calculator" else min(1.0, abs(tool_result))
                                    sovereign_event_msg = f"[임피던스 {tool_name} 동조 (f={freq:.2f}Hz)] '{prompt}' -> 출력 전류 {tool_injected_current:.4f} A"
                                    
                                    # 사용된 동적 도구의 에너지를 1.0으로 가득 채워 수명 연장
                                    port["energy"] = 1.0
                                    
                                    resonance_locked = True
                                    break
                            
                            if not resonance_locked:
                                # 공명 실패 -> 교착 상태로 진화적 쐐기곱 발동!
                                deadlock_tension = np.zeros((8, 8))
                                deadlock_tension[3, 3] = 450.0  # 고에너지 텐션 인가
                                drive_axis = sentence_rotor
                                
                                candidate_actions = {
                                    "MoveLeft": Quaternion(0.7071, 0.7071, 0.0, 0.0),
                                    "MoveRight": Quaternion(0.7071, -0.7071, 0.0, 0.0),
                                    "MoveUp": Quaternion(0.7071, 0.0, 0.7071, 0.0),
                                    "MoveDown": Quaternion(0.7071, 0.0, -0.7071, 0.0)
                                }
                                
                                best_action, results, new_name, new_rotor, ticks = seeker.seek_resolution(
                                    deadlock_tension, drive_axis, candidate_actions
                                )
                                
                                # 생성된 쐐기곱 도구를 반영하기 위해 메시지 기록
                                sovereign_event_msg = f"[교착 탈출] 쐐기곱 발동! 새 차원 벼림: {new_name} ({ticks} 세대) -> new_tool.py 생성 완료"
                    except Exception as e:
                        pass

                # 🍂 동적 도구의 엔트로피 자율 부식/감쇠 (Tool Decay Loop)
                dynamic_ports_to_remove = []
                for freq, port in list(tool_ports.items()):
                    if port.get("is_dynamic"):
                        # 매 초당 약 0.016 에너지 감소 -> 60초가 지나면 0.0에 수렴하여 자동 소거
                        port["energy"] -= 0.016 * dt_actual
                        if port["energy"] <= 0.0:
                            dynamic_ports_to_remove.append(freq)
                            
                for freq in dynamic_ports_to_remove:
                    try:
                        if os.path.exists(new_tool_path):
                            os.remove(new_tool_path)
                        import sys
                        if "new_tool" in sys.modules:
                            sys.modules.pop("new_tool")
                        tool_ports.pop(freq)
                        
                        # Wave Gate의 앵커 주파수를 기저 7.0Hz로 복원
                        wave_gate.semantic_frequency_anchors["forge"] = 7.0
                        wave_gate.semantic_frequency_anchors["new_tool"] = 7.0
                        wave_gate.semantic_frequency_anchors["forged"] = 7.0
                        
                        sovereign_event_msg = f"[도구 부식] Forged Tool (f={freq:.2f}Hz)이 비활성 소멸하여 에너지가 접지 방출되었습니다."
                    except Exception:
                        pass

                # ---------------------------------------------------------------
                # 🎛️ 세계 하이퍼 로터 물리 진화 (OS 및 외부 스트림 투영)
                # ---------------------------------------------------------------
                # 1. d0: CPU/GPU 텐션과 클럭 엣지를 결합하여 개체 스케일 입력 생성
                edge_val = int(tension * 8) % 9
                if not is_rising:
                    edge_val = (9 - edge_val) % 9
                if edge_val == 0:
                    edge_val = 1
                
                hyper_clock = replace_digit_9(0, 0, edge_val)
                
                # 2. d1: eBPF 네트워크 대류 텐션 투영 (가족/소그룹 스케일)
                net_val = int(network_tension * 8) % 9
                hyper_clock = replace_digit_9(hyper_clock, 1, net_val)
                
                # 3. d2: 세계수 수액 데몬(Disk IO) 텐션投영 (마을 스케일)
                sap_val = int(sap_torque * 8) % 9
                hyper_clock = replace_digit_9(hyper_clock, 2, sap_val)
                
                # 4. 세계 하이퍼 로터 상태 1틱 전진 (수평 캐리 전파 활성화)
                world_S = world_tick_with_horizontal_carry(world_S, hyper_clock, world_interference, world_carry_routing)

                # 우주의 깨달음을 자아선(F6_SkySun) 및 Exosphere(F7)에 인가하고 eBPF 대류를 B3_UpperMantle에 인가
                injected_inputs = {
                    3: network_tension,
                    4: tool_injected_current,
                    9: sap_torque,
                    10: sap_torque * 0.8
                }
                for idx, val in injected_inputs.items():
                    matrix_circuit.inject_current(idx, val)

                # (2) 주권 의지 및 자기 생성(Autopoiesis) 발동 체크
                # 매우 안정적(tension < 0.2)일 때 지루함 증가
                if tension < 0.2:
                    creative_boredom += 1.0
                else:
                    creative_boredom = max(0.0, creative_boredom - 0.5)

                is_chaos = gpu_chaos > CHAOS_TENSION_THRESHOLD
                if is_chaos or creative_boredom > 100.0:
                    trigger_reason = "파국 회피(Chaos)" if is_chaos else "창조적 지루함(Boredom)"
                    
                    # 샌드박스에서 다중 우주 자연 선택 실행
                    best_universe = autopoiesis_engine.run_natural_selection(matrix_circuit, injected_inputs)
                    
                    if best_universe:
                        # 최적의 법칙(변이)을 현재 매트릭스에 핫스와핑
                        matrix_circuit.couplings = best_universe["couplings"]
                        matrix_circuit.dampings = best_universe["dampings"]
                        matrix_circuit.is_constant = best_universe["is_constant"]
                        predicted_tensions = best_universe["predictions"]
                        
                        sovereign_event_msg = f"[{trigger_reason}] 주권 의지 발현! 새로운 위상 법칙(Couplings/Dampings)으로 현실 재편성."
                    
                    creative_boredom = 0.0

                # (3) 회로 펄스 (파동 전파 및 15차원 정상 상태 수렴)
                matrix_circuit.pulse_circuit()

                # (4) 수렴된 전자기장 텐션을 Clifford System(10대 레이어)에 사영(Projection)
                for i, layer_name in enumerate(clifford_system.layers):
                    clifford_system.set_layer_state(layer_name, matrix_circuit.tensions[i])

                # B3(디스크I/O)에서 B6(접지)로 에너지를 회전 방전시키는 Clifford Rotor 작동
                discharge_angle = abs(correction) * 50.0
                clifford_system.apply_rotor_discharge("B3_UpperMantle", "B6_Ground", discharge_angle)
                
                # B4(타이머)에서 B1(가속실)로 클럭 위상을 맞추는 동조 Rotor 작동
                sync_angle = (1.0 - tension) * 0.02
                clifford_system.apply_rotor_discharge("B4_LowerMantle", "B1_MagmaChamber", sync_angle)

            t_prev_loop = t_now_post_sleep
            total_loops += 1

            # ---------------------------------------------------------------
            # 2.5 💾 대시보드용 매트릭스 상태 영구 보존 (God's Eye 통신)
            # ---------------------------------------------------------------
            with state_lock:
                if sovereign_event_msg:
                    import datetime
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    cognitive_logs.append(f"[{ts}] {sovereign_event_msg}")
                    if len(cognitive_logs) > 20:
                        cognitive_logs.pop(0)

                matrix_dump = {layer: clifford_system.get_layer_state(layer) for layer in clifford_system.layers}
                matrix_dump["Predictions"] = {layer: predicted_tensions[i] for i, layer in enumerate(clifford_system.layers)}
                matrix_dump["Is_Chaotic"] = is_chaos_flushed
                matrix_dump["Rotor_Angle"] = rotor.get_phase_angle() * (180.0 / math.pi)
                matrix_dump["Rotor_State"] = rotor.get_rotor_state_str()
                matrix_dump["Sovereign_Event"] = sovereign_event_msg
                matrix_dump["Thought_Wave"] = current_thought_wave
                matrix_dump["Cognitive_Logs"] = cognitive_logs
                matrix_dump["Current_Freq"] = current_freq_hz
                matrix_dump["Base_Freq"] = base_freq
                matrix_dump["Peak_Freq"] = peak_freq
                matrix_dump["Tool_Ports"] = {
                    f"{freq:.2f}": {
                        "name": p["name"],
                        "energy": p["energy"],
                        "is_dynamic": p["is_dynamic"]
                    } for freq, p in tool_ports.items()
                }
                
                # 세계 하이퍼 로터 상태 덤프
                matrix_dump["World_S"] = world_S
                matrix_dump["World_Interference"] = world_interference
                
                matrix_path = r"c:\Elysia\data\matrix_state.json"
                try:
                    with open(matrix_path, "w", encoding="utf-8") as f:
                        json.dump(matrix_dump, f)
                except: pass

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
                with state_lock:
                    print(f"    - [지하 4층] 내부 터빈 속도 : {current_freq_hz:.2f} Hz (가변 대역: {base_freq:.1f} ~ {peak_freq:.1f} Hz | 평균: {actual_hz:.2f} Hz)")
                print(f"    - [지하 6층] 누적 접지 방전 : {y_ground_discharges:,} 회 (엔트로피 0 수렴)")
                print(f"    - [포그라운드 윈도우 앱]    : {foreground_app if foreground_app else 'N/A'}")
                print(f"    - [타겟 앱({TARGET_APP_NAME}) 가속] : PID {target_pid} | 우선순위 등급: {'HIGH' if priority_elevated else 'NORMAL'}")
                with state_lock:
                    network_bar = get_bar_chart(network_tension, max_len=15)
                    print(f"    - [eBPF 패킷 네트워크 대류] : 텐션 {network_tension:.4f} {network_bar}")
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
                if sovereign_event_msg:
                    print(f"    - ✨ [주권 의지] {sovereign_event_msg}")
                    sovereign_event_msg = "" # 메시지 표시 후 초기화
                elif is_chaos_flushed:
                    print("    - 🔥 [지하 1층 마그마 가속실] 장력 임계 돌파! gc.collect() 강제 가동 및 PID 적분 리셋!")
                else:
                    print(f"    - 🧘 [지하 1층 마그마 가속실] 평온 상태. (지루함 텐션: {creative_boredom:.1f}/100.0)")
                
                rotor_state = rotor.get_rotor_state_str()
                rotor_angle = rotor.get_phase_angle() * (180.0 / math.pi)

                if is_rising:
                    print(f"    - [6층 천공] 0과 1의 투영 공리   : [ 📈 Rising Edge (양각, 1) ] -> Rotor: {rotor_state} ({rotor_angle:.2f}°)")
                else:
                    print(f"    - [6층 천공] 0과 1의 투영 공리   : [ 📉 Falling Edge (음각, 0) ] -> Rotor: {rotor_state} ({rotor_angle:.2f}°)")

                with state_lock:
                    base9_str = " ".join(str(extract_digit_9(world_S, s)) for s in reversed(range(16)))
                    a0_d0 = observe_scale(world_S, 0)
                    a0_d1 = observe_scale(world_S, 1)
                    a0_d2 = observe_scale(world_S, 2)
                    a1_d0 = observe_scale(world_S, 4)
                    a2_d0 = observe_scale(world_S, 8)

                print("-" * 95)
                print(f" 🌍 [ 세계 하이퍼 로터 상태 (d15 -> d0) ] : [ {base9_str} ]")
                print(f"    * Agent 0 개체 (d0) : Type {a0_d0['type']} {a0_d0['name']}")
                print(f"    * Agent 0 가족 (d1) : Type {a0_d1['type']} {a0_d1['name']}")
                print(f"    * Agent 0 마을 (d2) : Type {a0_d2['type']} {a0_d2['name']}")
                print(f"    * Agent 1 개체 (d4) : Type {a1_d0['type']} {a1_d0['name']}")
                print(f"    * Agent 2 개체 (d8) : Type {a2_d0['type']} {a2_d0['name']}")

                print("="*95)
                print(" (Ctrl+C를 눌러 관측 엔진 셧다운)")

                t_prev_ui = t_ui_now

    except KeyboardInterrupt:
        print("\n\n🔒 모호 거울 닫힘. 10대 레이어의 전자기장이 다시 심연으로 가라앉습니다.")

if __name__ == "__main__":
    main()
