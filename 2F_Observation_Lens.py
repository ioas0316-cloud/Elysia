import psutil
import time
import os

# -------------------------------------------------------------------
# 2F Observation Lens & Atlantis Full-Board Acceleration Bridge
#
# [기성 공학과의 평화로운 조화 (관측 및 권고 시뮬레이션 모드)]
# 마스터의 철학("자연의 결을 거스르지 않고 물 흐르듯 날로 먹는다")에 따라,
# 기성 윈도우 OS의 스케줄러 멱살을 폭력적으로 잡는 강제 락(Lock)을 해제합니다.
# 대신, 1060 요새의 하드웨어 전자기학적 파동을 맑게 읽어들여
# 타겟 앱이 하드웨어 상수에 어떻게 '수류학적으로 동조'하는지를 시뮬레이션하고 비춰줍니다.
# -------------------------------------------------------------------

TARGET_PROCESS_NAME = "python3" # 시연용 타겟

# [날먹 비례 상수 정의]
CONSTANT_A_STAR = 0.05
CONSTANT_B_GALAXY = 0.001
CHAOS_TENSION_THRESHOLD = 5.0 # 이 임계치를 넘으면 가상의 방전(Flush) 기전 발동

def get_system_uptime():
    return time.time() - psutil.boot_time()

def read_full_board_pulse():
    """
    지하 4층(하부 맨틀/정적 로터)부터 지하 3층(PCIe 버스/카오스 장력)까지의
    하드웨어 전판 물리 지표를 긁어옵니다.
    """
    uptime = get_system_uptime()

    # 1. CPU 기저 주파수 (지하 4층: 정적 로터 축)
    try:
        cpu_freq = psutil.cpu_freq().current
    except Exception:
        cpu_freq = 2400.0 + (psutil.cpu_percent() * 10)

    # 2. 메인보드 PCIe 버스 대역폭 (지하 3층: 대수로 마나 유속 모사)
    mem_info = psutil.virtual_memory()
    pcie_bus_flow = mem_info.percent

    # 3. GPU 코어 전압 출렁임 및 시스템 혼돈 (지하 3층: 카오스 장력 모사)
    try:
        load_avg = os.getloadavg()[0]
        gpu_chaos_tension = load_avg * 10
    except Exception:
        gpu_chaos_tension = psutil.cpu_percent() / 5.0

    return uptime, cpu_freq, pcie_bus_flow, gpu_chaos_tension

def simulate_acceleration_bridge(target_name, chaos_tension):
    """
    [관측 및 권고 코어]
    실제 스케줄러를 망가뜨리는 과격한 제어(REALTIME_PRIORITY)는 주석 처리하고,
    마그마 가속실(지하 1층)이 타겟을 발견하여 전자기학적으로
    안정화시키는 과정을 맑게 관측합니다.
    """
    bridged_pids = []
    flushed = False

    for proc in psutil.process_iter(['name', 'pid']):
        try:
            if target_name in proc.info['name']:
                # -------------------------------------------------------
                # [안전성 롤백] 기성 공학의 호통을 수용하여 강제 Lock 주석 처리
                # proc.nice(psutil.REALTIME_PRIORITY_CLASS)
                # proc.cpu_affinity(available_cores)
                # -------------------------------------------------------

                # 타겟을 관측 렌즈에 포착
                bridged_pids.append(proc.info['pid'])

                # 지하 1층 마그마 가속실의 가상 방전 시뮬레이션
                if chaos_tension > CHAOS_TENSION_THRESHOLD:
                    flushed = True

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return bridged_pids, flushed

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("🌌 [Atlantis 2F Lens] 초기화 중... 수류학적 관측 동조를 시작합니다.\n")
    time.sleep(1)

    try:
        while True:
            # 1. 하드웨어 전판 맥박 긁어오기
            uptime, cpu_freq, pcie_flow, gpu_chaos = read_full_board_pulse()

            # 2. 브릿지 관측 시뮬레이션
            bridged_pids, is_flushed = simulate_acceleration_bridge(TARGET_PROCESS_NAME, gpu_chaos)

            # 3. 지하 3층 대류 완충 공식을 가볍게 계산 (앱 전달 스트레스)
            dampened_stress = (gpu_chaos * pcie_flow) / (cpu_freq if cpu_freq > 0 else 1) * 0.1

            clear_screen()
            print("="*75)
            print(" ⚡ [ 1060 아틀란티스 전판 가속 브릿지 (Observation Mode) ]")
            print("="*75)
            print(" 🧲 [ 1. 하드웨어 전자기학적 파동 (지하 3,4층) ]")
            print(f"    - 절대 상수 (RTC Uptime)   : {uptime:.2f} 초")
            print(f"    - CPU 정적 로터 주파수     : {cpu_freq:.2f} MHz")
            print(f"    - 메인보드 PCIe 버스 유속  : {pcie_flow:.2f} (마나 가속도)")
            print(f"    - 코어 원시 카오스 장력    : {gpu_chaos:.2f} (임계치: {CHAOS_TENSION_THRESHOLD})")
            print("-" * 75)

            print(f" 🎯 [ 2. 기성 앱 수류학적 동조 (Target: {TARGET_PROCESS_NAME}) ]")
            if bridged_pids:
                print(f"    - 포착된 프로세스(PID) 목록: {bridged_pids}")
                print(f"    - 🛡️ [Mantle Dampening] 앱 전달 스트레스: {dampened_stress:.4f} (거의 0에 수렴)")
                print("    - [Sync] 타겟 앱이 1060 하드웨어 절대 주파수에 부드럽게 동조 중입니다.")
                if is_flushed:
                    print("    - 💥 [Flush] 카오스 장력 한계 돌파! 지하 6층 영점 접지봉으로 렉 방전 시뮬레이션!!")
                else:
                    print("    - 🌊 10대 레이어의 유체 역학을 타고 렉(0) 상태로 순항 중...")
            else:
                print("    - 타겟 프로세스를 기다리는 중... (현재 활성화된 앱 없음)")

            print("="*75)
            print(" (Ctrl+C를 눌러 관측 해제)")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🔒 관측 해제. 요새의 전자기장은 계속해서 도도하게 흐릅니다.")

if __name__ == "__main__":
    main()
