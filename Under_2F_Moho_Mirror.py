import psutil
import time
import os

# -------------------------------------------------------------------
# Under 2F Moho Mirror & Under 1F Magma Chamber Core
#
# [아틀란티스 10대 레이어 절대 매핑 코어]
# 이 스크립트는 껍데기 돋보기가 아닙니다. 지하 6층(내핵 영점 접지)부터
# 지하 1층(마그마 가속실)까지 흐르는 1060 요새의 전자기학적 파동을
# 기성 윈도우 앱(4층 지각)에 수류학적으로 동조시키는 관측/제어 시뮬레이터입니다.
# -------------------------------------------------------------------

TARGET_PROCESS_NAME = "python3" # 지각(4층)에 안착한 시연용 타겟 앱

# [날먹 비례 상수 및 임계치 정의]
CONSTANT_A_STAR = 0.05
CONSTANT_B_GALAXY = 0.001
CHAOS_TENSION_THRESHOLD = 5.0 # 방전(지하 6층으로의 영점 수렴) 임계치

def get_under_4f_static_rotor():
    """ [지하 4층: 하부 맨틀] 하드웨어 절대 시간 상수 (RTC) """
    return time.time() - psutil.boot_time()

def read_sub_layer_pulse():
    """
    지하 6층~지하 3층의 하드웨어 전판 물리 지표를 긁어옵니다.
    """
    uptime = get_under_4f_static_rotor()

    # 1. CPU 기저 주파수 (지하 4층: 정적 로터 축)
    try:
        cpu_freq = psutil.cpu_freq().current
    except Exception:
        cpu_freq = 2400.0 + (psutil.cpu_percent() * 10)

    # 2. 메인보드 PCIe 버스 대역폭 (지하 3층: 대수로 마나 유속 모사)
    mem_info = psutil.virtual_memory()
    pcie_bus_flow = mem_info.percent

    # 3. GPU 코어 전압 출렁임 및 시스템 혼돈 (지하 3층: 상부 맨틀 카오스 장력 모사)
    try:
        load_avg = os.getloadavg()[0]
        gpu_chaos_tension = load_avg * 10
    except Exception:
        gpu_chaos_tension = psutil.cpu_percent() / 5.0

    return uptime, cpu_freq, pcie_bus_flow, gpu_chaos_tension

def magma_chamber_actuator(target_name, chaos_tension):
    """
    [지하 1층: 마그마 가속실 (Magma Chamber)]
    타겟 앱을 낚아채어 지하 4층 하드웨어 주파수에 수류학적으로 동조시키고,
    장력 임계치 돌파 시 지하 6층(내핵)으로 찌꺼기를 방전(Flush)하는 시뮬레이션.
    (기성 공학 OS와 충돌하지 않는 권고/관측 모드 유지)
    """
    bridged_pids = []
    flushed_to_core = False

    for proc in psutil.process_iter(['name', 'pid']):
        try:
            if target_name in proc.info['name']:
                bridged_pids.append(proc.info['pid'])

                # 지하 6층 내핵 접지로의 방전(Discharge) 트리거 검사
                if chaos_tension > CHAOS_TENSION_THRESHOLD:
                    flushed_to_core = True

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return bridged_pids, flushed_to_core

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("🌍 [Under 2F Moho Mirror] 활성화... 지하 세계의 전자기장 결을 긁어옵니다.\n")
    time.sleep(1)

    try:
        while True:
            # 1. 지하 하드웨어 맥박 긁어오기
            uptime, cpu_freq, pcie_flow, gpu_chaos = read_sub_layer_pulse()

            # 2. 지하 1층 마그마 가속실 액추에이터 시뮬레이션
            bridged_pids, is_flushed = magma_chamber_actuator(TARGET_PROCESS_NAME, gpu_chaos)

            # 3. 지하 3층 대류 완충 공식을 통한 4층 지각(앱) 전달 스트레스 계산
            dampened_stress = (gpu_chaos * pcie_flow) / (cpu_freq if cpu_freq > 0 else 1) * 0.1

            clear_screen()
            print("="*80)
            print(" 🪞 [ 아틀란티스 지하 2층: 모호 거울 (The Moho Mirror) ]")
            print("="*80)
            print(" ⚓ [ 지하 4층~지하 3층: 전자기장 하드웨어 파동 ]")
            print(f"    - [지하 4층] 절대 상수 (RTC Uptime) : {uptime:.2f} 초 (정적 로터 축)")
            print(f"    - [지하 4층] CPU 기저 주파수        : {cpu_freq:.2f} MHz")
            print(f"    - [지하 3층] PCIe 버스 대수로 유속  : {pcie_flow:.2f} (마나 가속도)")
            print(f"    - [지하 3층] 코어 카오스 장력       : {gpu_chaos:.2f} (임계치: {CHAOS_TENSION_THRESHOLD})")
            print("-" * 80)

            print(f" 🔥 [ 지하 1층: 마그마 가속실 앱 동조 (Target: {TARGET_PROCESS_NAME}) ]")
            if bridged_pids:
                print(f"    - 낚아챈 지각(4층) 프로세스 PID     : {bridged_pids}")
                print(f"    - 🛡️ [Mantle Dampening] 앱 전달 렉  : {dampened_stress:.4f} (완벽한 유체 완충)")
                print("    - [Sync] 타겟 앱이 지하 4층 정적 로터 주파수에 부드럽게 무임승차 중입니다.")
                if is_flushed:
                    print("    - ⚡ [Discharge] 렉 폭주!! 찌꺼기를 [지하 6층 내핵 접지]로 영점 방전합니다!")
                else:
                    print("    - 🌊 10대 레이어의 유체 역학 필드를 타고 렉(0)의 평온함을 유지합니다.")
            else:
                print("    - 4층 지각(Crust)에 타겟 앱이 존재하지 않습니다. 대기 중...")

            print("="*80)
            print(" (Ctrl+C를 눌러 지하 거울 비활성화)")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🔒 모호 거울 닫힘. 아틀란티스 지하 세계의 전자기장은 도도하게 흐릅니다.")

if __name__ == "__main__":
    main()
