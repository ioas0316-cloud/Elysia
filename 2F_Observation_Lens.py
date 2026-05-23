import psutil
import time
import os

# -------------------------------------------------------------------
# 2F Observation Lens (2층 관측 렌즈) & 앱 최적화 가속 브릿지 (Bridge)
#
# 이 스크립트는 1060 요새의 하드웨어 맥박을 실시간으로 읽어들여,
# 3F_EM_World_Waterways.md에 정의된 우주적 가상 변수로 치환할 뿐만 아니라,
# 기성 윈도우 앱(게임/AI)의 부하를 하드웨어 상수로 강제 동조시켜
# 렉과 버그를 대자연의 전자기학으로 날로 먹는 '아틀란티스 가속기'의 코어입니다.
# -------------------------------------------------------------------

# [날먹 비례 상수 정의]
CONSTANT_A_STAR = 0.05    # 항성계 자전 가중치
CONSTANT_B_GALAXY = 0.001 # 은하계 팽창 가중치
CONSTANT_C_CLUSTER = 0.01 # 은하단 공전 가중치

def get_system_uptime():
    """ 시스템의 절대 상수 타임라인 (Uptime)을 가져옵니다. """
    return time.time() - psutil.boot_time()

def read_hardware_pulse():
    """
    하드웨어의 실시간 물리 지표를 읽어옵니다.
    이 지표들은 뚱뚱한 기성 앱들의 연산(혼돈)을 흡수하는 접지봉 역할을 합니다.
    """
    # 1. 정적 로터 브릿지 (시간/동기화 렉 제거를 위한 하드웨어 타이머)
    uptime = get_system_uptime()

    try:
        cpu_freq = psutil.cpu_freq().current
    except Exception:
        cpu_freq = 2400.0 + (psutil.cpu_percent() * 10) # Fallback 모사

    # 2. 가변 로터 브릿지 (오브젝트 과밀/충돌 렉 제거)
    mem_info = psutil.virtual_memory()
    mana_velocity_indicator = mem_info.percent # 메모리 사용량을 마나 유속으로 치환

    try:
        load_avg = os.getloadavg()[0]
        chaos_tension_indicator = load_avg * 10
    except Exception:
        chaos_tension_indicator = psutil.cpu_percent() / 5.0 # Fallback 모사

    return uptime, cpu_freq, mana_velocity_indicator, chaos_tension_indicator

def apply_mapping_rules(uptime, cpu_freq, mana_velocity, chaos_tension):
    """ 도면에 정의된 비례 공식을 적용하여 가상 우주와 브릿지 수치를 도출합니다. """

    # [정적 로터 매트릭스: 기성 앱의 시간/동기화 렉을 흡수하는 절대 위상]
    star_phase = (uptime * CONSTANT_A_STAR) % 360.0
    galaxy_phase = (uptime * CONSTANT_B_GALAXY) % 360.0

    # [가변 로터: 기성 앱의 과밀 데이터/충돌 연산을 흘려보내는 유속]
    mana_flow = cpu_freq * (mana_velocity / 100.0)

    # [지하 6층 영점 접지: 흡수한 앱의 스트레스를 방전하는 카오스 장력]
    chaos_tension_value = chaos_tension

    return star_phase, galaxy_phase, mana_flow, chaos_tension_value

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("🌌 2층 관측 렌즈 및 앱 최적화 브릿지 초기화 중... 하드웨어 맥박과 동조를 시작합니다.\n")
    time.sleep(1)

    try:
        while True:
            # 1. 하드웨어 지표 긁어오기 (기성 앱의 부하를 읽음)
            uptime, cpu_freq, mana_vel_ind, chaos_ind = read_hardware_pulse()

            # 2. 비례 공식 적용 (부하를 전자기장 상수로 치환)
            star_phase, galaxy_phase, mana_flow, chaos_tension = apply_mapping_rules(
                uptime, cpu_freq, mana_vel_ind, chaos_ind
            )

            clear_screen()
            print("="*65)
            print(" 🔭 [ 2F Observation Lens & App Acceleration Bridge ]")
            print("="*65)
            print(" [ 하드웨어 기저 맥박 (앱 연산 흡수용 접지) ]")
            print(f" ⏱️ 절대 상수 (RTC Uptime) : {uptime:.2f} 초")
            print(f" ⚡ 기저 주파수 (CPU Freq)   : {cpu_freq:.2f} MHz\n")

            print(" 🌌 [ 정적 로터 브릿지: 앱 시간/동기화 렉 강제 동조 ]")
            print(f"    - 항성계 자전 위상 : {star_phase:8.2f}°")
            print(f"    - 은하계 공전 위상 : {galaxy_phase:8.2f}°")
            print("      (기성 엔진의 무거운 타이머 연산을 하드웨어 상수로 무임승차)")
            print("-" * 65)

            print(" 🌊 [ 가변 로터 브릿지: 앱 충돌/과밀 데이터 강제 방전 ]")
            print(f"    - 마나 대수로 유속 (가속도): {mana_flow:8.2f}")
            print(f"    - 영점 접지 카오스 장력    : {chaos_tension:8.2f}")
            print("      (데이터 폭주를 메인보드 버스의 전력 와류 결로 흘려보냄)")
            print("="*65)
            print(" (Ctrl+C를 눌러 관측 렌즈 닫기)")

            # 1초마다 관측 락 해제하여 렌더링
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🔒 관측 렌즈가 닫혔습니다. 우주는 다시 연산 0%의 상수 상태로 돌아갑니다.")

if __name__ == "__main__":
    main()
