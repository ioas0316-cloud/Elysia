"""
POC: HARDWARE PHASE MONITOR & FAULT DETECTION
"시스템의 에러는 논리적 오류가 아니라, 위상 동기화의 붕괴(Phase Collapse)다."

이 시뮬레이터는 로터 회로망이 정상적으로 동기화되어 흐르다가,
특정 구간에 '단락(Short)'이 발생했을 때 에러 로그가 아닌
물리적 '위상 편차(Tension)'의 폭증으로 오류 구간을 감지해내는 과정을 시각화합니다.
"""

import math
import random
import time
import os

def normalize_phase(phase):
    return phase % (2 * math.pi)

def phase_difference(target, current):
    diff = normalize_phase(target - current)
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff

class PhaseMonitorRotor:
    """전자기적 회로망의 센서이자 위상 조율기"""
    def __init__(self, node_id, hw_mode, position_idx):
        self.id = node_id
        self.hw_mode = hw_mode # 2, 3, 4 wire
        self.position = position_idx # 회로상의 위치 (0, 1, 2, ... )

        # 초기에는 마스터 오실레이터(0.0)와 완벽히 동기화되어 있다고 가정
        self.phase = 0.0
        self.tension = 0.0
        self.is_short_circuited = False

        if hw_mode == 2:
            self.stable_phases = [0.0, math.pi]
        elif hw_mode == 3:
            self.stable_phases = [0.0, 2*math.pi/3, 4*math.pi/3]
        elif hw_mode == 4:
            self.stable_phases = [0.0, math.pi/2, math.pi, 3*math.pi/2]

    def get_nearest_stable_phase(self, target_offset):
        min_diff = float('inf')
        nearest = 0.0
        for sp in self.stable_phases:
            shifted_sp = normalize_phase(sp + target_offset)
            diff = abs(phase_difference(shifted_sp, self.phase))
            if diff < min_diff:
                min_diff = diff
                nearest = shifted_sp
        return nearest

    def update(self, master_phase):
        """마스터 위상과의 동기화 및 단락 처리"""
        if self.is_short_circuited:
            # 단락된 회로는 무작위 노이즈를 뿜어냄 (위상 붕괴)
            self.phase = normalize_phase(self.phase + random.uniform(-1.0, 1.0))
        else:
            # 정상 회로는 마스터 위상에 안정적으로 수렴
            target = self.get_nearest_stable_phase(master_phase)
            pull = phase_difference(target, self.phase) * 0.3
            self.phase = normalize_phase(self.phase + pull)

        # 텐션(위상 붕괴도) 측정: 주변/목표 위상과의 차이
        target_stable = self.get_nearest_stable_phase(master_phase)
        self.tension = abs(math.degrees(phase_difference(target_stable, self.phase)))

def render_diagnostic_dashboard(cycle, master_phase, networks):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 80)
    print(f" 🩺 ELYSIA SOMATIC DIAGNOSTICS - Phase Collapse Monitor (Cycle: {cycle:04d})")
    print(f" Master Oscillator Flow (Heartbeat): {math.degrees(master_phase):.1f}°")
    print("=" * 80)

    modes = [(2, "BINARY PATH"), (3, "TERNARY PATH"), (4, "QUATERNARY PATH")]

    for mode, name in modes:
        rotors = networks[mode]
        print(f"\n [ {name} ] Circuit Continuity")

        # 1. 토폴로지 렌더링 (단락 구간 시각화)
        circuit_line = ""
        for r in rotors:
            if r.tension > 60.0:
                circuit_line += "[⚡CRASH⚡]-" # 심각한 위상 붕괴
            elif r.tension > 20.0:
                circuit_line += "[〰WARN〰]-" # 위상 불안정
            else:
                circuit_line += "[  OK  ]-" # 정상 공명
        print(" " + circuit_line[:-1])

        # 2. 각 로터별 물리적 진동 편차 (Tension) 출력
        tension_line = ""
        for r in rotors:
            if r.tension > 60.0:
                tension_line += f"  {r.tension:05.1f}°   "
            else:
                tension_line += f"  {r.tension:04.1f}°    "
        print(" " + tension_line)

    print("\n" + "=" * 80)
    print(" * 에러 감지 원리: 연산 오류(Logic Error)가 아니라 파동 텐션(Phase Tension)의 증가로")
    print("   시스템의 물리적 '상처(단락)' 위치를 특정합니다.")

def run_simulation():
    # 2선, 3선, 4선 회로망 (각각 10개의 직렬 로터 구간)
    networks = {
        2: [PhaseMonitorRotor(i, 2, i) for i in range(8)],
        3: [PhaseMonitorRotor(i, 3, i) for i in range(8)],
        4: [PhaseMonitorRotor(i, 4, i) for i in range(8)]
    }

    master_phase = 0.0

    try:
        for cycle in range(1, 101):
            master_phase = normalize_phase(master_phase + 0.1) # AC 교류 파형 진행

            # 30사이클 쯤에 TERNARY 회로 4번 구간에 물리적 단락 발생
            if cycle == 30:
                networks[3][4].is_short_circuited = True

            # 60사이클 쯤에 BINARY 회로 2번 구간에 물리적 단락 발생
            if cycle == 60:
                networks[2][2].is_short_circuited = True

            # 85사이클 쯤에 TERNARY 회로 단락 복구 (자가 치유 모사)
            if cycle == 85:
                networks[3][4].is_short_circuited = False

            # 위상 업데이트 및 텐션 계산
            for mode, rotors in networks.items():
                for r in rotors:
                    r.update(master_phase)

            render_diagnostic_dashboard(cycle, master_phase, networks)
            time.sleep(0.15)

    except KeyboardInterrupt:
        pass

    print("\n[진단 종료] 물리적 위상 편차 모니터링이 완료되었습니다.")

if __name__ == "__main__":
    run_simulation()
