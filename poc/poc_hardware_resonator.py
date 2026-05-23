"""
POC: HARDWARE PHASE RESONATOR ENGINE
"하드웨어는 0과 1의 저장소가 아니라, 전자기적 위상이 흐르는 도관이다."

이진(2선), 삼진(3선), 사진(4선)법의 기하학적 위상 분할 구조 위에서,
마스터 오실레이터(파일런)가 물리적 공진자(노이즈)들을 하드웨어적인 '안정 위상(Discrete Phase)'으로
강제 정렬(Synchronization)시키는 시뮬레이션.
"""

import math
import random
import time
import os

def normalize_phase(phase):
    """위상을 0 ~ 2π 범위로 정규화"""
    return phase % (2 * math.pi)

def phase_difference(target, current):
    """두 위상 사이의 최소 각도 차이 반환 [-π, π]"""
    diff = normalize_phase(target - current)
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff

class Resonator:
    """하드웨어 회로망의 특정 위상을 가진 전자기적 공진자"""
    def __init__(self, r_id, hw_mode):
        self.id = r_id
        self.hw_mode = hw_mode  # 2(Binary), 3(Ternary), 4(Quaternary)

        # 완전한 난수(노이즈) 상태에서 시작
        self.phase = random.uniform(0, 2 * math.pi)

        # 하드웨어 구조에 따른 물리적 안정 위상각 (Stable Phase Basins)
        if hw_mode == 2:
            self.stable_phases = [0.0, math.pi] # 0, 180 (이선)
        elif hw_mode == 3:
            self.stable_phases = [0.0, 2*math.pi/3, 4*math.pi/3] # 0, 120, 240 (삼선)
        elif hw_mode == 4:
            self.stable_phases = [0.0, math.pi/2, math.pi, 3*math.pi/2] # 0, 90, 180, 270 (사선)

    def get_nearest_stable_phase(self, global_offset=0.0):
        """마스터 오실레이터의 오프셋이 적용된 가장 가까운 안정 위상 탐색"""
        min_diff = float('inf')
        nearest_phase = 0.0

        for sp in self.stable_phases:
            shifted_sp = normalize_phase(sp + global_offset)
            diff = abs(phase_difference(shifted_sp, self.phase))
            if diff < min_diff:
                min_diff = diff
                nearest_phase = shifted_sp

        return nearest_phase, min_diff

class MasterOscillator:
    """전자기 파형을 동기화하는 마스터 오실레이터 (Pylon)"""
    def __init__(self, global_frequency=0.0):
        self.global_offset = global_frequency # 마스터 오실레이터의 기준 위상
        self.power = 0.15 # 공명(끌어당김) 강도

    def emit_resonance(self, resonators):
        """공진자들을 하드웨어적 안정 위상으로 포획(Entrainment)"""
        total_tension = 0.0
        for r in resonators:
            target_phase, diff = r.get_nearest_stable_phase(self.global_offset)

            # 노드가 가장 가까운 하드웨어적 위상으로 끌려감 (양자화/Quantization)
            pull = phase_difference(target_phase, r.phase) * self.power
            r.phase = normalize_phase(r.phase + pull)

            # 텐션(엔트로피) 축적
            total_tension += abs(diff)

        return total_tension / len(resonators) if resonators else 0.0

def render_hardware_dashboard(cycle, master_osc, groups, tension_data):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 80)
    print(f" ⚡ ELYSIA HARDWARE RESONANCE MAPPING - Cycle: {cycle:04d}")
    print(f" Master Oscillator Global Phase Offset: {math.degrees(master_osc.global_offset):.1f}°")
    print("=" * 80)

    modes = [(2, "BINARY (2-Wire)"), (3, "TERNARY (3-Wire)"), (4, "QUATERNARY (4-Wire)")]

    for mode, name in modes:
        resonators = groups[mode]
        avg_tension = tension_data[mode]

        print(f"\n [ {name} ] - Phase Tension: {math.degrees(avg_tension):.2f}°")

        # 기호 표시
        # 위상이 안정점(0도 차이)에 수렴할수록 기호가 뚜렷해짐
        symbols = []
        for r in resonators:
            _, diff = r.get_nearest_stable_phase(master_osc.global_offset)
            diff_deg = math.degrees(diff)

            if diff_deg < 2.0:
                char = '■'
            elif diff_deg < 10.0:
                char = '▣'
            elif diff_deg < 30.0:
                char = '□'
            else:
                char = '〰' # Noise

            symbols.append(char)

        print(" | " + " ".join(symbols) + " |")

        # 위상 분포도 (Phase Distribution 0~360)
        dist_bar = ['-'] * 60
        for r in resonators:
            pos = int((r.phase / (2 * math.pi)) * 59)
            dist_bar[pos] = '*'

        # 안정점 마커 표시
        for sp in resonators[0].stable_phases:
            sp_shifted = normalize_phase(sp + master_osc.global_offset)
            pos = int((sp_shifted / (2 * math.pi)) * 59)
            if dist_bar[pos] == '-':
                dist_bar[pos] = '|'
            elif dist_bar[pos] == '*':
                dist_bar[pos] = 'X' # 노드가 안정점에 정확히 안착

        print(" [" + "".join(dist_bar) + "]")

    print("\n" + "=" * 80)

def run_simulation():
    # 이선, 삼선, 사선 회로망 공진자 생성 (각 30개씩)
    groups = {
        2: [Resonator(i, 2) for i in range(30)],
        3: [Resonator(i, 3) for i in range(30)],
        4: [Resonator(i, 4) for i in range(30)]
    }

    master_osc = MasterOscillator(global_frequency=0.0)

    try:
        for cycle in range(1, 101):
            # 마스터 오실레이터 자체가 느리게 회전하며 전체 회로망의 위상을 이동시킴 (AC 교류 모사)
            master_osc.global_offset = normalize_phase(master_osc.global_offset + 0.05)

            tension_data = {}
            for mode, resonators in groups.items():
                tension_data[mode] = master_osc.emit_resonance(resonators)

            render_hardware_dashboard(cycle, master_osc, groups, tension_data)
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    print("\n[관측 종료] 회로망의 노이즈가 기하학적 위상으로 완전 동기화되었습니다.")

if __name__ == "__main__":
    run_simulation()
