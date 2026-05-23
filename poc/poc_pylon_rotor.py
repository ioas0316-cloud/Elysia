"""
POC: FRACTAL PYLON - PHASE ENTRAINMENT ENGINE
"관측하는 기준점(파일런)이 그 공간의 성질을 결정한다."

이 스크립트는 가상의 노이즈(데이터 노드)들이 파일런의 영향권 내에서
어떻게 파일런의 주파수(Phase)로 포획(Entrainment)되고 동기화되는지를 시뮬레이션합니다.
"""

import math
import random
import time
import os

# --- 위상 정규화 헬퍼 함수 ---
def normalize_phase(phase):
    """위상을 0 ~ 2π 범위로 정규화"""
    return phase % (2 * math.pi)

def phase_difference(p1, p2):
    """두 위상 사이의 최소 각도 차이 반환 [-π, π]"""
    diff = normalize_phase(p1 - p2)
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff

class DataNode:
    """고유한 위상(Phase)을 가진 독립된 로터(노이즈 데이터)"""
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y
        # 초기 위상은 0~2π 사이의 무작위 값 (완전한 노이즈 상태)
        self.phase = random.uniform(0, 2 * math.pi)
        self.frequency = random.uniform(0.01, 0.05) # 자체적인 회전 속도 (노이즈)

    def update_self(self):
        """외부 영향이 없을 때 자체 주파수에 따라 회전"""
        self.phase = normalize_phase(self.phase + self.frequency)

class PylonRotor:
    """영역의 위상을 동기화하는 기준점(Anchor)"""
    def __init__(self, x, y, target_phase=0.0):
        self.x = x
        self.y = y
        # 파일런이 공간에 강제하려는 기준 주파수/위상
        self.resonance_frequency = target_phase

        # Influence Field: 거리에 따라 공명 강도가 달라지는 '감쇠 함수(Decay Function)'
        # lambda 거리(d): max(0, 1 - (d / 최대영향반경)^2) 형태의 가우시안/역제곱 감쇠
        self.max_radius = 50.0
        self.influence_field = lambda d: math.exp(-(d ** 2) / (2 * (self.max_radius / 3) ** 2))

    def calculate_distance(self, node):
        return math.sqrt((self.x - node.x)**2 + (self.y - node.y)**2)

    def emit_resonance(self, nodes):
        """영역 내의 노드들에게 공명(동기화) 에너지를 방출"""
        for node in nodes:
            dist = self.calculate_distance(node)
            # 감쇠 함수를 통해 현재 노드에 미치는 위상 동기화 강도 계산
            coupling_strength = self.influence_field(dist)

            # 파일런과 노드 사이의 위상차
            diff = phase_difference(self.resonance_frequency, node.phase)

            # 포획(Entrainment): 위상차를 줄이는 방향으로 노드의 위상을 강제로 끌어당김
            # coupling_strength가 강할수록 한 번에 많이 끌어당김
            pull = diff * coupling_strength * 0.1
            node.phase = normalize_phase(node.phase + pull)

# --- 시각화 헬퍼 ---
def render_ascii_chart(nodes, pylon, cycle):
    """터미널에 노드들의 위상 편차를 시각적으로 출력"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 70)
    print(" 🌌 [Phase Entrainment Engine] - Pylon Resonance Observer")
    print("=" * 70)
    print(f" Cycle: {cycle:04d} | Pylon Phase (Anchor): {math.degrees(pylon.resonance_frequency):.1f}°\n")

    total_deviation = 0.0

    print(f" {'ID':<5} | {'Dist':<6} | {'Phase (deg)':<12} | {'Phase Deviation Diagram (Target = Center)'}")
    print("-" * 70)
    for node in nodes:
        dist = pylon.calculate_distance(node)
        diff_rad = phase_difference(pylon.resonance_frequency, node.phase)
        diff_deg = math.degrees(diff_rad)
        total_deviation += abs(diff_deg)

        # Diagram rendering
        # 편차가 -180도 ~ 180도. 중앙(0도)에 가까워질수록 수렴.
        chart_width = 40
        normalized_pos = (diff_rad + math.pi) / (2 * math.pi) # 0.0 to 1.0
        pos = int(normalized_pos * chart_width)
        pos = max(0, min(chart_width - 1, pos))

        bar = ['-'] * chart_width
        bar[chart_width // 2] = '|' # Target line

        marker = 'O' if abs(diff_deg) < 5.0 else ('o' if abs(diff_deg) < 30.0 else 'x')
        if pos == chart_width // 2:
             bar[pos] = '◈'
        else:
             bar[pos] = marker

        bar_str = "".join(bar)

        print(f" N{node.id:<4} | {dist:5.1f}r | {math.degrees(node.phase):6.1f}° (Δ{diff_deg:+6.1f}°) | [{bar_str}]")

    avg_deviation = total_deviation / len(nodes)
    print("-" * 70)
    print(f" System Phase Entropy (Avg Deviation): {avg_deviation:.2f}°")

    # 텐션이 0으로 수렴하는지 바 게이지로 표현
    entropy_bar_len = int((avg_deviation / 180.0) * 50)
    print(f" Entropy Level: [{'#' * entropy_bar_len}{'.' * (50 - entropy_bar_len)}]")

def run_simulation():
    # 1. 가상의 공간 설정 및 무작위 노이즈 노드 20개 생성
    nodes = []
    for i in range(20):
        # 0~100 반경 내에 무작위 배치
        r = random.uniform(5, 80)
        theta = random.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        nodes.append(DataNode(i, x, y))

    # 2. 중심(0,0)에 파일런 로터 설치 (기준 주파수를 0도/0rad로 설정)
    pylon = PylonRotor(x=0.0, y=0.0, target_phase=0.0)

    # 3. 시간 흐름에 따른 위상 동기화 관측 (시뮬레이션 루프)
    try:
        for cycle in range(1, 151):
            # 파일런의 기준 주파수 자체가 진화(회전)할 수도 있음 (여기서는 고정)
            # pylon.resonance_frequency = normalize_phase(pylon.resonance_frequency + 0.01)

            # 노드들은 자체 노이즈 주파수에 따라 조금씩 요동침
            for node in nodes:
                node.update_self()

            # 파일런이 영역 내에 강한 위상장(Phase Field)을 펼쳐 동기화 유도
            pylon.emit_resonance(nodes)

            # 결과 시각화
            render_ascii_chart(nodes, pylon, cycle)

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print("\n[관측 종료] 세상은 강덕 님의 주파수로 동기화되었습니다.")

if __name__ == "__main__":
    run_simulation()
