"""
FRACTAL ROTOR — 관측 기반 공명 엔진 (Observation-Based Resonance Engine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"모든 로터는 상위 로터의 회전 필드 안에서 정의되며, 관측이란 분리된 좌표의 확인이 아니라,
두 로터가 동일한 위상에 닿아 공명하는 '위상 동기화(Phase-Locking)의 결과'이다."

섭리 (Providence):
  절대적인 좌표(x, y, z)는 존재하지 않는다.
  오직 상위 로터(상수축)와의 위상차(Phase Offset)와 장력(Tension)만이 존재할 뿐이다.
  데이터는 계산되는 것이 아니라, 위상차가 0으로 수렴하여 공명할 때 물리적으로 '발현'된다.
"""

import math
import psutil
import time

# ═══════════════════════════════════════════════════════════
#  1. PURE PHASE DYNAMICS (순수 위상 역학 엔진)
# ═══════════════════════════════════════════════════════════

def normalize_phase(phase):
    """위상을 [-π, π] 범위로 정규화"""
    phase = phase % (2 * math.pi)
    if phase > math.pi:
        phase -= 2 * math.pi
    return phase

class Rotor:
    def __init__(self, id_tag, level=0, parent=None, initial_phase_offset=0.0):
        self.id = id_tag
        self.level = level
        self.parent = parent
        
        # 절대 좌표가 아니라 부모(상위 로터)를 기준으로 한 '위상차'
        self.phase_offset = normalize_phase(initial_phase_offset)

        # 부모-자식 간의 물리적 장력 (스트레스/에너지)
        self.tension = 0.0

        # 시스템이 스스로 붕괴(재정렬)하기 위한 한계 장력
        self.TENSION_LIMIT = math.pi / 2  # 위상차가 90도를 넘어가면 불안정성 극대화

        # 하위 로터들 (가변축)
        self.sub_rotors = []

        # (최상위 로터 전용) 우주 전체의 기준 주파수
        self._global_phase = 0.0

    @property
    def current_phase(self):
        """
        자신의 절대 위상은 저장되지 않는다.
        관측(호출) 시점에 상위 로터의 위상에 자신의 위상차를 더해 '발현'된다.
        """
        if self.parent:
            return normalize_phase(self.parent.current_phase + self.phase_offset)
        else:
            return normalize_phase(self._global_phase)

    def attach_child(self, child_rotor):
        self.sub_rotors.append(child_rotor)

    # ── 관측과 공명 (Observation & Phase-Locking) ──

    def observe(self, global_rotation_delta=0.0):
        """
        상위 로터가 회전 관측을 시작하면, 하위 로터들의 위상이
        상위 주파수로 '끌려들어오는(Pulling)' 물리 연산 (Phase-Locking)
        """
        if self.parent is None:
            # 최상위 우주 로터는 시간/에너지에 의해 스스로 회전
            self._global_phase += global_rotation_delta
            self._global_phase = normalize_phase(self._global_phase)

        for sub in self.sub_rotors:
            # 1. 하위 로터가 부모의 회전에 반응하여 위상 동기화 시도
            # 장력이 높을수록(위상차가 클수록) 강하게 끌어당김
            pull_force = math.sin(sub.phase_offset) * 0.1
            sub.phase_offset -= pull_force
            sub.phase_offset = normalize_phase(sub.phase_offset)

            # 2. 남은 위상차를 장력(Tension)으로 축적
            # 위상이 동기화(0)에 가까워지면 장력은 해소됨
            sub.tension = abs(sub.phase_offset)

            # 3. 조율의 물리학 (붕괴 및 재정렬)
            if sub.tension > sub.TENSION_LIMIT:
                sub.collapse_and_realign()

            # 하위 로터들도 자신의 자식들을 관측
            sub.observe()

    def collapse_and_realign(self):
        """
        최소 에너지 상태로의 붕괴.
        장력이 한계를 넘으면 에너지를 해소하기 위해 위상을 완전히 비틀거나 재배치한다.
        """
        # 상위 로터로의 에너지 역류 (부모의 축을 흔듦)
        if self.parent:
            # 내가 겪는 장력만큼 부모의 위상차에 타격을 줌 (역인과)
            impact = self.tension * 0.5
            if self.phase_offset > 0:
                self.parent.phase_offset -= impact
            else:
                self.parent.phase_offset += impact
            self.parent.phase_offset = normalize_phase(self.parent.phase_offset)

        # 자신은 에너지를 방출하고 가장 안정적인 위상(0 또는 π)으로 붕괴
        # (새로운 궤도에 안착)
        if abs(self.phase_offset) > math.pi / 2:
            self.phase_offset = math.pi if self.phase_offset > 0 else -math.pi
        else:
            self.phase_offset = 0.0
            
        self.tension = 0.0


# ═══════════════════════════════════════════════════════════
#  2. DISPLAY UTILITIES
# ═══════════════════════════════════════════════════════════

def phase_bar(phase, tension):
    """위상(Phase)을 시각적인 파동으로, 장력(Tension)을 기호로 표현"""
    # phase is in [-π, π]
    normalized = (phase + math.pi) / (2 * math.pi) # 0.0 ~ 1.0
    width = 20
    pos = int(normalized * width)
    
    # 파동 모양
    bar = ['-'] * width

    # 장력이 크면 파동이 격렬해짐
    marker = 'O' if tension < 0.5 else ('X' if tension < 1.0 else '⚡')
    if 0 <= pos < width:
        bar[pos] = marker
        
    return "".join(bar)

def display_rotors(rotor, prefix=""):
    phase_deg = math.degrees(rotor.current_phase)
    offset_deg = math.degrees(rotor.phase_offset)

    # 상태 출력 (절대 위상은 계산된 결과일 뿐, 실제 저장된 것은 offset과 tension)
    bar = phase_bar(rotor.phase_offset, rotor.tension)

    if rotor.parent is None:
        print(f"│ {prefix}{rotor.id:<5} [CORE] Phase: {phase_deg:6.1f}° | Global: {bar} │")
    else:
        print(f"│ {prefix}{rotor.id:<5} [T: {rotor.tension:4.2f}] Offset: {offset_deg:6.1f}° | Wave: [{bar}] │")
    
    for i, sub in enumerate(rotor.sub_rotors):
        branch = "├─" if i < len(rotor.sub_rotors)-1 else "└─"
        display_rotors(sub, prefix + branch)


# ═══════════════════════════════════════════════════════════
#  3. MAIN OBSERVER (우주적 합창)
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("━" * 65)
    print("  FRACTAL ROTOR — 관측 기반 공명 엔진 (Phase-Locking)")
    print("  \"좌표는 존재하지 않는다. 오직 위상차와 장력의 조율만이 있을 뿐이다.\"")
    print("━" * 65)
    
    # 1. 뼈대 생성 (좌표가 아닌 관계성만 정의)
    universe = Rotor("L0_C", level=0)
    
    # 3개의 1차 가변축 (120도씩 위상차를 가짐)
    r1 = Rotor("L1_A", level=1, parent=universe, initial_phase_offset=0.0)
    r2 = Rotor("L1_B", level=1, parent=universe, initial_phase_offset=2*math.pi/3)
    r3 = Rotor("L1_C", level=1, parent=universe, initial_phase_offset=-2*math.pi/3)

    universe.attach_child(r1)
    universe.attach_child(r2)
    universe.attach_child(r3)

    # 1차 가변축에 다시 종속된 2차 가변축들
    for parent_node in [r1, r2, r3]:
        parent_node.attach_child(Rotor(f"{parent_node.id}.1", level=2, parent=parent_node, initial_phase_offset=math.pi/4))
        parent_node.attach_child(Rotor(f"{parent_node.id}.2", level=2, parent=parent_node, initial_phase_offset=-math.pi/4))

    cycle = 0
    try:
        while True:
            cycle += 1
            
            # 외부 세계(하드웨어)의 맥동을 '회전력(Rotation Delta)'으로 치환
            cpu = psutil.cpu_percent(interval=0.1)
            # CPU 부하가 높을수록 우주의 회전이 빨라지고 위상이 뒤틀림
            rotation_delta = (cpu / 100.0) * math.pi * 0.5
            
            # 최상위 로터가 회전하며 하위 로터들을 관측 (공명 유도)
            # 이때 하위 로터들은 상위 위상에 동기화(Phase-Lock)되려 하며,
            # 저항하는 위상차는 장력(Tension)으로 쌓여 붕괴/재정렬을 일으킴.
            universe.observe(global_rotation_delta=rotation_delta)
            
            # 외부 노이즈(의도치 않은 섭동) 주입:
            # 시스템이 완벽히 정지하지 않고 끊임없이 조율하도록 방해
            if cycle % 10 == 0:
                r1.phase_offset += math.pi / 3  # A축을 강제로 비틀어버림
            
            print(f"┌─ Resonance Cycle {cycle:05d} [Rotation Delta: +{math.degrees(rotation_delta):5.1f}°] ─┐")
            display_rotors(universe, " ")
            print(f"└────────────────────────────────────────────────────────┘\n")
            
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("\n  ✧ 관측 및 공명 종료.")
