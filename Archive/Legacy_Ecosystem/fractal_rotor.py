"""
FRACTAL ROTOR — 다차원 프랙탈 가변 로터 스케일 (Multi-dimensional Fractal Rotor)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"0(우주) 안에 1~9가 들어있고, 10, 100은 0 자체가 경계로 확장된 것이다."

섭리 (Providence):
  억지스러운 규칙(if-else, mass 변수)은 존재하지 않는다.
  오직 위상(Phase)과 진폭(Amplitude)을 가진 파동의 중첩만이 존재한다.
  - 질량과 중력: 파동이 중첩되어 진폭이 커지면 그것이 곧 질량이며, 
                 큰 진폭은 작은 파동을 자연스럽게 자신 쪽으로 끌어당긴다(중력).
  - 삼진법(-1, 0, 1): 파동이 더해질 때 보강간섭(+1), 상쇄간섭(-1), 직교장력(0)으로 
                      수학적 섭리에 의해 스스로 발현된다.
"""

import cmath
import math
import psutil
import time

# ═══════════════════════════════════════════════════════════
#  1. FRACTAL ROTOR SCALE
# ═══════════════════════════════════════════════════════════

class FractalRotor:
    def __init__(self, id_tag, level=0, num_children=0):
        self.id = id_tag
        self.level = level
        
        # 4축 상태를 복소수(Complex)로 관리
        # 진폭(Amplitude) = 정보의 밀도/질량
        # 편각(Phase) = 위상
        self.states = [cmath.rect(0.5, 0.0) for _ in range(4)]
        self.free = [True] * 4  # 의지에 의한 잠금/열림

        self.BREATH = 0.05
        self.LOCK_THRESHOLD = 3.0  # 진폭(질량)이 커지면 자연스럽게 상수(기지)로 잠김

        # 프랙탈 포함 관계 (0 안에 1~9가 들어있다)
        self.sub_rotors = []
        if num_children > 0:
            for i in range(num_children):
                self.sub_rotors.append(FractalRotor(f"{id_tag}.{i+1}", level + 1, 0))

    # ── 의지 (Will) ──

    def will(self):
        """진폭(질량/밀도)이 커져 확고해진 축은 기지로 잠그고, 미지는 연다."""
        for i in range(4):
            amp = abs(self.states[i])
            self.free[i] = amp < self.LOCK_THRESHOLD
        
        for sub in self.sub_rotors:
            sub.will()

    # ── 공명 (Resonance): 순수한 파동 간섭 ──

    def resonate(self, incoming_states):
        """
        어떠한 인위적 규칙도 없다. 
        단지 두 복소 파동이 더해지는(간섭하는) 자연의 섭리뿐이다.
        """
        # 1. 자신(현재 스케일)의 공명
        for i in range(4):
            if not self.free[i]:
                continue
                
            my_wave = self.states[i]
            their_wave = incoming_states[i]
            
            # 파동의 중첩: 이것 하나로 인력, 배척, 장력, 중력이 모두 계산됨
            # - 같은 방향이면 진폭(질량)이 더해짐 (+1, 보강간섭)
            # - 반대 방향이면 진폭이 깎임 (-1, 상쇄간섭)
            # - their_wave의 진폭이 압도적으로 크면 방향이 그쪽으로 확 쏠림 (중력)
            interference = my_wave + (their_wave * self.BREATH)
            
            # 진폭이 0이 되는 특이점 방지 및 최대/최소 밀도 제약 (물리적 한계)
            new_amp = max(0.1, min(10.0, abs(interference)))
            new_phase = cmath.phase(interference)
            
            self.states[i] = cmath.rect(new_amp, new_phase)

        if not self.sub_rotors:
            return

        # 2. 하강 (Descending): 상위 로터의 거대한 파동이 하위 로터의 우주가 된다
        omega_phase = cmath.phase(self.states[3])
        num_sub = len(self.sub_rotors)
        
        for i, sub in enumerate(self.sub_rotors):
            topology_twist = cmath.rect(1.0, omega_phase * (i / num_sub))
            
            # 상위 로터의 상태(진폭=질량, 편각=방향)가 고스란히 하위로 쏟아진다
            child_universe = [
                self.states[0],                  
                self.states[1],                  
                self.states[2] * topology_twist, # 방향성만 토폴로지에 따라 비틀림
                self.states[3]                   
            ]
            sub.resonate(child_universe)

        # 3. 내부 공명 (Lateral): 하위 로터들끼리의 간섭
        sub_states_snapshot = [list(sub.states) for sub in self.sub_rotors]
        for i, sub in enumerate(self.sub_rotors):
            nxt = (i + 1) % num_sub
            sub.resonate(sub_states_snapshot[nxt])

        # 4. 상승 (Ascending): 하위 로터들의 전체 궤적이 융합되어 상위를 밀어올림
        combined_ascent = [0j] * 4
        for i in range(4):
            superposition = sum(sub.states[i] for sub in self.sub_rotors)
            combined_ascent[i] = superposition / num_sub
        
        # 하위에서 올라온 거대한 파동과 상위 로터가 다시 간섭
        for i in range(4):
            if self.free[i]:
                asc_interference = self.states[i] + (combined_ascent[i] * self.BREATH)
                self.states[i] = cmath.rect(max(0.1, min(10.0, abs(asc_interference))), cmath.phase(asc_interference))


# ═══════════════════════════════════════════════════════════
#  2. DISPLAY UTILITIES
# ═══════════════════════════════════════════════════════════

def amp_bar(amp, width=5):
    level = min(1.0, max(0.0, amp / 3.0)) # 3.0 이상이면 꽉 참
    filled = int(level * width)
    return '█' * filled + '░' * (width - filled)

def display_rotor(rotor, prefix=""):
    is_open = sum(1 for f in rotor.free if f)
    glyph = '◈' if is_open == 0 else ('·' if is_open == 1 else ('─' if is_open == 2 else ('△' if is_open == 3 else '□')))
    code = ''.join('1' if f else '0' for f in rotor.free)
    
    axes_str = ''
    total_mass = 0
    for i in range(4):
        mark = '◇' if rotor.free[i] else '◆'
        amp = abs(rotor.states[i])
        total_mass += amp
        axes_str += f"{mark}{amp_bar(amp)} "
        
    print(f"│ {prefix}{rotor.id:<5} [{code}]{glyph} (M:{total_mass:4.1f}) │ {axes_str}│")
    
    for i, sub in enumerate(rotor.sub_rotors):
        branch = "├─" if i < len(rotor.sub_rotors)-1 else "└─"
        display_rotor(sub, prefix + branch)


# ═══════════════════════════════════════════════════════════
#  3. MAIN OBSERVER
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("━" * 58)
    print("  FRACTAL ROTOR — 다차원 프랙탈 가변 로터 스케일")
    print("  \"인위적 규칙은 없다. 오직 파동의 간섭이라는 자연 섭리뿐이다.\"")
    print("━" * 58)
    
    universe_rotor = FractalRotor("L0", level=0, num_children=3)
    
    cycle = 0
    try:
        for _ in range(15):
            cycle += 1
            
            # 하드웨어의 미세한 맥박을 복소 파동(진폭 고정 1.0)으로 변환
            cpu = psutil.cpu_percent(interval=0.05)
            mem = psutil.virtual_memory().percent
            t = time.time()
            
            hw_wave = [
                cmath.rect(1.0, (cpu / 100.0) * 2 * math.pi),
                cmath.rect(1.0, (mem / 100.0) * 2 * math.pi),
                cmath.rect(1.0, ((math.sin(t * 0.1) + 1) / 2) * 2 * math.pi),
                cmath.rect(1.0, ((math.sin(t * 2.7) + 1) / 2) * 2 * math.pi),
            ]
            
            universe_rotor.will()
            universe_rotor.resonate(hw_wave)
            
            omega = cmath.phase(universe_rotor.states[3]) % (2 * math.pi)
            if omega < 0.5 or omega > 5.7:
                topology = "Y-수렴 (초점화)"
            elif 1.8 < omega < 2.4 or 3.9 < omega < 4.5:
                topology = "Δ-순환 (안정 오비탈)"
            else:
                topology = "↕-삼중나선 (비틀림 진행)"
            
            print(f"┌─ Cycle {cycle:05d} ──────────────────────────────────────────────┐")
            print(f"│  우주 토포스 (L0.ω) : {topology:<25}  │")
            print(f"├────────────────────────────────────────────────────────┤")
            display_rotor(universe_rotor, " ")
            print(f"└────────────────────────────────────────────────────────┘\n")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n  ✧ 관측 종료.")
