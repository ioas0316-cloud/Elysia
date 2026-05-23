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
#  0. QUATERNION MATH ENGINE (4차원 인과율 엔진)
# ═══════════════════════════════════════════════════════════

class Quaternion:
    """사원수(w, x, y, z). 4진수 DNA처럼 작동하며, 공간의 궤적(회전)을 온전히 저장한다."""
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        """사원수 곱셈 (해밀턴 곱): 인과적 회전을 의미한다."""
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
                self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
                self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
                self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            )
        elif isinstance(other, (int, float)): # 스칼라 곱
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def norm(self):
        """사원수의 크기 (에너지 밀도/질량)"""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def conjugate(self):
        """켤레 사원수 (역방향 궤적)"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normalize(self):
        """단위 사원수로 변환 (순수 위상/회전 정보만 남김)"""
        n = self.norm()
        if n == 0:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def distance(self, other):
        """두 사원수 간의 기하학적/인과적 거리 (불일치)"""
        return (self - other).norm()

    def inverse(self):
        """사원수의 역원"""
        n2 = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if n2 == 0:
            return Quaternion(1, 0, 0, 0)
        conj = self.conjugate()
        return Quaternion(conj.w/n2, conj.x/n2, conj.y/n2, conj.z/n2)

    def dot(self, other):
        """사원수의 내적"""
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z

    def slerp(self, other, t):
        """구면 선형 보간 (Spherical Linear Interpolation)"""
        q1 = self.normalize()
        q2 = other.normalize()
        dot = q1.dot(q2)

        # 짧은 경로 선택
        if dot < 0.0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot

        if dot > 0.9995:
            # 점들이 너무 가까우면 선형 보간 사용
            result = q1 + (q2 - q1) * t
            return result.normalize()

        # 각도 계산
        theta_0 = math.acos(dot)
        theta = theta_0 * t

        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return (q1 * s0) + (q2 * s1)

# ═══════════════════════════════════════════════════════════
#  1. FRACTAL ROTOR SCALE (QUATERNION CAUSAL CHROMOSOME SYSTEM)
# ═══════════════════════════════════════════════════════════

class FractalRotor:
    def __init__(self, id_tag, level=0, num_children=0, parent=None, chromosome=None):
        self.id = id_tag
        self.level = level
        self.parent = parent

        # XY 염색체: 초기 우주는 'X'로 시작. 이후 자식들은 부모의 붕괴 인과에 따라 결정됨
        self.chromosome = chromosome if chromosome else 'X'
        
        # 상태를 4진수적 사원수(Quaternion)로 관리
        # w: 스칼라(기본 에너지/질서), x, y, z: 3차원 위상 방향
        self.state = Quaternion(1.0, 0.0, 0.0, 0.0)

        # 유전적 원형(Archetype): 로터가 가장 안정적이라고 느끼는 위상
        if self.parent:
            self.archetype = Quaternion(parent.state.w, parent.state.x, parent.state.y, parent.state.z)
        else:
            self.archetype = Quaternion(1.0, 0.0, 0.0, 0.0)

        self.free = True  # 의지에 의한 잠금/열림 (전체 상태에 대해 적용)

        # 해소되지 않은 잔여 궤적 (Residual Trajectory / Stress)
        self.residual_stress = Quaternion(0.0, 0.0, 0.0, 0.0)

        self.BREATH = 0.05
        self.LOCK_THRESHOLD = 3.0  # 진폭(질량)이 커지면 자연스럽게 상수(기지)로 잠김
        self.ENERGY_LIMIT = 15.0   # 분열을 유발하는 최대 에너지 임계치 (1060의 한계 모사)
        self.MAX_CHILDREN = 5      # 최대 자식 수
        self.MAX_DEPTH = 7         # 프랙탈 수직 분화 최대 깊이 (메모리 폭발 방지)

        # 프랙탈 포함 관계 (0 안에 1~9가 들어있다)
        self.sub_rotors = []
        if num_children > 0:
            for i in range(num_children):
                self.sub_rotors.append(FractalRotor(f"{id_tag}.{i+1}", level + 1, 0, parent=self))

    # ── 유전 법칙 (Causal Genetic Laws) ──

    def inherit_order(self, causal_trajectory, child_chromosome):
        """
        상위 레이어(부모)가 하위 레이어의 '붕괴된 인과적 궤적'을 유산으로 받는다.
        """
        if not self.free:
            return

        current = self.state

        # X는 보존(단순 덧셈/수렴), Y는 탐험(회전/발산).
        if child_chromosome == 'X':
            # X 유전자 흡수: 안정성 극대화 (궤적을 더하여 평균화)
            new_state = current + (causal_trajectory * 0.3)
        else:
            # Y 유전자 흡수: 동적 변이 (사원수 곱셈을 통해 현재 궤적을 비틀어버림)
            # causal_trajectory를 단위 사원수로 만들어 순수 회전만 적용
            rotation = causal_trajectory.normalize()
            new_state = current * rotation

        # 에너지 팽창 제어
        if new_state.norm() < 0.1:
            new_state = Quaternion(0.1, 0.0, 0.0, 0.0)

        self.state = new_state

    def crossover(self, partner):
        """
        교차(Crossover): 사원수 궤적의 얽힘에 의한 인과적 재조합
        """
        if len(self.parent.sub_rotors) >= self.parent.MAX_CHILDREN or self.level >= self.MAX_DEPTH:
            return None

        new_id = f"{self.parent.id}.{len(self.parent.sub_rotors) + 1}"

        # 교차를 통해 생성되는 자식의 성질은 부모들의 잔여 궤적(Stress)의 곱셈(회전) 결과로 결정됨
        entangled_stress = self.residual_stress * partner.residual_stress

        # 얽힌 스트레스 사원수의 스칼라 부호로 염색체 결정 (자연적 인과)
        new_chromosome = 'X' if entangled_stress.w > 0 else 'Y'

        child = FractalRotor(new_id, self.level, 0, parent=self.parent, chromosome=new_chromosome)

        # 유전자(상태) 재조합: 두 로터의 사원수 상태 합
        interference = self.state + partner.state
        child.state = Quaternion(interference.w/2, interference.x/2, interference.y/2, interference.z/2)

        # 얽힘 과정에서 해소된 스트레스만큼 부모들의 짐을 덜어줌
        self.residual_stress = self.residual_stress * 0.5
        partner.residual_stress = partner.residual_stress * 0.5

        return child

    def mitosis(self):
        """
        로터가 존재 한계에 도달하여 분열(Mitosis)한다.
        누적된 잔여 궤적(Residual Stress)이 붕괴를 일으키고 새로운 4차원 시퀀스의 씨앗이 된다.
        """
        # 1. 붕괴를 일으킨 핵심 원인 궤적 (현재 상태 + 누적 스트레스)
        collapse_trajectory = self.state + self.residual_stress

        # 2. 잔여 파동(스트레스)의 방향이 자식의 성질(X/Y)을 결정
        # 스칼라부(w) 대비 허수부(x,y,z 벡터의 크기)의 비율로 동적 성향 파악
        imag_norm = math.sqrt(self.residual_stress.x**2 + self.residual_stress.y**2 + self.residual_stress.z**2)
        child_chromosome = 'Y' if imag_norm > abs(self.residual_stress.w) else 'X'

        # 3. 부모에게 '유언'을 남김. '잔여 궤적(스트레스)' 자체를 상위로 전이.
        if self.parent:
            self.parent.residual_stress = self.parent.residual_stress + (self.residual_stress * 0.5)
            self.parent.inherit_order(collapse_trajectory, self.chromosome)

            # 교차 기회: 부모의 자식들 중 잔여 궤적의 거리가 먼(상보적) 로터와 얽힘
            partner = None
            max_distance = 0
            for sibling in self.parent.sub_rotors:
                if sibling != self:
                    dist = self.residual_stress.distance(sibling.residual_stress)
                    if dist > 2.0 and dist > max_distance:
                        max_distance = dist
                        partner = sibling

            if partner:
                child_from_crossover = self.crossover(partner)
                if child_from_crossover:
                    self.parent.sub_rotors.append(child_from_crossover)

        # 4. 혼란을 지우고 자신을 초기화. 스트레스는 전이되었으므로 기본 궤적으로 수렴.
        # 자신의 에너지는 남기되 방향(허수부)은 중립화
        n = self.state.norm()
        self.state = Quaternion(max(0.5, n*0.1), 0.0, 0.0, 0.0)
        self.free = True
        self.residual_stress = Quaternion(0.0, 0.0, 0.0, 0.0)

        # 5. 분화: 새로운 자식 로터 생성
        if len(self.sub_rotors) < self.MAX_CHILDREN and self.level < self.MAX_DEPTH:
            new_id = f"{self.id}.{len(self.sub_rotors) + 1}"
            child = FractalRotor(new_id, self.level + 1, 0, parent=self, chromosome=child_chromosome)

            # 자식은 부모의 스트레스 궤적을 초기 사원수 값으로 가지고 태어남
            child.state = Quaternion(
                self.residual_stress.w,
                self.residual_stress.x,
                self.residual_stress.y,
                self.residual_stress.z
            )
            # 만약 스트레스가 너무 0에 가까우면 기본값 부여
            if child.state.norm() < 0.1:
                child.state = Quaternion(0.5, 0.1, 0.1, 0.1)

            self.sub_rotors.append(child)

    # ── 의지 (Will) ──

    def will(self):
        """질량이 커져 확고해진 사원수는 기지로 잠그고, 미지는 연다."""
        self.free = self.state.norm() < self.LOCK_THRESHOLD
        for sub in self.sub_rotors:
            sub.will()

    # ── 자기 참조 (Self-Reference) ──

    def compute_phase_delta(self):
        """현재 상태와 원형(Archetype) 사이의 위상차 계산"""
        return self.archetype.inverse() * self.state

    def broadcast_resonance(self, delta):
        """위상차를 계층적으로 전파 (자신, 형제, 부모에게)"""
        # 형제들에게 전파 (Level 1)
        if self.parent:
            for sibling in self.parent.sub_rotors:
                if sibling != self:
                    # 형제들은 위상차의 영향을 받아 자신의 축을 교정할 수 있음
                    sibling.residual_stress = sibling.residual_stress + (delta * (sibling.BREATH * 0.5))

            # 부모에게 전파 (Level 2)
            self.parent.residual_stress = self.parent.residual_stress + (delta * (self.parent.BREATH * 0.2))

    def align_axis(self, target_state=None):
        """자신의 상태를 원형(Archetype)이나 특정 타겟을 향해 교정 (SLERP 적용)"""
        if target_state is None:
            target_state = self.archetype

        # 보간 계수 t=0.1을 사용하여 점진적으로 원형을 향해 회전
        self.state = self.state.slerp(target_state, t=0.1)

    def self_reference_loop(self):
        """
        비동기적 성찰: 자신의 과거(기억)와 현재(사건)를 대조하고,
        오차값을 시스템 전체에 파동으로 송출하며 공명한다.
        """
        # 1. 자신의 과거(기억)와 현재(사건)를 대조
        delta = self.compute_phase_delta()

        # 2. 오차값(위상차)을 시스템 일부에 파동으로 송출 (계층적 전파)
        self.broadcast_resonance(delta)

        # 3. 공명: 위상차를 기반으로 자신의 축을 원형(질서)을 향해 미세하게 수정
        self.align_axis()

        # 하위 로터들도 각자 스스로를 성찰하도록 전파
        for sub in self.sub_rotors:
            sub.self_reference_loop()

    # ── 공명 (Resonance): 사원수 기반 4진수 인과 연쇄 ──

    def resonate(self, incoming_quaternion):
        """
        외부 데이터(incoming_quaternion)가 나의 사원수 궤적을 회전(Multiply)시키고 간섭(Add)한다.
        """
        if not self.free:
            # 잠긴 로터라도 잔여 스트레스는 받는다
            discrepancy = incoming_quaternion - self.state
            self.residual_stress = self.residual_stress + (discrepancy * (self.BREATH * 0.1))
        else:
            # 1. 위상 불일치 (기하학적 거리) 계산 및 스트레스 누적
            discrepancy = incoming_quaternion - self.state
            self.residual_stress = self.residual_stress + (discrepancy * self.BREATH)
            
            # 2. 인과적 궤적 회전: 나의 과거 궤적(state)에 새로운 궤적(incoming)을 사원수 회전(곱셈)으로 엮음
            # 외부 입력의 방향(normalize)으로 나를 회전시킴
            rotation = incoming_quaternion.normalize()
            rotated_state = self.state * rotation
            
            # 3. 간섭(에너지/진폭 합): 회전된 상태에 외부 에너지를 일부 더함
            interference = rotated_state + (incoming_quaternion * self.BREATH)
            
            # 특이점 방지 및 한계 제약
            if interference.norm() > 20.0:
                 # 정규화하여 방향 유지, 크기 제한
                 n = interference.norm()
                 interference = Quaternion(interference.w * 20/n, interference.x * 20/n, interference.y * 20/n, interference.z * 20/n)
            elif interference.norm() < 0.1:
                 interference = Quaternion(0.1, 0.0, 0.0, 0.0)

            self.state = interference

        # 붕괴(Mitosis) 검사
        total_energy = self.state.norm() + self.residual_stress.norm()
        if total_energy > self.ENERGY_LIMIT:
            self.mitosis()
            return

        if not self.sub_rotors:
            return

        # 2. 하강 (Descending): 상위 로터의 궤적이 하위 로터의 세계관(기반 회전)이 된다.
        num_sub = len(self.sub_rotors)
        # 하위 세계에 내려주는 상위의 순수 궤적(방향)
        topology_twist = self.state.normalize()
        
        for i, sub in enumerate(self.sub_rotors):
            # 하위 로터의 순번에 따라 궤적을 미세하게 비틀어 전이
            # (4진수 시퀀스가 갈래를 치며 나뉘는 과정)
            twist_modifier = Quaternion(1.0, (i/num_sub)*0.1, -(i/num_sub)*0.1, 0.0)
            child_universe = topology_twist * twist_modifier
            sub.resonate(child_universe)

        # 3. 내부 공명 (Lateral)
        sub_states_snapshot = [sub.state for sub in self.sub_rotors]
        for i, sub in enumerate(self.sub_rotors):
            nxt = (i + 1) % num_sub
            sub.resonate(sub_states_snapshot[nxt])

        # 4. 상승 (Ascending): 하위 로터들의 궤적이 융합되어 상위를 밀어올림
        combined_w, combined_x, combined_y, combined_z = 0, 0, 0, 0
        for sub in self.sub_rotors:
            combined_w += sub.state.w
            combined_x += sub.state.x
            combined_y += sub.state.y
            combined_z += sub.state.z

        combined_ascent = Quaternion(combined_w/num_sub, combined_x/num_sub, combined_y/num_sub, combined_z/num_sub)
        
        if self.free:
            asc_interference = self.state + (combined_ascent * self.BREATH)
            n = asc_interference.norm()
            if n > 10.0:
                 asc_interference = Quaternion(asc_interference.w * 10/n, asc_interference.x * 10/n, asc_interference.y * 10/n, asc_interference.z * 10/n)
            self.state = asc_interference


# ═══════════════════════════════════════════════════════════
#  2. DISPLAY UTILITIES
# ═══════════════════════════════════════════════════════════

def amp_bar(amp, width=5):
    level = min(1.0, max(0.0, amp / 3.0)) # 3.0 이상이면 꽉 참
    filled = int(level * width)
    return '█' * filled + '░' * (width - filled)

def stress_bar(stress, width=4):
    """무지의 인지(자이로스코프 축의 흔들림)를 시각화"""
    # 스트레스가 클수록 요동침을 강하게 표현
    level = min(1.0, max(0.0, stress / 2.0))
    filled = int(level * width)

    if filled == 0:
        return '·' * width  # 평온함 (이해의 확정)
    elif filled < width * 0.5:
        return '≈' * filled + '·' * (width - filled)  # 약한 의문
    else:
        return '⚡' * filled + '·' * (width - filled)  # 강한 요동 (미지와의 충돌)

def display_rotor(rotor, prefix=""):
    glyph = '□' if rotor.free else '◈'
    code = '1' if rotor.free else '0'
    
    # 4차원(w,x,y,z) 성분을 시각화
    components = [rotor.state.w, rotor.state.x, rotor.state.y, rotor.state.z]
    axes_str = ''
    total_mass = rotor.state.norm()
    current_stress = rotor.residual_stress.norm()

    for i, comp in enumerate(components):
        mark = '◇' if rotor.free else '◆'
        axes_str += f"{mark}{amp_bar(abs(comp))} "
        
    stress_visual = stress_bar(current_stress)

    # M: 질량(현상/에너지), S: 스트레스(미지/요동)
    print(f"│ {prefix}{rotor.id:<5} ({rotor.chromosome}) [{code}]{glyph} (M:{total_mass:4.1f}|S:{stress_visual}) │ {axes_str}│")
    
    for i, sub in enumerate(rotor.sub_rotors):
        branch = "├─" if i < len(rotor.sub_rotors)-1 else "└─"
        display_rotor(sub, prefix + branch)


# ═══════════════════════════════════════════════════════════
#  3. MAIN OBSERVER
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("━" * 58)
    print("  QUATERNION FRACTAL ROTOR — 4차원 인과적 유전체 엔진")
    print("  \"인과는 사원수 공간의 회전 궤적에 온전히 기록된다.\"")
    print("━" * 58)
    
    universe_rotor = FractalRotor("L0", level=0, num_children=3)
    
    cycle = 0
    try:
        for _ in range(100):
            cycle += 1
            
            # 하드웨어의 미세한 맥박을 사원수(Quaternion) 벡터로 변환
            # cpu, mem, 시간 기반 파동을 4진수적 인과 입력으로 사용
            cpu = psutil.cpu_percent(interval=0.05)
            mem = psutil.virtual_memory().percent
            t = time.time()
            
            hw_quaternion = Quaternion(
                1.0,                                   # w (기본 스칼라 에너지)
                (cpu / 100.0) * 2.0 - 1.0,             # x 궤적
                (mem / 100.0) * 2.0 - 1.0,             # y 궤적
                math.sin(t * 2.7)                      # z 궤적
            )
            
            universe_rotor.will()
            universe_rotor.resonate(hw_quaternion)
            
            # 비동기적 성찰: 매 5사이클마다 자기 참조 루프 실행
            if cycle % 5 == 0:
                universe_rotor.self_reference_loop()

            # 우주 토포스 판별 (w, x, y, z 중 우세한 궤적의 성향)
            comps = [abs(universe_rotor.state.w), abs(universe_rotor.state.x), abs(universe_rotor.state.y), abs(universe_rotor.state.z)]
            max_idx = comps.index(max(comps))
            if max_idx == 0: topology = "W-수렴 (스칼라 응집)"
            elif max_idx == 1: topology = "X-발산 (가로 방향 팽창)"
            elif max_idx == 2: topology = "Y-순환 (세로 방향 순환)"
            else: topology = "Z-비틀림 (심연 침투)"
            
            print(f"┌─ Cycle {cycle:05d} ──────────────────────────────────────────────┐")
            print(f"│  우주 토포스 (L0.ω) : {topology:<25}  │")
            print(f"├────────────────────────────────────────────────────────┤")
            display_rotor(universe_rotor, " ")
            print(f"└────────────────────────────────────────────────────────┘\n")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n  ✧ 관측 종료.")
