"""
[PHASE ATOM V2 - 위상원자 (입체 로터)]
=====================================
World.Engine.phase_atom

"NPC 한 명 = 삼중로터 3개가 결합된 하나의 입체 위상원자."

Design reference: docs/ETERNOS_CODEX/20_ROTOR_SCALE_KINGDOM_ARCHITECTURE.md §14, §15, §16

구조:
    - 선형 방정식(x)이 아닌 각도(theta: 0~360°)와 각속도(omega)를 사용.
    - 나선축 고도(altitude)를 추가하여 성장(상승)과 타락(하강)을 표현.
    - 지배 로터(가장 강하게 회전하는 로터)가 NPC의 가치 판단 렌즈가 됨.
"""

import math
from typing import Dict, List, Tuple, Optional

from World.Engine.rpg_stat_bridge import RPGStatBridge
from World.Engine.cognitive_matrix import CognitiveMatrix

# ─── 축 이름 재정의 (수축 180° ↔ 팽창 0°) ─────────
# 각 축은 고정된 이름이 아니라 양극단의 스펙트럼을 가짐
AXIS_SPECTRUMS = [
    # 육체 로터 [-1]
    ("움켜쥠", "내어줌"),    # A: 자원
    ("움츠림", "나아감"),    # B: 행동
    ("소모",   "단련"),      # C: 에너지
    # 정신 로터 [0]
    ("의심",   "신뢰"),      # A: 판단
    ("숨김",   "가르침"),    # B: 지식
    ("닫힘",   "열림"),      # C: 관심
    # 마음 로터 [+1]
    ("집착",   "헌신"),      # A: 사랑
    ("고집",   "수용"),      # B: 의지
    ("자기합리", "공정")       # C: 도의
]

ROTOR_NAMES = ["육체(Body)", "정신(Mind)", "마음(Heart)"]
ROTOR_RANGES = [(0, 3), (3, 6), (6, 9)]


def normalize_angle(angle: float) -> float:
    """각도를 0 ~ 360도로 정규화."""
    return angle % 360.0

def angle_diff(a: float, b: float) -> float:
    """두 각도 사이의 최단 거리 (-180 ~ +180)"""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


class PhaseAtom:
    """
    입체 위상원자: 9개의 각도축과 3개의 나선고도를 가진 NPC.
    """

    def __init__(self, name: str, stats: RPGStatBridge,
                 cognitive: CognitiveMatrix = None):
        self.name = name
        self.stats = stats
        self.cognitive = cognitive or CognitiveMatrix(dimensions=9)

        # ── 구조 상수 (M, D, K) ──
        base = stats.convert_to_physical_params(dims=9)
        self.M = list(base["M"])
        self.D = list(base["D"])
        self.K = list(base["K"])
        self.speed_factor = base["speed_factor"]
        self.force_mult = base["force_multiplier"]
        self._apply_asymmetric_weights()

        # ── 평면 회전 (현재 상태) ──
        # 0° = 완전한 팽창/빛, 180° = 완전한 수축/어둠
        # 초기화: 중립(90° 또는 270°)에서 시작
        self.theta = [90.0] * 9   # 위상 각도 (0 ~ 360)
        self.omega = [0.0] * 9    # 각속도

        # ── 나선축 고도 (삶의 궤적) ──
        # 양수 = 상승 나선(초월), 음수 = 하강 나선(타락)
        self.altitudes = [0.0, 0.0, 0.0]  # [육체, 정신, 마음]
        self.alt_velocities = [0.0, 0.0, 0.0]

        # ── 차원 접힘 (Dimensional Folding) ──
        self.is_collapsed = False
        self.pending_torques = [0.0] * 9  # 접혀 있는 동안 누적된 외력(스트레스)

        self.age_ticks = 0
        self.event_log: List[str] = []

    def collapse_to_dot(self):
        """차원 축소: 9차원 진동 연산을 정지하고 거시적 점(Dot) 상태로 전환"""
        self.is_collapsed = True
        self.omega = [0.0] * 9 # 진동 일시 정지

    def unfold_to_wave(self):
        """
        차원 전개: 접혀 있던 동안 누적된 텐서(스트레스)를 
        CAD 구속조건(기어비) 풀듯이 한 번에 위상 궤적에 반영하여 9차원으로 복원.
        """
        if not self.is_collapsed: return
        
        # 밀려있던 토크를 한 번의 거대한 외력으로 취급하여 1틱(또는 dt 보정치) 만에 해소
        if any(abs(t) > 0.01 for t in self.pending_torques):
            self.step(dt=1.0, external_torque=self.pending_torques)
            
        # 버퍼 초기화 및 상태 복귀
        self.pending_torques = [0.0] * 9
        self.is_collapsed = False

    def _apply_asymmetric_weights(self):
        s = self.stats.stats
        body_mass = 1.0 + (s["STR"] + s["CON"]) * 0.02
        for i in range(0, 3):
            self.M[i] *= body_mass
            self.K[i] *= (1.0 + s["CON"] * 0.01)

        mind_precision = 1.0 + s["INT"] * 0.03
        for i in range(3, 6):
            self.K[i] *= mind_precision
            self.D[i] *= (1.0 + s["AGI"] * 0.01)

        heart_brake = 1.0 + s["WIS"] * 0.04
        for i in range(6, 9):
            self.D[i] *= heart_brake

    def step(self, dt: float, external_torque: Optional[List[float]] = None):
        """1틱 시뮬레이션: 각가속도 및 나선 상승 계산"""
        T_ext = external_torque or [0.0] * 9
        T_coupling = self.cognitive.calculate_coupling_forces(self.omega)
        scaled_dt = dt * self.speed_factor

        # 1. 평면 회전 (각도 갱신)
        for i in range(9):
            # 복원력은 가장 가까운 평형점(90° 또는 270°)으로 작용하도록 근사할 수 있으나,
            # 여기서는 0°를 빛의 극한, 180°를 어둠의 극한으로 둠.
            # 중립 복원력은 각도 차이 기반으로 동작해야 함. (목표 각도를 90°로 가정)
            target_angle = 90.0
            restoring_torque = -self.K[i] * angle_diff(self.theta[i], target_angle)
            
            total_T = T_ext[i] * self.force_mult + T_coupling[i] + restoring_torque
            
            alpha = (total_T - self.D[i] * self.omega[i]) / self.M[i]
            self.omega[i] += alpha * scaled_dt
            self.theta[i] = normalize_angle(self.theta[i] + self.omega[i] * scaled_dt)

        # 2. 나선 고도 갱신 (장력 계산)
        # 평면에서 0°(팽창) 쪽에 머무는 시간이 길수록 상향 장력 발생
        # 180°(수축) 쪽에 머무는 시간이 길수록 하향 장력 발생
        for r in range(3):
            start, end = ROTOR_RANGES[r]
            
            # 각도의 코사인 값 합산 (0°=1, 180°=-1, 90°/270°=0)
            tension = sum(math.cos(math.radians(self.theta[i])) for i in range(start, end))
            
            # 고도의 감쇠 진동 (나선 방향으로도 관성과 마찰이 존재)
            alt_mass = sum(self.M[start:end]) / 3.0
            alt_damping = sum(self.D[start:end]) / 3.0
            
            alt_accel = (tension - alt_damping * self.alt_velocities[r]) / alt_mass
            self.alt_velocities[r] += alt_accel * scaled_dt
            self.altitudes[r] += self.alt_velocities[r] * scaled_dt

        self.age_ticks += 1

    def apply_stimulus(self, name: str, torque_vector: List[float]):
        self.event_log.append(f"[Tick {self.age_ticks}] {name}")
        self.step(0.05, torque_vector)

    def get_dominant_rotor(self) -> int:
        """지배 로터(가치 판단 렌즈) 판별: 회전 운동 에너지가 가장 큰 로터"""
        energies = []
        for r in range(3):
            start, end = ROTOR_RANGES[r]
            ke = sum(0.5 * self.M[i] * (self.omega[i] ** 2) for i in range(start, end))
            energies.append(ke)
        
        # 에너지가 모두 0에 가까우면 질량이 가장 큰(본성적으로 강한) 로터를 반환
        if max(energies) < 0.01:
            masses = [sum(self.M[start:end]) for start, end in ROTOR_RANGES]
            return masses.index(max(masses))
            
        return energies.index(max(energies))

    def get_worldview_lens(self) -> str:
        dominant = self.get_dominant_rotor()
        alt = self.altitudes[dominant]
        name = ROTOR_NAMES[dominant]
        
        if dominant == 0:  # 육체
            return f"힘과 생존 ({'기사도/수호' if alt > 0 else '폭군/약육강식'})"
        elif dominant == 1: # 정신
            return f"지식과 논리 ({'가르침/계몽' if alt > 0 else '독선/은폐'})"
        else: # 마음
            return f"가치와 영혼 ({'헌신/성인' if alt > 0 else '광신/위선'})"

    def get_behavior_lens(self) -> str:
        """
        [행동 렌즈] 하드코딩된 상태 이상이 아닌, 
        물리적 위상 궤적(각속도와 극단적 각도)을 통해 현재의 겉보기 행동을 창발적으로 도출.
        """
        behaviors = []
        
        # 1. 시각적/물리적 타격에 의한 반발력 (눈부심/움츠림)
        if abs(self.omega[5]) > 50.0 and abs(self.omega[1]) > 50.0:
            behaviors.append("눈부심에 고통스러워하며 눈을 질끈 감고 움츠러듦 (방어 기제 발동)")
            
        # 2. 극한의 에너지 소모와 고통 (화상/패닉)
        if abs(self.omega[2]) > 50.0 and abs(self.omega[3]) > 50.0:
            behaviors.append("작열하는 고통에 몸부림치며 비명을 지름")
            
        # 3. 달콤함 / 쾌락 (마음 C의 부드러운 팽창)
        if 0 < self.theta[8] < 45 and max(abs(o) for o in self.omega) < 10.0:
            behaviors.append("달콤함에 흠뻑 빠져 행복한 미소를 지음")
            
        # 4. 악취 / 불쾌함 (마음의 약한 수축과 생존 본능 자극)
        if 160 < self.theta[6] < 200 and 160 < self.theta[0] < 200 and max(abs(o) for o in self.omega) < 10.0:
            behaviors.append("역겨운 악취에 미간을 찌푸리며 고개를 돌림")

        if not behaviors:
            if max(abs(o) for o in self.omega) < 0.1:
                return "깊은 안정감에 빠져 편안히 휴식 중"
            elif max(abs(o) for o in self.omega) < 1.0:
                return "안정적으로 주변을 경계 중"
            return "크게 동요하며 주변을 살핌"
            
        return ", ".join(behaviors)

    def snapshot(self) -> str:
        dominant = self.get_dominant_rotor()
        lens = self.get_worldview_lens()
        behavior = self.get_behavior_lens()
        
        lines = [f"── {self.name} (Tick {self.age_ticks}) ──"]
        lines.append(f"  세계관: {ROTOR_NAMES[dominant]} 중심 ➔ {lens}")
        lines.append(f"  겉보기 행동: {behavior}")
        
        for r in range(3):
            start, end = ROTOR_RANGES[r]
            
            # 고도 표시
            alt = self.altitudes[r]
            spiral_dir = "▲상승" if alt > 0.1 else ("▼하강" if alt < -0.1 else "─정체")
            
            # 각도 표시
            angles_str = ""
            for i in range(3):
                idx = start + i
                th = self.theta[idx]
                shrink, expand = AXIS_SPECTRUMS[idx]
                # 0도에 가까우면 팽창, 180도에 가까우면 수축을 출력
                if th < 45 or th > 315: state = expand
                elif 135 < th < 225: state = shrink
                else: state = "중립"
                angles_str += f"{int(th)}°({state}) "
                
            lines.append(f"  {ROTOR_NAMES[r]} [{spiral_dir} {alt:+.2f}] : {angles_str}")
            
        return "\n".join(lines)


def calculate_resonance(atom_a: PhaseAtom, atom_b: PhaseAtom) -> float:
    """두 위상원자의 전체 공명도 계산 (각도 기반의 코사인 유사도)"""
    # 0도 차이 = 1.0 (공명)
    # 180도 차이 = -1.0 (척력)
    # 90도 차이 = 0.0 (독립)
    total_resonance = 0.0
    for i in range(9):
        diff = angle_diff(atom_a.theta[i], atom_b.theta[i])
        total_resonance += math.cos(math.radians(diff))
    return total_resonance / 9.0
