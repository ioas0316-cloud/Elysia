"""
Elysia Core Fractal Rotor Logic
===============================
Implements the continuous phase-locking physical equations:
- Phase offset relative to parent rotor
- Pulling force for phase synchronization
- Mechanical tension and collapse/realignment thresholds
"""

import math

def normalize_phase(phase: float) -> float:
    """Normalize phase angle to the range [-pi, pi]."""
    phase = phase % (2.0 * math.pi)
    if phase > math.pi:
        phase -= 2.0 * math.pi
    return phase

class Rotor:
    """
    Rotor representing a phase coordinate in a hierarchical scale.
    """
    def __init__(self, id_tag: str, level: int = 0, parent: 'Rotor' = None, initial_phase_offset: float = 0.0):
        self.id = id_tag
        self.level = level
        self.parent = parent
        self.phase_offset = normalize_phase(initial_phase_offset)
        self.tension = 0.0
        self.sub_rotors = []
        self._global_phase = 0.0
        self.coupling_map = {}  # Hebbian 시냅스 결선 강도 맵: {(id_i, id_j): K_ij}
        self.plasticity_mode = "normal"  # "normal", "frozen", "melted"

        # Dynamic dimensional state variables
        self.active_axes = 3
        self.MAX_AXES = 8
        self.MIN_AXES = 1
        self.stable_ticks = 0

    def bifurcate(self):
        """Expands active axes (Dimension Split) when local tension is too high."""
        if self.active_axes < self.MAX_AXES:
            self.active_axes += 1
            # Distribute shock: reduce current phase offset as energy flows into new dimension
            self.phase_offset = normalize_phase(self.phase_offset * 0.5)
            self.tension = abs(self.phase_offset)
            self.stable_ticks = 0

    def compress(self):
        """Locks unused dimensions and reduces active axes when long-term stability is detected."""
        if self.active_axes > self.MIN_AXES:
            self.active_axes -= 1
            self.stable_ticks = 0

    @property
    def tension_limit(self) -> float:
        """Dynamically scales tension limit based on rotor hierarchical level."""
        return (math.pi / 2.0) / (self.level + 1.0)

    @property
    def current_phase(self) -> float:
        """Dynamically resolves absolute phase by summing up the hierarchy."""
        if self.parent:
            return normalize_phase(self.parent.current_phase + self.phase_offset)
        return normalize_phase(self._global_phase)

    def attach_child(self, child_rotor: 'Rotor'):
        self.sub_rotors.append(child_rotor)

    def fuse_stable_children(self):
        """
        자식 로터 중 stable_ticks가 높은 로터(특히 위성 등)를 찾아
        그들의 위상 정보를 부모의 phase_offset에 중첩(Superposition)시킨 후
        자식 목록에서 삭제하여 '기억의 중력 압축(망각과 통찰)'을 수행합니다.
        """
        stable_subs = [sub for sub in self.sub_rotors if sub.stable_ticks >= 20]
        for sub in stable_subs:
            # Rotorization 지침: 위상의 간섭 중첩(Resonance Induction)
            # 부모의 오프셋을 자식 오프셋의 가중치만큼 회전 변환시킴
            fusion_weight = 0.15 / (self.level + 1.0)
            self.phase_offset = normalize_phase(self.phase_offset + sub.phase_offset * fusion_weight)
            # 자식 로터에 쌓였던 텐션도 일부 부모의 텐션으로 전이시킴
            self.tension = min(math.pi, self.tension + sub.tension * 0.1)
            
            # 하위 sub_rotors 삭제 (망각)
            self.sub_rotors.remove(sub)

    def observe(self, global_rotation_delta: float = 0.0):
        """
        Pulling force is exerted by parent and sibling rotors (Hebbian Kuramoto).
        Tension builds up in case of high phase differences, triggering bifurcation or collapse.
        """
        if self.parent is None:
            self._global_phase = normalize_phase(self._global_phase + global_rotation_delta)

        # 부모의 plasticity_mode를 자식에게도 전파하여 계층 전체 동조화
        if self.parent:
            self.plasticity_mode = self.parent.plasticity_mode

        # 1. 형제 로터들 간의 Hebbian 가소성 (시냅스 결선 갱신)
        n_subs = len(self.sub_rotors)
        for i in range(n_subs):
            for j in range(i + 1, n_subs):
                sub_i = self.sub_rotors[i]
                sub_j = self.sub_rotors[j]
                
                diff = sub_i.current_phase - sub_j.current_phase
                alignment = math.cos(diff)
                
                key = (sub_i.id, sub_j.id) if sub_i.id < sub_j.id else (sub_j.id, sub_i.id)
                current_K = self.coupling_map.get(key, 0.1)
                
                if self.plasticity_mode == "frozen":
                    # 동결(Freeze): 어떠한 결선 변경도 일어나지 않음
                    continue
                elif self.plasticity_mode == "melted":
                    # 교란(Melt): 무작위 요동 및 대폭적인 학습/자연 감쇄
                    import random
                    eta = 0.1     # 초가속 학습
                    gamma = 0.03  # 급격한 감쇄
                    noise = random.uniform(-0.1, 0.1)
                    delta_K = eta * max(-0.5, alignment) + noise
                    new_K = max(0.01, min(2.0, current_K + delta_K - gamma))
                else:
                    # 일반 Hebbian: 정렬되면 인력 상승, 역정렬 시 감쇄
                    eta = 0.02   # 가소성 학습률
                    gamma = 0.005 # 자연 감쇄 (엔트로피 부식)
                    delta_K = eta * max(-0.3, alignment)
                    new_K = max(0.01, min(2.0, current_K + delta_K - gamma))
                
                self.coupling_map[key] = new_K

        # 2. 개별 로터 물리 관측 및 위상 고정 (Kuramoto Entrainment)
        for sub in list(self.sub_rotors):
            # (1) 부모가 자식을 당기는 힘
            alignment = math.cos(sub.phase_offset)
            pull_strength = 0.05 + 0.15 * max(0.0, alignment)
            pull_force = math.sin(sub.phase_offset) * pull_strength
            
            # (2) 형제 로터들 간의 Hebbian 인력 작용
            brother_pull = 0.0
            for other in self.sub_rotors:
                if other.id == sub.id:
                    continue
                key = (sub.id, other.id) if sub.id < other.id else (other.id, sub.id)
                K_ij = self.coupling_map.get(key, 0.1)
                
                diff = other.current_phase - sub.current_phase
                brother_pull += math.sin(diff) * K_ij * 0.03 # 형제 간 위상 동조 결선력 인가
                
            sub.phase_offset = normalize_phase(sub.phase_offset - pull_force + brother_pull)
            sub.tension = abs(sub.phase_offset)

            # Relieve stress / Dimensional control
            if sub.tension > sub.tension_limit:
                if sub.active_axes < sub.MAX_AXES:
                    sub.bifurcate()
                else:
                    sub.collapse_and_realign()
            else:
                if sub.tension < sub.tension_limit * 0.2:
                    sub.stable_ticks += 1
                    if sub.stable_ticks >= 5:
                        sub.compress()
                else:
                    sub.stable_ticks = 0

            # Recursively observe descendants
            sub.observe()

        # 관측 후 안정을 찾은 자식 노드들을 부모 로터에 융합 (중력 압축)
        self.fuse_stable_children()

    def collapse_and_realign(self):
        """
        Releases accumulated stress energy into the parent and collapses
        to the nearest stable state (0 or pi).
        """
        if self.parent:
            impact = self.tension * 0.5
            if self.phase_offset > 0:
                self.parent.phase_offset = normalize_phase(self.parent.phase_offset - impact)
            else:
                self.parent.phase_offset = normalize_phase(self.parent.phase_offset + impact)

        # Drop to stable attractor point (0 or pi)
        if abs(self.phase_offset) > math.pi / 2.0:
            self.phase_offset = math.pi if self.phase_offset > 0 else -math.pi
        else:
            self.phase_offset = 0.0
            
        self.tension = 0.0

def phase_bar(phase: float, tension: float) -> str:
    normalized = (phase + math.pi) / (2 * math.pi)
    width = 20
    pos = int(normalized * width)
    bar = ['-'] * width
    marker = 'O' if tension < 0.5 else ('X' if tension < 1.0 else '⚡')
    if 0 <= pos < width:
        bar[pos] = marker
    return "".join(bar)

def display_rotors(rotor: Rotor, prefix=""):
    phase_deg = math.degrees(rotor.current_phase)
    offset_deg = math.degrees(rotor.phase_offset)
    bar = phase_bar(rotor.phase_offset, rotor.tension)

    if rotor.parent is None:
        print(f"│ {prefix}{rotor.id:<5} [CORE] Phase: {phase_deg:6.1f}° | Global: {bar} │")
    else:
        print(f"│ {prefix}{rotor.id:<5} [T: {rotor.tension:4.2f}] Offset: {offset_deg:6.1f}° | Wave: [{bar}] │")
    
    for i, sub in enumerate(rotor.sub_rotors):
        branch = "├─" if i < len(rotor.sub_rotors)-1 else "└─"
        display_rotors(sub, prefix + branch)
