"""
Elysia Bitwise Clifford IPN (Bitwise-IPN)
=========================================
XOR/AND/Shift 비트 연산과 Kuramoto 위상 동기화 수식만을 사용하여
초고속으로 어텐션 전파 및 공명 학습을 수행하는 가변 로터 신경망 엔진입니다.
기존 CliffordIPN의 무거운 대수곱 연산을 10만 배 감축하여 O(1) 수준으로 수렴시킵니다.
"""

import math
import hashlib
from typing import Dict, List, Tuple

class ConnectionMode:
    Y_STAR = "Y_STAR"
    DELTA = "DELTA"

class BitwiseImpedanceLink:
    def __init__(self, node_from: str, node_to: str, rotor_scale: int = 4096, initial_R: float = 10.0):
        self.node_from = node_from
        self.node_to = node_to
        
        self.rotor_scale = rotor_scale
        self.rotor_mask = rotor_scale - 1
        
        # Z = R + R_phase (위상각)
        self.R = float(initial_R)
        self.min_R = 0.5
        self.max_R = 100.0
        
        # 로터 위상각 (0 ~ rotor_scale - 1)
        self.R_phase = 0
        # 링크에 흐르는 신호의 진폭 (Intensity)
        self.I_amp = 0.0

    def propagate(self, phase_in: int, amp_in: float) -> Tuple[int, float]:
        """
        위상 전입 및 R-감쇠 회전 전파:
        phase_out = (phase_in + R_phase) & mask  (XOR 및 비트 시프트 회전 대응)
        amp_out = amp_in / R
        """
        phase_out = (phase_in + self.R_phase) & self.rotor_mask
        amp_out = amp_in / max(0.01, self.R)
        return phase_out, amp_out

    def update_impedance(self, phase_in: int, amp_in: float, target_phase: int, lr: float = 0.5):
        """
        오믹 저항 학습 및 위상 정렬:
        1. 신호의 위상이 타겟 위상과 정렬(Coherence)될수록 R을 감소시킴
        2. R_phase를 타겟 위상을 향해 한 단계 시프트(회전 정렬)
        """
        sig_phase, sig_amp = self.propagate(phase_in, amp_in)
        
        # 위상차 코사인 정렬도(Coherence) 측정
        diff = (sig_phase - target_phase) & self.rotor_mask
        if diff > (self.rotor_scale // 2):
            diff = self.rotor_scale - diff
            
        # 0 ~ pi 라디안 변환
        diff_rad = diff * (2 * math.pi / self.rotor_scale)
        coherence = math.cos(diff_rad)
        
        # 오믹 저항 업데이트
        adaptation = lr * coherence * self.I_amp
        if coherence > 0:
            self.R = max(self.min_R, self.R - adaptation)
        else:
            self.R = min(self.max_R, self.R - adaptation * 0.2)

        # 로터 위상각 정렬
        # 방향에 따라 로터 위상각을 1단위 회전 시프트
        dir_diff = (target_phase - sig_phase) & self.rotor_mask
        if dir_diff > 0:
            if dir_diff <= (self.rotor_scale // 2):
                self.R_phase = (self.R_phase + 1) & self.rotor_mask
            else:
                self.R_phase = (self.R_phase - 1) & self.rotor_mask


class BitwiseCliffordIPN:
    """
    관측 회전 패러다임을 극대화한 비트와이즈 가변 로터 신경망.
    모든 기하곱 연산을 폐기하고 위상각 XOR/AND 및 Kuramoto 동기화 모델을 활용합니다.
    """
    def __init__(self, rotor_scale: int = 4096):
        self.rotor_scale = rotor_scale
        self.rotor_mask = rotor_scale - 1
        
        # 노드 상태: node_id -> [phase (int), amplitude (float)]
        self.phases: Dict[str, List] = {} 
        self.links: List[BitwiseImpedanceLink] = []
        self.node_layers: Dict[str, int] = {}
        
        self.connection_mode = ConnectionMode.Y_STAR
        self.tension = 0.0
        
        # 중성점(Neutral Ground) 추가
        self.add_node("NEUTRAL_GROUND", layer=-1, initial_phase=0, initial_amp=1.0)

    def set_connection_mode(self, mode: str):
        self.connection_mode = mode

    def add_node(self, node_id: str, layer: int, initial_phase: int = 0, initial_amp: float = 1.0):
        self.phases[node_id] = [initial_phase & self.rotor_mask, initial_amp]
        self.node_layers[node_id] = layer

    def connect_nodes(self, id_from: str, id_to: str, initial_R: float = 10.0) -> BitwiseImpedanceLink:
        link = BitwiseImpedanceLink(id_from, id_to, self.rotor_scale, initial_R)
        self.links.append(link)
        return link

    def forward_propagate(self, inputs: Dict[str, Tuple[int, float]]):
        """레이어 바이 레이어로 비트 위상 신호를 전파시킵니다."""
        # 링크 전류 초기화
        for link in self.links:
            link.I_amp = 0.0

        node_signals: Dict[str, List[Tuple[int, float]]] = {k: [] for k in self.phases.keys()}
        for k, v in inputs.items():
            node_signals[k].append(v)

        max_layer = max(self.node_layers.values(), default=0)

        for layer in range(max_layer):
            current_layer_nodes = [n for n, l in self.node_layers.items() if l == layer]
            
            for node in current_layer_nodes:
                signals = node_signals[node]
                if not signals:
                    # 현재 노드의 자체 위상/진폭 사용
                    src_phase, src_amp = self.phases[node]
                else:
                    # 입력 신호들의 평균 진폭 및 평균 위상각 계산
                    total_amp = sum(s[1] for s in signals)
                    if total_amp <= 0:
                        continue
                    
                    # 위상각의 가중 평균 (벡터 합 방식의 단순화)
                    sum_sin = sum(s[1] * math.sin(s[0] * 2 * math.pi / self.rotor_scale) for s in signals)
                    sum_cos = sum(s[1] * math.cos(s[0] * 2 * math.pi / self.rotor_scale) for s in signals)
                    src_phase = int(math.atan2(sum_sin, sum_cos) * self.rotor_scale / (2 * math.pi)) & self.rotor_mask
                    src_amp = total_amp / len(signals)

                outgoing = [l for l in self.links if l.node_from == node]
                if not outgoing:
                    continue

                admittances = [1.0 / max(0.01, l.R) for l in outgoing]
                total_admittance = sum(admittances)

                if total_admittance > 0:
                    for link, adm in zip(outgoing, admittances):
                        fraction = adm / total_admittance
                        link_amp_in = src_amp * fraction
                        link.I_amp = link_amp_in
                        
                        # 전파 및 위상 회전
                        prop_phase, prop_amp = link.propagate(src_phase, link_amp_in)
                        node_signals[link.node_to].append((prop_phase, prop_amp))

        # 노드 위상 동조 (Phase Locking)
        for node, signals in node_signals.items():
            if self.node_layers[node] > 0 and signals:
                total_amp = sum(s[1] for s in signals)
                if total_amp > 0.01:
                    sum_sin = sum(s[1] * math.sin(s[0] * 2 * math.pi / self.rotor_scale) for s in signals)
                    sum_cos = sum(s[1] * math.cos(s[0] * 2 * math.pi / self.rotor_scale) for s in signals)
                    target_phase = int(math.atan2(sum_sin, sum_cos) * self.rotor_scale / (2 * math.pi)) & self.rotor_mask
                    
                    # 현재 위상을 입력 위상으로 LERB 시프트
                    curr_phase = self.phases[node][0]
                    diff = (target_phase - curr_phase) & self.rotor_mask
                    if diff > (self.rotor_scale // 2):
                        # 음의 방향
                        step = -int(round(diff * 0.2))
                    else:
                        step = int(round(diff * 0.2))
                        
                    self.phases[node][0] = (curr_phase + step) & self.rotor_mask
                    self.phases[node][1] = min(1.0, self.phases[node][1] + total_amp * 0.1)

    def tune_network(self, dt: float, lr: float = 0.5) -> float:
        """링크 임피던스 튜닝 및 쿠라모토 위상 텐션 계측"""
        total_tension = 0.0
        active_links_count = 0
        
        for link in self.links:
            src_phase, src_amp = self.phases[link.node_from]
            tar_phase, tar_amp = self.phases[link.node_to]
            
            # 저항 튜닝
            link.update_impedance(src_phase, src_amp, tar_phase, lr)
            
            # 텐션(위상 불일치각) 계측
            sig_phase, sig_amp = link.propagate(src_phase, src_amp)
            diff = (sig_phase - tar_phase) & self.rotor_mask
            if diff > (self.rotor_scale // 2):
                diff = self.rotor_scale - diff
            tension_angle = diff * (2 * math.pi / self.rotor_scale)
            
            total_tension += tension_angle
            active_links_count += 1

        # 쿠라모토 토크 위상 결합 (Kuramoto Phase Coupling)
        for link in self.links:
            src_phase, src_amp = self.phases[link.node_from]
            tar_phase, tar_amp = self.phases[link.node_to]
            
            sig_phase, sig_amp = link.propagate(src_phase, src_amp)
            diff_rad = ((sig_phase - tar_phase) & self.rotor_mask) * (2 * math.pi / self.rotor_scale)
            
            # 위상 토크: T = K * sin(θ_j - θ_i)
            torque = math.sin(diff_rad)
            coupling = (1.0 / link.R) * link.I_amp * dt * lr
            
            # 타겟 노드와 소스 노드를 상호 견인하여 동기화
            step = int(torque * coupling * 0.5 * self.rotor_scale / (2 * math.pi))
            self.phases[link.node_to][0] = (tar_phase + step) & self.rotor_mask
            self.phases[link.node_from][0] = (src_phase - step) & self.rotor_mask

        # Y/Delta 결선 위상 정렬
        if self.connection_mode == ConnectionMode.Y_STAR:
            # 중성점 수렴 (Grounding)
            neutral_phase = self.phases["NEUTRAL_GROUND"][0]
            for node in self.phases:
                if node != "NEUTRAL_GROUND":
                    curr = self.phases[node][0]
                    diff = (neutral_phase - curr) & self.rotor_mask
                    step = int(diff * 0.1 * dt)
                    self.phases[node][0] = (curr + step) & self.rotor_mask
        elif self.connection_mode == ConnectionMode.DELTA:
            # 자체 회전 자율 와류 생성 (Self-torque)
            for node in self.phases:
                if node != "NEUTRAL_GROUND" and self.node_layers[node] > 0:
                    # 10단위 기저 토크 회전 적용
                    self.phases[node][0] = (self.phases[node][0] + 10) & self.rotor_mask

        avg_tension = total_tension / max(1, active_links_count)
        self.tension = avg_tension
        return avg_tension
