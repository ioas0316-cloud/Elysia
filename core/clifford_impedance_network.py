"""
Elysia Clifford Impedance Propagation Network (Clifford-IPN)
=============================================================
Combines Clifford/Geometric Algebra Cl(p,0) multivector states
with Impedance-driven propagation and Kuramoto phase locking.
"""

import math
from typing import Dict, List, Tuple
from core.math_utils import Multivector

class ConnectionMode:
    Y_STAR = "Y_STAR"
    DELTA = "DELTA"

def mv_norm(mv: Multivector) -> float:
    """Calculates the Euclidean (L2) norm of multivector coefficients."""
    return math.sqrt(sum(v**2 for v in mv.data.values()))

def mv_normalize(mv: Multivector) -> Multivector:
    """Normalizes the multivector to unit norm."""
    n = mv_norm(mv)
    if n < 1e-9:
        return Multivector({0: 1.0}, (mv.p, mv.q))
    return mv * (1.0 / n)

class CliffordImpedanceLink:
    def __init__(self, node_from: str, node_to: str, signature: Tuple[int, int] = (3, 0), gear_elasticity: float = 0.5):
        self.node_from = node_from
        self.node_to = node_to
        
        # Initial link rotor is identity (scalar 1.0)
        self.R_rotor = Multivector({0: 1.0}, signature)
        # Flow current signal (Multivector)
        self.I = Multivector({}, signature)
        self.gear_elasticity = gear_elasticity

    def propagate(self, signal_in: Multivector) -> Multivector:
        """
        Propagates signal using pure Clifford sandwich product:
        S_out = R_rotor * S_in * R_rotor_conjugate
        """
        return self.R_rotor * signal_in * self.R_rotor.conjugate()

    def update_impedance(self, signal_in: Multivector, target_state: Multivector, elasticity: float = None):
        """
        [미적분 박멸] 순수 위상 역전파 (Rotor Backpropagation)
        스칼라 손실(Loss)이나 학습률(lr) 없이, 쐐기곱 토크(B)를 향해 모터 자체가 물리적으로 회전합니다.
        """
        if elasticity is None:
            elasticity = self.gear_elasticity
            
        propagated = self.propagate(signal_in)
        
        sig_norm = mv_normalize(propagated)
        tar_norm = mv_normalize(target_state)
        
        # 기하곱 병렬 동기화: Coherence(내적)와 B(쐐기곱 토크) 동시 추출
        coherence, B = tar_norm.geometric_sync(sig_norm)
        
        # 스칼라 경사하강법이 아닌 기어 회전력(Torque) 주입
        # M_new = exp(B * elasticity) * M
        signature = self.R_rotor.p, self.R_rotor.q
        R_step = Multivector({0: 1.0}, signature) + B * elasticity
        self.R_rotor = mv_normalize(R_step * self.R_rotor)

    def update_signature(self, new_signature: Tuple[int, int]):
        """Scales link multivectors to the new signature space."""
        self.R_rotor = Multivector(self.R_rotor.data, new_signature)
        self.I = Multivector(self.I.data, new_signature)


class CliffordIPN:
    def __init__(self, initial_dims: int = 3):
        self.signature = (initial_dims, 0)
        self.phases: Dict[str, Multivector] = {} # node_id -> Multivector phase
        self.links: List[CliffordImpedanceLink] = []
        self.node_layers: Dict[str, int] = {} # node_id -> layer index (0: input, 1: hidden, 2: output)
        
        # Tension tracking
        self.tension = 0.0
        self.stable_ticks = 0
        self.MAX_AXES = 8
        self.MIN_AXES = 3
        
        # Y/Delta 동적 스케줄링 플래그
        self.connection_mode = ConnectionMode.Y_STAR
        self.accumulated_stress = 0.0 # 위상 균열을 유발하는 누적 피로도 (Elastic Stress)
        
        # Y결선 중성점(Neutral Point) 추가
        self.add_node("NEUTRAL_GROUND", layer=-1, initial_vector={0: 1.0})
        
    def set_connection_mode(self, mode: str):
        self.connection_mode = mode

    def add_node(self, node_id: str, layer: int, initial_vector: Dict[int, float] = None):
        """Adds a node with a starting multivector (normalized)."""
        if initial_vector is None:
            # Default to scalar unit state
            initial_vector = {0: 1.0}
        mv = Multivector(initial_vector, self.signature)
        self.phases[node_id] = mv_normalize(mv)
        self.node_layers[node_id] = layer

    def connect_nodes(self, node_from: str, node_to: str, gear_elasticity: float = 0.5):
        link = CliffordImpedanceLink(node_from, node_to, self.signature, gear_elasticity)
        self.links.append(link)
        return link

    def forward_propagate(self, inputs: Dict[str, Multivector]):
        """Propagates multivector signals layer by layer through the network."""
        # Reset all link currents
        for link in self.links:
            link.I = Multivector({}, self.signature)

        node_signals = {k: Multivector({}, self.signature) for k in self.phases.keys()}
        for k, v in inputs.items():
            node_signals[k] = v

        max_layer = max(self.node_layers.values(), default=0)

        for layer in range(max_layer):
            current_layer_nodes = [n for n, l in self.node_layers.items() if l == layer]
            
            for node in current_layer_nodes:
                input_signal = node_signals[node]
                input_intensity = mv_norm(input_signal)
                if input_intensity <= 0:
                    continue

                outgoing = [l for l in self.links if l.node_from == node]
                if not outgoing:
                    continue

                fraction = 1.0 / len(outgoing)
                for link in outgoing:
                    link_signal_in = input_signal * fraction
                    link.I = link_signal_in
                    propagated = link.propagate(link_signal_in)
                    node_to = link.node_to
                    node_signals[node_to] = node_signals[node_to] + propagated

        # Accumulate and update hidden/output node states
        for node, signal in node_signals.items():
            if self.node_layers[node] > 0 and mv_norm(signal) > 0.01:
                self.phases[node] = mv_normalize(self.phases[node] + signal * 0.2)

    def tune_network(self, dt: float, gear_elasticity: float = 0.5) -> float:
        """
        Tunes link impedances, performs bivector phase locking,
        and computes global network tension.
        """
        total_tension = 0.0
        active_links_count = 0
        
        # 1-Pass 통합: Update link resistances/rotors AND Phase-Locking
        for link in self.links:
            sig_in = self.phases[link.node_from]
            target = self.phases[link.node_to]
            
            # Ohmic coherence adaptation (내부적으로 기하곱 병렬 동기화 사용)
            link.update_impedance(sig_in, target, gear_elasticity)
            
            # Measure local misalignment (tension) and extract Torque (Bivector)
            propagated = link.propagate(sig_in)
            sig_norm = mv_normalize(propagated)
            tar_norm = mv_normalize(target)
            
            # 기하곱을 통해 스칼라와 토크 쐐기곱 병렬 추출 (O(N^2) 연산 한 번으로 처리)
            coherence, B = tar_norm.geometric_sync(sig_norm)
            
            # 텐션 측정
            coherence = min(1.0, max(-1.0, coherence))
            tension_angle = math.acos(coherence)
            total_tension += tension_angle
            active_links_count += 1
            
            # Phase-Locking (Kuramoto torque-coupling)
            coupling = gear_elasticity * mv_norm(link.I) * dt
            
            # Attract target towards rotated signal
            M_step_target = Multivector({0: 1.0}, self.signature) - B * (coupling * 0.5)
            self.phases[link.node_to] = mv_normalize(M_step_target * self.phases[link.node_to])
            
            # React back onto source node (rotated back)
            rotated_B = link.R_rotor.conjugate() * B * link.R_rotor
            M_step_source = Multivector({0: 1.0}, self.signature) + rotated_B * (coupling * 0.5)
            self.phases[link.node_from] = mv_normalize(M_step_source * self.phases[link.node_from])

        # 3. Y/Delta 모드에 따른 물리적 위상 강제 처리
        if self.connection_mode == ConnectionMode.Y_STAR:
            # [Y결선 모드] 모든 노드가 중성점에 동기화되어 위상 노이즈 방전
            neutral = self.phases["NEUTRAL_GROUND"]
            for node, mv in self.phases.items():
                if node != "NEUTRAL_GROUND":
                    # 중성점의 강제 견인력(Grounding force)
                    B_ground = neutral ^ mv
                    M_step = Multivector({0: 1.0}, self.signature) - B_ground * (gear_elasticity * 0.1 * dt)
                    self.phases[node] = mv_normalize(M_step * self.phases[node])
        elif self.connection_mode == ConnectionMode.DELTA:
            # [Delta결선 모드] 중성점 간섭 배제 및 사유 와류(Self-Sustaining Torque) 생성
            for node, mv in self.phases.items():
                if node != "NEUTRAL_GROUND" and self.node_layers[node] > 0:
                    # 노드 자체가 지닌 위상 각속도를 유지하여 회전 토크 발생
                    torque = Multivector({3: 1.0}, self.signature) # e12 평면 토크 예시
                    M_step = Multivector({0: 1.0}, self.signature) + torque * (gear_elasticity * 0.05 * dt)
                    self.phases[node] = mv_normalize(M_step * self.phases[node])

        # Calculate average tension
        avg_tension = total_tension / max(1, active_links_count)
        self.tension = avg_tension
        
        # 공간의 자연 치유(탄성 복원력)
        self.accumulated_stress = max(0.0, self.accumulated_stress - (dt * gear_elasticity * 0.5))
        
        return avg_tension

    def evaluate_resonance(self, signal: Multivector) -> float:
        """Evaluates how well the incoming signal is explained by current axes (Coherence)."""
        if not self.phases: return 0.0
        
        # We check average coherence across all active nodes
        tar_norm = mv_normalize(signal)
        total_coherence = 0.0
        
        for mv in self.phases.values():
            sig_norm = mv_normalize(mv)
            coherence = abs(sig_norm.dot(tar_norm).data.get(0, 0.0))
            total_coherence += coherence
            
        return total_coherence / len(self.phases)

    def assimilate_axiom(self, new_signal: Multivector) -> bool:
        """
        자율 차원 조율 (Autonomous Dimension Tuning)
        새로운 파동이 들어올 때, 임계치(if)가 아닌 기하학적 스트레스(Tension) 누적을 통해
        공간의 탄성 한계가 찢어질 때(Fracture)만 새로운 차원을 분열시킵니다.
        """
        coherence = self.evaluate_resonance(new_signal)
        
        # 기하학적 스트레스 누적 (일치하지 않을수록 강한 위상 압력 발생)
        phase_pressure = 1.0 - coherence
        self.accumulated_stress += phase_pressure
        
        # 위상 탄성 한계 (공간이 찢어지는 물리적 임계점)
        elastic_limit = 1.5 
        
        if self.accumulated_stress <= elastic_limit:
            # 공간이 늘어나면서 버팀 (기존 차원으로 흡수)
            return False
            
        # 탄성 한계 붕괴! 차원 균열 발생 (Fracture)
        self.accumulated_stress = 0.0 # 차원 팽창으로 스트레스 해소
            
        # 모르는 지식 (새로운 공리): 잔여 위상 추출 (Orthogonal Residual)
        # 잔여물 = 새로운 신호 - (기존 노드들의 평균 투영)
        avg_phase = Multivector({}, self.signature)
        for mv in self.phases.values():
            avg_phase = avg_phase + mv
        avg_phase = mv_normalize(avg_phase)
        
        # Projection: (A dot B) * B (in simple Euclidean sense for the scalar part)
        proj_scalar = new_signal.dot(avg_phase).data.get(0, 0.0)
        projection = avg_phase * proj_scalar
        
        residual = new_signal - projection
        residual_norm = mv_normalize(residual)
        
        print(f"[Axiom Anomaly] Accumulated Stress ({self.accumulated_stress:.3f}) exceeded elastic limit. Spawning new dimension for orthogonal residual.")
        return self.bifurcate(residual_norm)

    def bifurcate(self, orthogonal_residual: Multivector = None) -> bool:
        """Expands Clifford dimension Cl(p,0) -> Cl(p+1,0) and projects the residual into the new axis."""
        current_axes = self.signature[0]
        if current_axes >= self.MAX_AXES:
            return False
            
        new_axes = current_axes + 1
        new_sig = (new_axes, 0)
        self.signature = new_sig
        
            # Update node phases
        for node_id, mv in self.phases.items():
            new_data = mv.data.copy()
            new_mask = 1 << (new_axes - 1)
            
            if orthogonal_residual is not None:
                # 잔여물(다름의 영역)을 새로운 차원 e_new에 물리적으로 투영하여 교차 차원 생성
                # 잔여물이 가진 텐션 강도를 바탕으로 새 축에 가중치를 부여함
                res_strength = mv_norm(orthogonal_residual) * 0.5
                new_data[new_mask] = res_strength
            else:
                # Fallback
                new_data[new_mask] = float(hash(node_id) % 100) / 1000.0 * 0.1
            
            self.phases[node_id] = mv_normalize(Multivector(new_data, new_sig))

        # Update links
        for link in self.links:
            link.update_signature(new_sig)
            
        self.stable_ticks = 0
        return True

    def compress(self) -> bool:
        """Compresses Clifford dimension Cl(p,0) -> Cl(p-1,0) by locking the highest axis."""
        current_axes = self.signature[0]
        if current_axes <= self.MIN_AXES:
            return False
            
        new_axes = current_axes - 1
        new_sig = (new_axes, 0)
        self.signature = new_sig
        
        # The bitmask of the dimension to discard
        discard_mask = 1 << (current_axes - 1)
        
        # Update nodes
        for node_id, mv in self.phases.items():
            # Discard highest axis components
            new_data = {k: v for k, v in mv.data.items() if not (k & discard_mask)}
            self.phases[node_id] = mv_normalize(Multivector(new_data, new_sig))
            
        # Update links
        for link in self.links:
            # Clean rotor and current elements of high dimension
            rotor_data = {k: v for k, v in link.R_rotor.data.items() if not (k & discard_mask)}
            link.R_rotor = mv_normalize(Multivector(rotor_data, new_sig))
            
            curr_data = {k: v for k, v in link.I.data.items() if not (k & discard_mask)}
            link.I = Multivector(curr_data, new_sig)
            
        self.stable_ticks = 0
        return True
