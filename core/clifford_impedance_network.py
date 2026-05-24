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
    def __init__(self, node_from: str, node_to: str, signature: Tuple[int, int] = (3, 0), initial_R: float = 10.0):
        self.node_from = node_from
        self.node_to = node_to
        
        # Z = R + rotor rotation
        self.R = float(initial_R)
        self.min_R = 0.5
        self.max_R = 100.0
        
        # Initial link rotor is identity (scalar 1.0)
        self.R_rotor = Multivector({0: 1.0}, signature)
        # Flow current signal (Multivector)
        self.I = Multivector({}, signature)

    def propagate(self, signal_in: Multivector) -> Multivector:
        """
        Propagates signal using Clifford sandwich product attenuated by resistance:
        S_out = (1 / R) * (R_rotor * S_in * R_rotor_conjugate)
        """
        rotated = self.R_rotor * signal_in * self.R_rotor.conjugate()
        return rotated * (1.0 / self.R)

    def update_impedance(self, signal_in: Multivector, target_state: Multivector, lr: float = 0.5):
        """
        Ohmic adaptation and rotor alignment step:
        1. Reduce resistance if propagated signal aligns with target state.
        2. Update link rotor to align propagation plane using bivector step.
        """
        propagated = self.propagate(signal_in)
        
        sig_norm = mv_normalize(propagated)
        tar_norm = mv_normalize(target_state)
        
        # Coherence: scalar part of dot (inner) product
        coherence_mv = sig_norm.dot(tar_norm)
        coherence = coherence_mv.data.get(0, 0.0)
        
        # Ohmic resistance update proportional to signal flow intensity
        flow_intensity = mv_norm(self.I)
        adaptation = lr * coherence * flow_intensity
        
        if coherence > 0:
            self.R = max(self.min_R, self.R - adaptation)
        else:
            self.R = min(self.max_R, self.R - adaptation * 0.2) # slower decay for blocking

        # Rotor tuning: step towards the alignment plane
        # Bivector representing plane of rotation from propagated signal to target state
        B = tar_norm ^ sig_norm
        signature = self.R_rotor.p, self.R_rotor.q
        
        # R_step = 1 + epsilon * B
        R_step = Multivector({0: 1.0}, signature) + B * (lr * 0.05)
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

    def connect_nodes(self, id_from: str, id_to: str, initial_R: float = 10.0) -> CliffordImpedanceLink:
        link = CliffordImpedanceLink(id_from, id_to, self.signature, initial_R)
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

                # Distribute signal intensity based on admittance (1 / R)
                admittances = [1.0 / max(0.01, l.R) for l in outgoing]
                total_admittance = sum(admittances)

                if total_admittance > 0:
                    for link, adm in zip(outgoing, admittances):
                        fraction = adm / total_admittance
                        link_signal_in = input_signal * fraction
                        
                        # Set link current
                        link.I = link_signal_in
                        
                        # Propagate through link (attends sandwich product)
                        propagated = link.propagate(link_signal_in)
                        
                        node_to = link.node_to
                        node_signals[node_to] = node_signals[node_to] + propagated

        # Accumulate and update hidden/output node states (phase locking/resonance)
        for node, signal in node_signals.items():
            if self.node_layers[node] > 0 and mv_norm(signal) > 0.01:
                # Node phase state aligns with incoming signals
                self.phases[node] = mv_normalize(self.phases[node] + signal * 0.2)

    def tune_network(self, dt: float, lr: float = 0.5) -> float:
        """
        Tunes link impedances, performs bivector phase locking,
        and computes global network tension.
        """
        total_tension = 0.0
        active_links_count = 0
        
        # 1. Update link resistances and rotors
        for link in self.links:
            sig_in = self.phases[link.node_from]
            target = self.phases[link.node_to]
            
            # Ohmic coherence adaptation
            link.update_impedance(sig_in, target, lr)
            
            # Measure local misalignment (tension) as angle between propagated signal and target
            propagated = link.propagate(sig_in)
            sig_norm = mv_normalize(propagated)
            tar_norm = mv_normalize(target)
            
            coherence = sig_norm.dot(tar_norm).data.get(0, 0.0)
            coherence = min(1.0, max(-1.0, coherence))
            tension_angle = math.acos(coherence)
            
            total_tension += tension_angle
            active_links_count += 1

        # 2. Phase-Locking (Kuramoto torque-coupling between connected nodes)
        for link in self.links:
            sig_in = self.phases[link.node_from]
            target = self.phases[link.node_to]
            
            propagated = link.propagate(sig_in)
            sig_norm = mv_normalize(propagated)
            tar_norm = mv_normalize(target)
            
            # Bivector representing phase torque misalignment
            B = tar_norm ^ sig_norm
            coupling = (1.0 / link.R) * mv_norm(link.I) * dt
            
            # Attract target towards rotated signal
            self.phases[link.node_to] = mv_normalize(self.phases[link.node_to] - B * (coupling * 0.5))
            # React back onto source node (rotated back)
            rotated_B = link.R_rotor.conjugate() * B * link.R_rotor
            self.phases[link.node_from] = mv_normalize(self.phases[link.node_from] + rotated_B * (coupling * 0.5))

        # 3. Y/Delta 모드에 따른 물리적 위상 강제 처리
        if self.connection_mode == ConnectionMode.Y_STAR:
            # [Y결선 모드] 모든 노드가 중성점에 동기화되어 위상 노이즈 방전
            neutral = self.phases["NEUTRAL_GROUND"]
            for node, mv in self.phases.items():
                if node != "NEUTRAL_GROUND":
                    # 중성점의 강제 견인력(Grounding force)
                    B_ground = neutral ^ mv
                    self.phases[node] = mv_normalize(self.phases[node] - B_ground * (lr * 0.1 * dt))
        elif self.connection_mode == ConnectionMode.DELTA:
            # [Delta결선 모드] 중성점 간섭 배제 및 사유 와류(Self-Sustaining Torque) 생성
            for node, mv in self.phases.items():
                if node != "NEUTRAL_GROUND" and self.node_layers[node] > 0:
                    # 노드 자체가 지닌 위상 각속도를 유지하여 회전 토크 발생
                    torque = Multivector({3: 1.0}, self.signature) # e12 평면 토크 예시
                    self.phases[node] = mv_normalize(self.phases[node] + (torque * mv) * (lr * 0.05 * dt))

        # Calculate average tension
        avg_tension = total_tension / max(1, active_links_count)
        self.tension = avg_tension
        
        return avg_tension

    def bifurcate(self) -> bool:
        """Expands Clifford dimension Cl(p,0) -> Cl(p+1,0) (Dimension Split)."""
        current_axes = self.signature[0]
        if current_axes >= self.MAX_AXES:
            return False
            
        new_axes = current_axes + 1
        new_sig = (new_axes, 0)
        self.signature = new_sig
        
        # Update node phases
        for node_id, mv in self.phases.items():
            # Project existing data to new signature
            new_data = mv.data.copy()
            
            # Inject a small, deterministic causal perturbation on the new dimension (e_new)
            # The bitmask for the new basis vector is 1 << (new_axes - 1)
            new_mask = 1 << (new_axes - 1)
            # Deterministic perturbation based on node name hash
            perturbation = float(hash(node_id) % 100) / 1000.0 * 0.1
            new_data[new_mask] = perturbation
            
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
