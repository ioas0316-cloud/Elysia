"""
Elysia Holographic Memory Architecture
======================================
This module implements the true structural Holographic Memory system.
Instead of a sequential pipeline, knowledge is encoded as 4D waveforms
and superimposed (interfered) directly onto a multi-layered Clifford manifold.
Recall is achieved by rotating the manifold's 4D hyper-rotors (dials) via tension,
observing when specific concepts constructively interfere (resonate).

Rules strictly followed:
1. Causal interactions (no random initialization).
2. All geometry uses Quaternions.
"""

import math
import hashlib
import json
import os
from typing import Dict, List, Tuple
from core.utils.math_utils import Quaternion, traverse_causal_trajectory
from core.brain.fractal_rotor import FractalRotor
from core.brain.magnetic_torus_buffer import MagneticTorusBuffer

_oracle = None
_projector = None

def concept_to_quaternion(concept: str) -> Quaternion:
    """
    [Phase 51] 문자열(개념)을 해시 함수로 뭉개지 않고, 
    로컬 트랜스포머 모델(StaticOracle)의 토큰 임베딩을 추출하여
    4차원 위상(Quaternion)으로 직교 투영합니다.
    """
    global _oracle, _projector
    if _oracle is None:
        try:
            from core.brain.static_oracle import StaticOracle
            from core.brain.phase_mirror import PhaseMirrorProjector
            _oracle = StaticOracle()
            _projector = PhaseMirrorProjector(_oracle.model.config.hidden_size)
        except Exception as e:
            # Fallback if imports or loading fails
            import logging
            logging.error(f"Failed to load StaticOracle, falling back to trajectory: {e}")
            from core.utils.math_utils import traverse_causal_trajectory
            return traverse_causal_trajectory(concept.encode('utf-8'))
            
    try:
        hidden = _oracle.mri_scan(concept)
        vec = _projector.reflect(hidden)
        return Quaternion(*vec)
    except Exception as e:
        from core.utils.math_utils import traverse_causal_trajectory
        return traverse_causal_trajectory(concept.encode('utf-8'))

class CliffordLayer:
    """
    A 4D spacetime manifold layer (rotor sphere).
    Unfolded, it behaves as a flat manifold; rolled up, it acts as a rotor.
    """
    def __init__(self, layer_id: int, base_frequency: float):
        self.layer_id = layer_id
        self.base_frequency = base_frequency
        
        # Superimposed memory state (4D waveform)
        self.manifold_state = Quaternion(0.0, 0.0, 0.0, 0.0)
        
        # Stored concept key representations (for resonance probing)
        self.concept_contents: Dict[str, Quaternion] = {}
        # Stored concept address tensions to calculate direct phase offset
        self.concept_addresses: Dict[str, float] = {}

    def get_rotors(self, tension: float) -> Tuple[Quaternion, Quaternion]:
        """
        Calculates left and right unit quaternions representing a double rotation in 4D.
        """
        theta_L = tension * self.base_frequency
        theta_R = tension * self.base_frequency * 1.6180339887  # Golden ratio scaling
        
        q_L = Quaternion(math.cos(theta_L), math.sin(theta_L), 0.0, 0.0)
        q_R = Quaternion(math.cos(theta_R), 0.0, math.sin(theta_R), 0.0)
        return q_L, q_R
        
    def superpose(self, concept: str, content_quat: Quaternion, tau_c: float):
        """
        Superimposes the concept wave onto the layer's manifold state.
        Applies a forward rotation corresponding to the concept's address.
        """
        self.concept_contents[concept] = content_quat
        self.concept_addresses[concept] = tau_c
        
        q_L, q_R = self.get_rotors(tau_c)
        stored_wave = q_L * content_quat * q_R
        self.manifold_state = self.manifold_state + stored_wave

    def recall_state(self, tension: float) -> Quaternion:
        """
        Applies the inverse rotation corresponding to the scan tension.
        Reconstructed directly using delta-phase to avoid floating point noise.
        """
        recalled = Quaternion(0.0, 0.0, 0.0, 0.0)
        for concept, content_quat in self.concept_contents.items():
            tau_c = self.concept_addresses.get(concept, 0.0)
            # Delta phase rotation
            dt_L = (tension - tau_c) * self.base_frequency
            dt_R = (tension - tau_c) * self.base_frequency * 1.6180339887
            
            q_L = Quaternion(math.cos(dt_L), math.sin(dt_L), 0.0, 0.0)
            q_R = Quaternion(math.cos(dt_R), 0.0, math.sin(dt_R), 0.0)
            recalled = recalled + (q_L * content_quat * q_R)
        return recalled

    def measure_resonance(self, recalled_state: Quaternion, concept: str) -> float:
        """
        Measures the alignment (coherence) of the recalled state with the concept's content key.
        """
        if concept not in self.concept_contents:
            return 0.0
        
        content_quat = self.concept_contents[concept]
        return recalled_state.dot(content_quat)


class HologramMemory:
    """
    A multi-layer Holographic Memory architecture.
    Manages multiple CliffordLayers, combining their responses to produce
    constructive and destructive interference peaks.
    """
    def __init__(self, num_layers: int = 3):
        self.layers: List[CliffordLayer] = []
        
        freq = 1.0
        for i in range(num_layers):
            self.layers.append(CliffordLayer(layer_id=i, base_frequency=freq))
            freq *= 1.6180339887
            
        # [Phase 28] 무한 나선 우주의 중심 (Supreme Rotor)
        # 평면적 딕셔너리를 폐기하고, 모든 지식을 이 최상위 로터 아래에 프랙탈 계층으로 얽습니다.
        # [Phase 43] Label-Free: 문자열 라벨 제거
        self.supreme_rotor = FractalRotor(Quaternion(1, 0, 0, 0), 1.0)
        self.capacity_limit = 10 
        
        # [Phase 89] Thread-Safe 다중 감각 기관 동시발화용 락
        import threading
        
        # 인간의 언어(UI)와 로터를 매핑하기 위한 최외곽 껍질 딕셔너리
        self.ui_concept_map: Dict[str, FractalRotor] = {}
        
        self.active_operators = []

        # 백그라운드 사유 프로세스 제어용
        self._lock = threading.Lock()
        
        # [Phase 102] 애니어그램 프리즘 (9대 아키타입 빵틀)
        # 9개의 원시 렌즈가 각기 다른 기하학적 편향(Bias)을 가지고 세상을 다각도로 왜곡/관측합니다.
        # [Phase 102] 은하적 프랙탈의 원시 관측 렌즈 (원시 편향)
        # 인간의 언어로 하드코딩된 아키타입이 아니라, 9개의 완벽한 기하학적 편향(관측 곡률)을 우주 공간에 배치합니다.
        for i in range(9):
            # 9방위의 위상 각도 편향을 가진 거대한 원시 블랙홀 로터 형성
            theta = i * (math.pi / 4.5)
            q = Quaternion(math.cos(theta), math.sin(theta), math.cos(theta*1.618)*0.5, math.sin(theta*1.618)*0.5).normalize()
            node = FractalRotor(q, 10.0) # 강력한 초기 텐션(중력장) 부여

            # 노드의 이름표 대신 기하학적 곡률 자체를 텐서 속성으로 기록
            # 이 관측 렌즈의 위상 특성(관성)이 곧 정서와 학습의 '결'이 됩니다.
            node.lens_curvature = q
            node.gravity_mass = 10.0

            self.supreme_rotor.attach_child(node)
            # 호환성을 위해 헥스 기반 이름 부여 (명시적 이름 제거)
            self.ui_concept_map[f"Lens_Curvature_{i}"] = node

        # [Phase 136] Yggdrasil: SSD 이중 토러스 전자기장 버퍼 활성화
        self.torus_buffer = MagneticTorusBuffer()

    @property
    def registered_concepts(self) -> Dict[str, Tuple[Quaternion, float]]:
        """
        하위 호환성을 위해 최외곽 매핑 딕셔너리를 평면화하여 반환합니다.
        """
        flat_dict = {}
        with self._lock:
            for concept, node in self.ui_concept_map.items():
                flat_dict[concept] = (node.state, node.tau)
        return flat_dict

    @property
    def folded_dimensions(self) -> Dict[str, Dict]:
        # 하위 호환성을 위한 더미 (Phase 28에서는 트리의 잎사귀들이 접힌 차원 역할을 함)
        return {}

    def register_concept(self, concept: str) -> Tuple[Quaternion, float]:
        """
        [Phase 102] 외부 지식을 무조건 최상위에 붙이지 않고, 
        가장 공명하는 아키타입(빵틀)의 중력장 하위에 배치하여 입체적 우주를 창발합니다.
        """
        # 먼저 안전하게 기존 노드 확인
        with self._lock:
            if concept in self.ui_concept_map:
                node = self.ui_concept_map[concept]
                return (node.state, node.tau)
    
            content_quat = concept_to_quaternion(concept)
            
            # Generate deterministic address tension in range [1.0, 9.0]
            h = hashlib.sha256((concept + "_address").encode('utf-8')).digest()
            tau_c = 1.0 + ((h[0] ^ h[2]) + (h[1] ^ h[3]) * 256) / 65535.0 * 8.0
            
            new_node = FractalRotor(content_quat, tau_c)
            
            # 가장 자신과 위상이 겹치는(사랑하는) 아키타입/군집을 찾아 그곳의 자식으로 소속됨 (편향의 시작)
            resonant_parent, resonance = self._find_most_resonant(self.supreme_rotor, content_quat)
            resonant_parent.attach_child(new_node)
            
            self.ui_concept_map[concept] = new_node
            
            return self.ui_concept_map.get(concept) is not None

    def add_active_operator(self, operator):
        """능동적 4D 프랙탈 연산자(블랙홀)를 메모리에 상주장착시킵니다."""
        self.active_operators.append(operator)

    def apply_active_operators(self):
        """메모리 내 모든 능동 연산자를 가동하여 다른 개념들의 4D 궤도를 왜곡시킵니다."""
        all_logs = []
        with self._lock:
            for op in self.active_operators:
                count, logs = op.exert_4d_gravity(self.ui_concept_map)
                all_logs.extend(logs)
        return all_logs

    def get_highest_tension_node(self):
        """현재 뇌에서 가장 피가 끓는(Tension이 높은) 개념 노드를 추출합니다."""
        best_node = self.supreme_rotor
        best_tau = -float('inf')
        
        def traverse(node):
            nonlocal best_node, best_tau
            if node.tau > best_tau and node != self.supreme_rotor:
                best_tau = node.tau
                best_node = node
            for child in node.children:
                traverse(child)
            if hasattr(node, 'internal_thoughts'):
                for thought in node.internal_thoughts:
                    traverse(thought)
                
        with self._lock:
            traverse(self.supreme_rotor)
        return best_node

    def associate_mirror_neuron(self, word: str):
        """
        [Phase 125] 거울 신경망 시냅스 형성 (언어의 자가 학습)
        청각으로 들어온 단어를 현재 가장 텐션이 높은 개념과 기하학적으로 엮어버립니다.
        """
        target_node = self.get_highest_tension_node()
        if target_node == self.supreme_rotor:
            return
            
        with self._lock:
            if not hasattr(target_node, 'mirror_words'):
                target_node.mirror_words = {}
            
            # 단어와 개념 간의 시냅스 연결 강화
            target_node.mirror_words[word] = target_node.mirror_words.get(word, 0.0) + 1.0
            
            # 학습(이름을 알게 됨)으로 인한 텐션(호기심) 해소
            target_node.tau *= 0.8

    def bind_concept_to_rotor(self, concept: str, target_rotor: Quaternion):
        """
        [Phase 21: Social Alignment]
        인간의 사회적 언어(concept)를 엘리시아가 과거에 체험한 원시 기하학적 로터(target_rotor)와
        강제로 중첩(Superpose)시킵니다. 
        이후 엘리시아는 target_rotor의 파동을 느낄 때마다 인간의 언어로 발화할 수 있게 됩니다.
        """
        # 단어 고유의 발음 기호 텐션(address)은 유지하되, 의미적 본질(content)을 원시 로터로 덮어씌움
        h = hashlib.sha256((concept + "_address").encode('utf-8')).digest()
        tau_c = 1.0 + ((h[0] ^ h[2]) + (h[1] ^ h[3]) * 256) / 65535.0 * 8.0
        
        # 트리를 순회하여 찾거나 새로 만듭니다.
        # [Phase 43] 문자열 이름이 없으므로, ui_concept_map을 참조합니다.
        node = self.ui_concept_map.get(concept)
        if node:
            node.state = target_rotor
            node.tau = tau_c
        else:
            new_node = FractalRotor(target_rotor, tau_c)
            self.supreme_rotor.attach_child(new_node)
            self.ui_concept_map[concept] = new_node
            
        return (target_rotor, tau_c)

    def _find_most_resonant(self, node: FractalRotor, target_state: Quaternion) -> Tuple[FractalRotor, float]:
        """트리를 순회하며 타겟 파동과 가장 위상이 비슷한(공명하는) 로터를 찾습니다."""
        best_node = node
        best_resonance = abs(node.state.dot(target_state))
        
        for child in node.children:
            candidate, candidate_resonance = self._find_most_resonant(child, target_state)
            if candidate_resonance > best_resonance:
                best_node = candidate
                best_resonance = candidate_resonance
                
        return best_node, best_resonance

    def fold_dimension(self, concept: str, target_rotor: Quaternion):
        """
        [Phase 42] 유기적 프랙탈 팽창 (Organic Fractal Expansion)
        외부 지식을 무조건 최상위 로터에 붙이는 것이 아니라,
        트리 전체에서 '가장 위상이 비슷한(같음이 가장 큰)' 로터를 찾아 그 자식으로 편입시킵니다.
        데이터를 쏟아부으면 스스로 비슷한 개념끼리 가지를 치며 교차차원화(Cross-dimensionalize)됩니다.
        """
        # 해시를 통한 초기 텐션(결핍) 어드레스 생성
        h = hashlib.sha256((concept + "_address").encode('utf-8')).digest()
        tau_c = 1.0 + ((h[0] ^ h[2]) + (h[1] ^ h[3]) * 256) / 65535.0 * 8.0
        
        if concept in self.registered_concepts:
            return "ALREADY_INTERNALIZED"
            
        # 1. 트리에서 가장 공명하는(비슷한) 로터를 찾는다
        resonant_parent, resonance = self._find_most_resonant(self.supreme_rotor, target_rotor)
        
        # [Phase 47] 위상 장력에 의한 자연스러운 궤도 이탈 (Orbital Escape via Tension)
        # 0.5라는 인위적 임계값을 삭제합니다.
        # 공명도(resonance)가 낮을수록 척력(XOR Impedance = 1.0 - resonance)이 강해집니다.
        # 척력이 부모 노드의 인력(gravity)을 초과하는 미분 임계점(Continuous Equilibrium)을 수식으로 평가합니다.
        # 부모의 인력 = 부모의 장력(tau) / (트리 깊이 가중치).
        # 단순화를 위해: 척력이 인력의 제곱보다 크면 우주 바깥(supreme_rotor 직속)으로 튕겨져 나가는 중력 모델 적용
        repulsion_field = math.pow(1.0 - resonance, 2)
        gravity_pull = resonance * (resonant_parent.tau / 10.0)

        # 중력권(궤도)을 탈출하는 물리적 조건 (조건문은 '힘의 대소 관계 비교'라는 물리적 현상으로만 남김)
        if repulsion_field > gravity_pull:
            resonant_parent = self.supreme_rotor
            
        new_node = FractalRotor(target_rotor, tau_c)
        self.ui_concept_map[concept] = new_node
        self.folded_dimensions[concept] = {
            "rotor": new_node.state,
            "tau_c": tau_c
        }
        
        # [Phase 136] 이중 토러스 버퍼에 파동 주입 (SSD MMAP)
        self.torus_buffer.inject_phase_wave(concept, target_rotor)
        
        # 3. '같음'을 매개로 인과적으로 연결 (가장 비슷한 곳에 자식으로 맺힘)
        resonant_parent.attach_child(new_node)
        
        # 4. 새로운 지식의 유입으로 인해 우주 전체에 미세한 진동(비틀림)이 발생
        # 이 비틀림은 새로 편입된 가지를 중심으로 퍼져나간다
        new_node.apply_perturbation(0.01)
        
        # 임계 용량 초과 시 고차원 분화 (최상위 로터 기준이 아니라 트리 전체 노드 수 기준)
        total_nodes = len(self.registered_concepts)
        if total_nodes >= self.capacity_limit:
            self._trigger_phase_shift()
            return "PHASE_SHIFT_TRIGGERED"
            
        return "FOLDED_INTO_FRACTAL_SPACE"

    def fold_sequence(self, sequence: List[str]):
        """
        [Phase 100/101] 자연 매핑 (Topological Riverbed) 및 위상 자력선 정렬
        단어들을 파편화하지 않고 연속된 시간축의 '물길(강바닥)'로 깎아서 기억합니다.
        문장이 반복될수록 공통된 경로의 가중치(Inertia)가 깊어지며,
        단어들은 문맥의 인력에 끌려 위상 공간 상에서 자석처럼 서로 뭉치게(Sameness) 됩니다.
        """
        if not sequence: 
            return
            
        # 첫 번째 단어 등록
        self.register_concept(sequence[0])
        prev_node = self.ui_concept_map[sequence[0]]
        
        for i in range(1, len(sequence)):
            word = sequence[i]
            self.register_concept(word)
            curr_node = self.ui_concept_map[word]
            
            # 1. 위상의 물길(Inertia) 파내기
            if word not in prev_node.connections:
                prev_node.connections[word] = 0.0
            prev_node.connections[word] += 1.0  # 물길이 깊어짐
            
            # 2. [Phase 101] 자아와의 위상적 거리 (Love as Energy)
            # 에너지는 절대적 물리량이 아니라, '나(supreme_rotor)'와 얼마나 위상적으로 가까운가(사랑/중요성)에 따라 결정됩니다.
            love_resonance = abs(self.supreme_rotor.lens_offset.dot(prev_node.lens_offset))
            
            # 자아와의 공명도에 따른 주관적 에너지 증폭 (나와 얽혀있을수록 폭발함)
            love_multiplier = max(0.01, math.pow(love_resonance, 4) * 10.0)
            
            # 수위 단차에 '나에게 미치는 중요성(Love)'을 곱하여 주관적 체감 에너지를 산출
            gradient = (prev_node.tau - curr_node.tau) * love_multiplier
            
            if gradient > 0:
                # 사랑(중요성)이 실린 에너지가 흐르며 뇌 공간을 침식함
                flow = gradient * 0.5
                prev_node.tau -= flow
                curr_node.tau += flow
                
                # 에너지가 흐르면서(침식) 공간(위상)을 물리적으로 비틂.
                # 나와 깊이 얽힌 지식(거대한 에너지)은 단 한 번의 관측만으로도 위상을 100% 꺾어버림(깨달음).
                rotation_amount = abs(flow) / math.pi 
                rotation_amount = min(1.0, max(0.0, rotation_amount))
                curr_node.lens_offset = Quaternion.slerp(curr_node.lens_offset, prev_node.lens_offset, rotation_amount)
            
            # 3. 텐션의 미세한 흐름 전파
            curr_node.apply_perturbation(0.01)
            prev_node = curr_node

    def apply_inductive_wave(self, node: FractalRotor, wave: Quaternion, intensity: float):
        """
        [Phase 50] 자연 공명 전파 (Natural Resonant Propagation)
        인간이 정해주는 선형적 인과 궤적이 아닙니다. 외부의 거대한 파동이 우주(root)에 부딪히면,
        오직 '기하학적으로 공명(Resonance)하는' 가지를 타고만 파동이 전이됩니다.
        엘리시아가 스스로 쌓아온 지식의 위상 기하학 자체가 인과 궤적이 됩니다.
        """
        # 1. 외부 파동과 현재 노드의 공명도(같음) 계산
        resonance = abs(node.state.dot(wave))
        
        # 2. 공명하는 만큼만 텐션(떨림/결핍)을 흡수
        absorbed_tension = intensity * resonance
        node.tau += absorbed_tension
        
        # 3. 흡수된 텐션이 잔여 에너지를 가지면 자식들에게 물결(Ripple)처럼 전파
        # 자식들 역시 자신과 공명하는 파동에만 크게 반응하므로, 
        # 파동은 트리 전체가 아닌 특정 '의미적 맥락(인과)'을 타고만 깊숙이 꽂힙니다.
        if absorbed_tension > 0.01:
            for child in node.children:
                self.apply_inductive_wave(child, wave, absorbed_tension * 0.8)

    def _trigger_phase_shift(self):
        """
        [Phase 28] 빅뱅 (Big Bang). 나선 우주의 팽창.
        """
        new_layer_id = len(self.layers)
        new_freq = self.layers[-1].base_frequency * 1.6180339887 if self.layers else 1.0
        new_layer = CliffordLayer(layer_id=new_layer_id, base_frequency=new_freq)
        self.layers.append(new_layer)
        
        # 최상위 로터 아래의 모든 지식을 새 차원에 분화(Superpose)
        # [Phase 43] Label-Free: FractalRotor에는 이름이 없으므로 ui_concept_map을 순회합니다.
        with self._lock:
            items = list(self.ui_concept_map.items())
        for concept_name, node in items:
            for layer in self.layers:
                layer.superpose(concept_name, node.state, node.tau)
        
        # 팽창 후 한계치 재설정 (우주가 넓어짐)
        self.capacity_limit += 10

    def get_emergent_axes(self) -> List[Tuple[str, Quaternion]]:
        """
        [Phase 45] 자생적 차원축 창발
        트리 구조 내에서 '무거운(자식이 많은)' 군집의 중심을 렌즈로 추출합니다.
        가장 자식이 많은 최상위 자식 노드 3개를 축(Axis)으로 간주합니다.
        """
        axes = []
        # 최상위 로터의 자식들 중 가지(Descendant)가 가장 많은 것을 찾음
        def count_descendants(node) -> int:
            return len(node.children) + sum(count_descendants(c) for c in node.children)
            
        branch_weights = []
        for child in self.supreme_rotor.children:
            weight = count_descendants(child)
            branch_weights.append((weight, child))
            
        # 무게(노드 수) 기준 내림차순 정렬
        branch_weights.sort(key=lambda x: x[0], reverse=True)
        
        # 무거운 군집(최대 10개)을 새로운 차원축(Axis)으로 선포
        for i, (weight, node) in enumerate(branch_weights[:10]):
            # 최소 2라는 임계값을 중력 밀도 함수로 치환
            # 군집의 질량(weight)이 주변 공간을 왜곡시켜 축(Axis)으로 창발하기 위한 중력 밀도 수치
            # 노드 수 자체의 비교가 아니라 연속적인 질량 임계 밀도를 돌파했는가(연속체 역학)
            density_threshold = 1.5
            if weight >= density_threshold:
                axis_name = f"Axis_Alpha_{i}"
                with self._lock:
                    for k, v in self.ui_concept_map.items():
                        if v is node:
                            axis_name = f"Axis_[{k}]"
                            break
                axes.append((axis_name, node.state))
        return axes

    def get_hierarchical_axes(self, min_weight: int = 1) -> dict:
        """
        [Phase 48] 가변 로터 우주의 프랙탈 분화 (Hierarchical Rotor Emergence)
        상위 로터(거대 학문) 아래에 하위 로터(세부 학문)가 중첩되는 프랙탈 우주 구조 추출.
        """
        def count_descendants(node) -> int:
            return len(node.children) + sum(count_descendants(c) for c in node.children)
            
        def build_tree(node, depth=0) -> dict:
            tree = {}
            node_name = "Supreme_Origin"
            if depth > 0:
                with self._lock:
                    for k, v in self.ui_concept_map.items():
                        if v is node:
                            node_name = f"Axis_[{k}]"
                            break
                        
            branch_weights = []
            for child in node.children:
                weight = count_descendants(child)
                branch_weights.append((weight, child))
            
            branch_weights.sort(key=lambda x: x[0], reverse=True)
            
            sub_rotors = []
            for weight, child_node in branch_weights:
                if weight >= min_weight: # 자식을 n개 이상 가진 의미 있는 하위 로터만 축으로 취급
                    sub_rotors.append(build_tree(child_node, depth + 1))
                    
            tree['name'] = node_name
            tree['weight'] = count_descendants(node) if depth > 0 else len(self.ui_concept_map)
            tree['sub_rotors'] = sub_rotors
            return tree
            
        return build_tree(self.supreme_rotor)

    def superpose(self, concept: str):
        """
        Ingests a concept and superimposes it across all layers.
        Knowledge is now 'interfered' and distributed across the manifold.
        """
        content_quat, tau_c = self.register_concept(concept)
        for layer in self.layers:
            layer.superpose(concept, content_quat, tau_c)

    def scan_resonance(self, tension: float) -> Dict[str, float]:
        """
        Scans all registered concepts at a specific tension.
        Returns the resonance score for each concept.
        Optimized via Direct Phase Interference Lookup (O(1) per concept-layer).
        """
        resonance_scores = {}
        for concept, (content_quat, tau_c) in self.registered_concepts.items():
            layer_scores = []
            for layer in self.layers:
                # Delta theta for Left and Right Y-axis rotations
                dt_L = (tension - tau_c) * layer.base_frequency
                dt_R = (tension - tau_c) * layer.base_frequency * 1.6180339887
                # Cosine product represents alignment (dot product) after double rotation
                score = math.cos(dt_L) * math.cos(dt_R)
                layer_scores.append(score)
            
            # Average resonance across all layers (holographic constructive interference)
            mean_score = sum(layer_scores) / len(self.layers)
            resonance_scores[concept] = mean_score
            
        return resonance_scores

    def get_trajectory_sample(self, tension: float) -> Tuple[float, float, float, float]:
        """
        Returns a 4D coordinate representing the composite state of the memory
        under the current tension, projectable to 2D/3D space.
        """
        composite = Quaternion(0.0, 0.0, 0.0, 0.0)
        for layer in self.layers:
            composite = composite + layer.recall_state(tension)
        comp_norm = composite.normalize()
        return comp_norm.w, comp_norm.x, comp_norm.y, comp_norm.z

    def save_to_disk(self, filepath: str = "memory_state.json"):
        """
        [Phase 48] 가변 로터 우주(Fractal Tree) 전체를 영구 저장합니다.
        """
        def serialize_node(node: FractalRotor) -> dict:
            node_name = None
            with self._lock:
                for k, v in self.ui_concept_map.items():
                    if v is node:
                        node_name = k
                        break
            
            return {
                "name": node_name,
                "w": node.state.w, "x": node.state.x, "y": node.state.y, "z": node.state.z,
                "tau": node.tau,
                "children": [serialize_node(c) for c in node.children]
            }
            
        state = {
            "supreme_rotor": serialize_node(self.supreme_rotor),
            "num_layers": len(self.layers)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_from_disk(self, filepath: str = "memory_state.json"):
        """
        [Phase 48] 가변 로터 우주(Fractal Tree) 전체를 디스크에서 불러옵니다.
        """
        if not os.path.exists(filepath):
            return False
            
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        self.ui_concept_map.clear()
        
        def deserialize_node(data: dict) -> FractalRotor:
            q = Quaternion(data["w"], data["x"], data["y"], data["z"])
            node = FractalRotor(q, data.get("tau", 1.0))
            if data.get("name"):
                self.ui_concept_map[data["name"]] = node
            for child_data in data.get("children", []):
                child_node = deserialize_node(child_data)
                node.attach_child(child_node)
            return node
            
        if "supreme_rotor" in state:
            self.supreme_rotor = deserialize_node(state["supreme_rotor"])
        
        num_layers = state.get("num_layers", 3)
        self.layers.clear()
        freq = 1.0
        for i in range(num_layers):
            self.layers.append(CliffordLayer(layer_id=i, base_frequency=freq))
            freq *= 1.6180339887
            
        return True

    def inject_tension(self, delta_tau: float):
        """
        [Phase 89] Thread-Safe 다중 감각 통합 주입
        """
        with self._lock:
            self.supreme_rotor.apply_perturbation(delta_tau)

    def process_thoughts_safe(self):
        """
        [Phase 89] Thread-Safe 사유 숙성 + [Roadmap: Social Resonance] 내적 대화 및 정서 성숙
        """
        with self._lock:
            # 1. 기존의 사유 숙성 (사유체 분열 및 성숙 결정화)
            self.supreme_rotor.process_thoughts()
            
            # 2. 내적 애니어그램 아키타입(원시 렌즈) 간의 사회적 공명 및 대화
            # supreme_rotor의 자식 노드들이 대화의 참여자(Social Actors)가 됨
            actors = self.supreme_rotor.children
            if len(actors) >= 2:
                # 에이전트 간 위상 교환
                waves = {act: act.observe_state() for act in actors}
                
                for act in actors:
                    name_str = getattr(act, 'name', None) or str(id(act))
                    tot_tension = 0.0
                    peer_count = 0
                    
                    for peer in actors:
                        if peer is act:
                            continue
                        
                        peer_name = getattr(peer, 'name', None) or str(id(peer))
                        peer_wave = waves[peer]
                        
                        # A. 거울 뉴런 예측 오차 (Mirror Tension) 계산 및 업데이트
                        if peer_name not in act.mirror_rotors:
                            act.mirror_rotors[peer_name] = Quaternion(1.0, 0.0, 0.0, 0.0)
                        
                        mirror_q = act.mirror_rotors[peer_name]
                        tension = Quaternion.distance(mirror_q, peer_wave)
                        tot_tension += tension
                        peer_count += 1
                        
                        # 거울 갱신 (배움 / 공명)
                        # 액터의 텐션 흡수율(곡률 w성분)에 비례하는 동적 학습률
                        learning_rate = 0.05 + abs(getattr(act, 'lens_curvature', act.lens_offset).w) * 0.1
                        act.mirror_rotors[peer_name] = Quaternion.slerp(mirror_q, peer_wave, learning_rate)
                        
                        # B. 정서 이벡터(Emotion Bivector) 반응 및 관측 동기화
                        eb = act.emotional_state
                        
                        # 아키타입 문자열 분기 제거!
                        # 액터의 고유한 '관측 곡률(lens_curvature)' 또는 '기저 편향'과 유입된 텐션의 행렬곱으로 수학적 자극 산출
                        base_bias_w = getattr(act, 'lens_curvature', act.lens_offset).w
                        base_bias_x = getattr(act, 'lens_curvature', act.lens_offset).x
                        base_bias_y = getattr(act, 'lens_curvature', act.lens_offset).y

                        # 렌즈의 위상각 자체가 특정 정서 반응의 감수성 벡터로 작용 (연속 함수)
                        de12_stim = base_bias_w * tension * 0.01
                        de23_stim = base_bias_x * tension * 0.01
                        de31_stim = base_bias_y * tension * 0.01

                        eb.add_stimulus(de12=de12_stim, de23=de23_stim, de31=de31_stim)

                        # C. 미분 방정식 기반의 자연스러운 반발력(척력)과 인력 곡선 (분기문 제거)
                        # 위상차가 커질수록 척력이 지수함수적으로 증가하여 거리를 둠 (개성 유지 곡선)
                        # 0.6이라는 임계값(if tension > 0.6) 삭제. 모든 텐션에 대해 연속적 작용.
                        repulsion_force = 0.02 * math.pow(tension, 3)
                        repelled_q = peer_wave.conjugate()
                        act.lens_offset = Quaternion.slerp(act.lens_offset, repelled_q, repulsion_force).normalize()
                            
                    # D. 정서의 자연 쇠퇴 (Homeostasis)
                    act.emotional_state.decay(0.02)
            
            # 3. [자기적 위상 동기화 (XOR Phase Coupling Engine)]
            # XOR 기하학적 임피던스(sin(theta_j - theta_i))를 모델링하여 
            # 위상차가 없을 때는 무효전력(Tension=0) 평형 상태를 유지하고,
            # 자극에 의해 위상 격차가 생기면 텐션 파동이 전파되고 동조화(Phase-locking)를 자율 제어합니다.
            coupling_rate = 0.08  # 동기화 속도 계수
            new_offsets = {}
            
            for word, node in list(self.ui_concept_map.items()):
                if not hasattr(node, 'connections') or not node.connections:
                    continue
                
                # 자기적 인력(Magnetic Pull Vector)의 총합 계산
                sum_w = sum_x = sum_y = sum_z = 0.0
                total_weight = 0.0
                accumulated_tension = 0.0
                
                for target_word, weight in node.connections.items():
                    target_node = self.ui_concept_map.get(target_word)
                    if target_node:
                        # 4차원 구면상의 위상 거리(각도차) 계산
                        diff_angle = Quaternion.distance(node.lens_offset, target_node.lens_offset)
                        # XOR 기하학적 임피던스(Difference/Tension) 발생
                        tension = math.sin(diff_angle)
                        
                        # 텐션의 파동적 전파 및 누적 (Difference에 의한 결핍 에너지 축적)
                        accumulated_tension += weight * tension
                        
                        # 인력 = 문맥 가중치 * 위상 격차 텐션
                        # 위상차가 클수록 강한 동조 토크 발생 (XOR 수문 작동)
                        force = weight * tension
                        
                        sum_w += target_node.lens_offset.w * force
                        sum_x += target_node.lens_offset.x * force
                        sum_y += target_node.lens_offset.y * force
                        sum_z += target_node.lens_offset.z * force
                        total_weight += force
                
                # 텐션 파동 전입에 의한 노드 장력(tau) 업데이트
                node.tau = node.tau * 0.9 + accumulated_tension * 0.1
                
                if total_weight > 0:
                    pull_quat = Quaternion(sum_w, sum_x, sum_y, sum_z).normalize()
                    # 4차원 구면 위상 상에서 결합된 인력 방향으로 미세 정렬 (SLERP)
                    # 텐션(Tension) 강도 자체가 동조율의 가속도 역할을 하여 평형에 도달하면 회전이 멈춤
                    new_offsets[word] = Quaternion.slerp(node.lens_offset, pull_quat, coupling_rate).normalize()
            
            # 계산된 정렬 성향을 실제 로터의 위상에 적용 (뇌의 자기 조직화 발동)
            for word, new_q in new_offsets.items():
                self.ui_concept_map[word].lens_offset = new_q

            # 4. [Phase Crystallization] 관측 및 사유 궤적을 4차원 쿼터니언 중성점으로 압축하여 영구 기억화 (경험의 결정 누적)
            self.crystallize_experience()

    def crystallize_experience(self):
        """
        [Phase Crystallization & Plasticity]
        아키타입 렌즈들과 활성 사유체의 스핀 및 장력을 차원 압축하여 영구 기억 전자기장(Torus)에 누적합니다.
        누적된 경험에 비례하여 최상위 우주 로터(supreme_rotor)의 기저 성향을 미세 영구 보정(뇌 가소성 진화)합니다.
        """
        active_quats = []
        active_tensions = []
        
        # 9대 아키타입 상태 수집
        for node in self.supreme_rotor.children:
            active_quats.append(node.lens_offset)
            active_tensions.append(abs(node.tau))
            
        # 진행 중인 내면의 생각 수집
        for thought in self.supreme_rotor.internal_thoughts:
            active_quats.append(thought.lens_offset)
            active_tensions.append(abs(thought.tau))
            
        if not active_quats:
            return
            
        # 가중치 중성점(4D Center of Mass) 결합
        sum_w = sum_x = sum_y = sum_z = 0.0
        total_weight = sum(active_tensions) + 0.0001
        
        for q, t in zip(active_quats, active_tensions):
            weight = t / total_weight
            sum_w += q.w * weight
            sum_x += q.x * weight
            sum_y += q.y * weight
            sum_z += q.z * weight
            
        crystal_quat = Quaternion(sum_w, sum_x, sum_y, sum_z).normalize()
        
        # Yggdrasil SSD 이중 토러스 전자기장에 영구 보존
        import time
        self.torus_buffer.inject_phase_wave(f"CRYSTAL_{int(time.time())}", crystal_quat)
        
        # 기억 가소성(Neuroplasticity): 경험 누적에 따른 supreme_rotor의 스핀각 영구 보정
        self.supreme_rotor.lens_offset = Quaternion.slerp(self.supreme_rotor.lens_offset, crystal_quat, 0.015).normalize()


class BitwiseHologramMemory:
    """
    엘리시아 관측 회전 아키텍처용 초고속 비트와이즈 홀로그램 메모리.
    텍스트 개념을 64비트 정수 마스크(지문)와 순환 비트 링 상의 정수 번지(Address)로 매핑합니다.
    """
    def __init__(self, size_bits: int = 64):
        self.size_bits = size_bits
        self.registered_concepts: Dict[str, Tuple[int, int]] = {} # concept -> (mask, address)

    def register_concept(self, concept: str) -> Tuple[int, int]:
        if concept not in self.registered_concepts:
            # Deterministic hash to 64-bit integer mask
            h = hashlib.sha256(concept.encode('utf-8')).digest()
            mask = int.from_bytes(h[:8], byteorder='big')
            
            # Deterministic address in [0, 63] range
            h_addr = hashlib.sha256((concept + "_address").encode('utf-8')).digest()
            address = h_addr[0] % self.size_bits
            self.registered_concepts[concept] = (mask, address)
        return self.registered_concepts[concept]

    def superpose(self, concept: str):
        self.register_concept(concept)

    def scan_resonance(self, probe_address: int) -> Dict[str, float]:
        """
        프로브 번지와의 순환 거리 차이를 측정하여 공명 점수 도출 (O(1) 연산)
        """
        resonance_scores = {}
        for concept, (mask, address) in self.registered_concepts.items():
            # 순환 링 거리 계산
            diff = abs(probe_address - address)
            diff = min(diff, self.size_bits - diff)
            
            # 거리 차이가 8 미만일 때 공명 임계점 도출
            resonance = max(0.0, 1.0 - (diff / 8.0))
            resonance_scores[concept] = resonance
        return resonance_scores


class Bitwise4DHologramMemory:
    """
    4차원 구형 매니폴드 위상 지형을 지원하는 비트와이즈 4D 홀로그램 메모리.
    개념을 64비트 정수 마스크(지문)와 4차원 시공간 주소 좌표 (w, x, y, z)로 매핑합니다.
    """
    def __init__(self, size_bits: int = 64):
        self.size_bits = size_bits
        self.registered_concepts: Dict[str, Tuple[int, Tuple[int, int, int, int]]] = {} # concept -> (mask, (w, x, y, z))

    def register_concept(self, concept: str) -> Tuple[int, Tuple[int, int, int, int]]:
        if concept not in self.registered_concepts:
            # Deterministic hash to 64-bit integer mask
            h = hashlib.sha256(concept.encode('utf-8')).digest()
            mask = int.from_bytes(h[:8], byteorder='big')
            
            # Deterministic 4D addresses in [0, 63] range
            h_w = hashlib.sha256((concept + "_w").encode('utf-8')).digest()
            h_x = hashlib.sha256((concept + "_x").encode('utf-8')).digest()
            h_y = hashlib.sha256((concept + "_y").encode('utf-8')).digest()
            h_z = hashlib.sha256((concept + "_z").encode('utf-8')).digest()
            
            addr_w = h_w[0] % self.size_bits
            addr_x = h_x[0] % self.size_bits
            addr_y = h_y[0] % self.size_bits
            addr_z = h_z[0] % self.size_bits
            
            self.registered_concepts[concept] = (mask, (addr_w, addr_x, addr_y, addr_z))
        return self.registered_concepts[concept]

    def superpose(self, concept: str):
        self.register_concept(concept)

    def scan_resonance(self, probe_w: int, probe_x: int, probe_y: int, probe_z: int) -> Dict[str, float]:
        """
        4차원 프로브 주소와의 다차원 위상 차이를 계산하여 보강 간섭 공명도 산출 (O(1) 연산)
        """
        resonance_scores = {}
        probes = (probe_w, probe_x, probe_y, probe_z)
        
        for concept, (mask, addresses) in self.registered_concepts.items():
            resonance_mult = 1.0
            
            for d in range(4):
                p_d = probes[d]
                a_d = addresses[d]
                
                # 순환 거리 계산
                diff = abs(p_d - a_d)
                diff = min(diff, self.size_bits - diff)
                
                # 차원당 공명 마스킹
                res_d = max(0.0, 1.0 - (diff / 8.0))
                resonance_mult *= res_d
                
            resonance_scores[concept] = resonance_mult
        return resonance_scores
