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
from core.math_utils import Quaternion, traverse_causal_trajectory
from core.fractal_rotor import FractalRotor

def concept_to_quaternion(concept: str) -> Quaternion:
    """
    [Phase 51] 문자열(개념)을 해시 함수로 뭉개지 않고, 
    순차적인 위상 궤적(Biological Trajectory)으로 변환합니다.
    """
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
        
        # 인간의 언어(UI)와 로터를 매핑하기 위한 최외곽 껍질 딕셔너리
        self.ui_concept_map: Dict[str, FractalRotor] = {}

    @property
    def registered_concepts(self) -> Dict[str, Tuple[Quaternion, float]]:
        """
        하위 호환성을 위해 최외곽 매핑 딕셔너리를 평면화하여 반환합니다.
        """
        flat_dict = {}
        for concept, node in self.ui_concept_map.items():
            flat_dict[concept] = (node.state, node.tau)
        return flat_dict

    @property
    def folded_dimensions(self) -> Dict[str, Dict]:
        # 하위 호환성을 위한 더미 (Phase 28에서는 트리의 잎사귀들이 접힌 차원 역할을 함)
        return {}

    def register_concept(self, concept: str) -> Tuple[Quaternion, float]:
        """
        Generates and registers a canonical 4D content key and its resonant tension address.
        """
        existing = self.registered_concepts.get(concept)
        if existing:
            return existing

        content_quat = concept_to_quaternion(concept)
        
        # Generate deterministic address tension in range [1.0, 9.0]
        # Uses concept name salt to maintain causality
        h = hashlib.sha256((concept + "_address").encode('utf-8')).digest()
        tau_c = 1.0 + ((h[0] ^ h[2]) + (h[1] ^ h[3]) * 256) / 65535.0 * 8.0
        
        new_node = FractalRotor(content_quat, tau_c)
        self.supreme_rotor.attach_child(new_node)
        self.ui_concept_map[concept] = new_node
        
        return (content_quat, tau_c)

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
        
        # [Phase 47] 만약 찾은 로터와의 공명도(resonance)가 일정 수치(0.5) 미만이라면, 
        # 이는 기존의 어떠한 군집과도 다른 완전히 새로운 축(다름)을 의미하므로, 
        # 억지로 다른 자식에 붙이지 않고 최상위 우주(supreme_rotor)에서 직접 새로운 가지를 뻗습니다.
        if resonance < 0.5:
            resonant_parent = self.supreme_rotor
            
        new_node = FractalRotor(target_rotor, tau_c)
        self.ui_concept_map[concept] = new_node
        self.folded_dimensions[concept] = {
            "rotor": new_node.state,
            "tau_c": tau_c
        }
        
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
        for concept_name, node in self.ui_concept_map.items():
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
            if weight >= 2: # 최소 2개의 자식(본인 포함 3개 노드)을 가진 군집만 과목으로 인정
                # 축의 이름은 인간이 정하지 않음. 노드의 원시 파동 헥스 코드로 스스로 명명.
                # (테스트 출력을 위해 UI 맵에서 역산출 시도, 없으면 헥스)
                axis_name = f"Axis_Alpha_{i}"
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
