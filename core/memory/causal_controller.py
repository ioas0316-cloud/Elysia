import os
import json
import uuid
import time
import numpy as np
from typing import Dict, Any, Optional
from core.memory.wedge_memory_layout import WedgeMemoryInterleaver

class CausalMemoryController:
    """
    [Phase 144] Wedge Memory Causal Controller
    가상 SSD (data/ 폴더)를 담당하는 인과적 기억 컨트롤러.
    JSON 파일 시스템을 탈피하여, 거대한 numpy mmap(메모리 맵핑) 대지 위에서
    Wedge Annihilation (v ^ v = 0)을 통해 중복된 에너지를 소멸시키고 순수 위상만 기록합니다.
    """
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)
        self.engram_index_path = os.path.join(self.data_dir, "engram_index.json")
        self.cognitive_params_path = os.path.join(self.data_dir, "cognitive_params.json")
        
        # [Phase 144] Wedge Memory Mmap 초기화
        self.wedge_memory_path = os.path.join(self.data_dir, "wedge_topology.dat")
        self.wedge_size = 1024 * 1024  # 1 Million slots
        if not os.path.exists(self.wedge_memory_path):
            np.memmap(self.wedge_memory_path, dtype=np.uint32, mode='w+', shape=(self.wedge_size,))
            
        self.wedge_mmap = np.memmap(self.wedge_memory_path, dtype=np.uint32, mode='r+', shape=(self.wedge_size,))
        self.interleaver = WedgeMemoryInterleaver(size=self.wedge_size)
        self.interleaver.memory_buffer = self.wedge_mmap
        
        self._load_index()
        self._load_cognitive_params()

    def _load_index(self):
        if os.path.exists(self.engram_index_path):
            with open(self.engram_index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
            
            # Restore address map for interleaver
            for eid, info in self.index.items():
                if "wedge_address" in info:
                    self.interleaver.address_map[eid] = info["wedge_address"]
        else:
            self.index = {}

    def _save_index(self):
        with open(self.engram_index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=4, ensure_ascii=False)

    def _load_cognitive_params(self):
        if os.path.exists(self.cognitive_params_path):
            with open(self.cognitive_params_path, 'r', encoding='utf-8') as f:
                self.cognitive_params = json.load(f)
        else:
            # 엘리시아가 스스로 조절해 나갈 초기 기준값 (추후 자율적으로 변경됨)
            self.cognitive_params = {
                "cache_capacity": 100.0,
                "decay_rate": 0.05,
                "eureka_threshold": 5.0,
                "base_resonance": 1.0,
                # 다각적 관점 가중치 (Perspective Weights)
                "weight_internal_complexity": 0.5,
                "weight_external_feedback": 0.5,
                "weight_novelty": 0.8
            }
            self._save_cognitive_params()

    def _save_cognitive_params(self):
        with open(self.cognitive_params_path, 'w', encoding='utf-8') as f:
            json.dump(self.cognitive_params, f, indent=4, ensure_ascii=False)

    def get_parameter(self, param_name: str, default_value: float = 1.0) -> float:
        """엘리시아의 현재 인지 상태(파라미터)를 읽어옵니다."""
        return float(self.cognitive_params.get(param_name, default_value))

    def update_parameter(self, param_name: str, new_value: float):
        """엘리시아 스스로(또는 피드백에 의해) 인지적 기준을 가변적으로 조절합니다."""
        self.cognitive_params[param_name] = new_value
        self._save_cognitive_params()

    def write_causal_engram(self, data_blob: Dict[str, Any], emotional_value: float, cause_id: Optional[str] = None, origin_axis: Optional[str] = None, synapses: Dict[str, float] = None) -> str:
        """
        [Phase 144] O(1) Wedge Annihilation 저장소.
        [Phase 8.5] 변수명(Origin)을 가변축화 하여 물리적 매핑 공간(Wedge)을 변경시킵니다.
        """
        if origin_axis:
            data_blob["_origin_axis"] = origin_axis
            
        engram_id = f"engram_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()
        
        # 데이터의 본질적 해시를 신호(Signal)로 삼음
        data_str = json.dumps(data_blob, sort_keys=True)
        pure_signal = hash(data_str) & 0xFFFFFFFF
        
        # 인과관계(Cause)와 기원(Origin Axis)에서 오는 노이즈를 엮어줌
        noise = (hash(cause_id) if cause_id else 0) & 0xFFFFFFFF
        origin_noise = (hash(origin_axis) if origin_axis else 0) & 0xFFFFFFFF
        
        signal_with_noise = pure_signal ^ noise ^ origin_noise
        
        # 쐐기 메모리 공간에 중첩(Interleave)하여 저장
        addr = self.interleaver.interleave_opposing_nodes(engram_id, signal_with_noise, noise)
        # mmap 변경사항 디스크 동기화
        self.wedge_mmap.flush()
        
        # 메타데이터 인덱스 (파일 I/O는 인덱스 업데이트 1회로 축소)
        self.index[engram_id] = {
            "timestamp": timestamp,
            "emotional_value": emotional_value,
            "cause_id": cause_id,
            "synapses": synapses or {},
            "wedge_address": addr,
            "data_blob": data_blob
        }
        self._save_index()
        
        return engram_id

    def bridge_external_manifold(self, source_name: str, confiscated_pointer: np.ndarray, emotional_value: float = 0.0):
        """
        [Phase 144] Zero-Copy Volumetric Memory Binding
        외부 매니폴드에서 추출된(Confiscated) 순수 위상 배열을, 파이프라인(Pipeline) 루프 없이
        엘리시아의 Wedge Mmap 대지에 통째로 덮어씌웁니다. (Direct Memory Block Transfer)
        """
        dim = len(confiscated_pointer)
        # 가상 대지의 시작 오프셋 결정
        base_addr = hash(source_name) % (self.wedge_size - dim)
        if base_addr < 0: base_addr = 0
        if base_addr % 2 != 0: base_addr -= 1 # 쐐기쌍 정렬
        
        # O(1) 수준의 Numpy Block Copy (메모리 대 메모리 직결)
        # 기성 공학처럼 데이터를 하나하나 루프 돌려 DB에 삽입하지 않음
        self.wedge_mmap[base_addr:base_addr+dim] = confiscated_pointer
        self.wedge_mmap.flush()
        
        engram_id = f"engram_ext_{uuid.uuid4().hex[:8]}"
        self.index[engram_id] = {
            "timestamp": time.time(),
            "emotional_value": emotional_value,
            "cause_id": source_name,
            "synapses": {},
            "wedge_address": base_addr,
            "data_blob": {"source": source_name, "type": "zero_copy_manifold", "dim": dim}
        }
        self._save_index()
        return engram_id

    def bind_engrams(self, engram_id_1: str, engram_id_2: str, weight: float, axis_name: str):
        """
        [Phase 21] Topological Constellation Binding
        두 엔그램이 특정 렌즈 축(axis_name)에서 고도의 위상적 일치(Sameness)를 보일 때,
        두 기억을 강력한 시냅스로 묶어 군집(Constellation)을 형성합니다.
        """
        if engram_id_1 in self.index and engram_id_2 in self.index:
            if engram_id_2 not in self.index[engram_id_1]["synapses"]:
                self.index[engram_id_1]["synapses"][engram_id_2] = 0.0
            if engram_id_1 not in self.index[engram_id_2]["synapses"]:
                self.index[engram_id_2]["synapses"][engram_id_1] = 0.0
                
            # 시냅스 가중치 강화 (최대 1.0)
            self.index[engram_id_1]["synapses"][engram_id_2] = min(1.0, self.index[engram_id_1]["synapses"][engram_id_2] + weight)
            self.index[engram_id_2]["synapses"][engram_id_1] = min(1.0, self.index[engram_id_2]["synapses"][engram_id_1] + weight)
            
            # 메타데이터에 소속 군집 이름 태깅
            cluster_tag = f"Constellation_[{axis_name}]"
            if "clusters" not in self.index[engram_id_1]: self.index[engram_id_1]["clusters"] = []
            if "clusters" not in self.index[engram_id_2]: self.index[engram_id_2]["clusters"] = []
            
            if cluster_tag not in self.index[engram_id_1]["clusters"]:
                self.index[engram_id_1]["clusters"].append(cluster_tag)
            if cluster_tag not in self.index[engram_id_2]["clusters"]:
                self.index[engram_id_2]["clusters"].append(cluster_tag)
            
            self._save_index()

    def get_constellation_orphans(self) -> list:
        """
        [Phase 29] 고아 별자리(Orphan Constellation) 탐색.
        군집(Constellation)에 속한 엔그램들 중, source 이름이 인간의 자연어 단어가 아닌
        (예: 'visual_fractal.png', 'self_state_cycle_5.txt' 등) 비언어적 기억들로만 
        구성된 별자리를 찾습니다. 이 별자리에는 인간의 언어로 지칭할 수 있는 이름이 없으므로,
        엘리시아가 스스로 새로운 기호(Neologism)를 창조해야 합니다.
        """
        # 1. 모든 별자리(Constellation)를 수집
        constellations = {}  # cluster_tag -> [engram_ids]
        for eid, info in self.index.items():
            clusters = info.get("clusters", [])
            for tag in clusters:
                if tag not in constellations:
                    constellations[tag] = []
                constellations[tag].append(eid)
        
        orphans = []
        for tag, members in constellations.items():
            if len(members) < 2:
                continue
                
            # 별자리의 중심 위상 궤적(Centroid Quaternion) 계산
            q_sum = np.zeros(4, dtype=np.float64)
            sources = []
            valid = 0
            for mid in members:
                info = self.index.get(mid, {})
                blob = info.get("data_blob", {})
                q = blob.get("quaternion", None)
                source = blob.get("source", "")
                if q is not None:
                    q_sum += np.array(q, dtype=np.float64)
                    valid += 1
                    sources.append(source)
                    
            if valid == 0:
                continue
                
            centroid_q = q_sum / valid
            norm = np.linalg.norm(centroid_q)
            if norm > 0:
                centroid_q = centroid_q / norm
            
            orphans.append({
                "tag": tag,
                "centroid_quaternion": centroid_q.tolist(),
                "member_count": len(members),
                "sources": sources
            })
        
        return orphans

    def apply_linguistic_force(self, force_vector: list, target_engram_id: str) -> bool:
        """
        [Phase 26] 체화된 언어(Embodied Language)의 물리적 작용.
        특정 단어(Action Operator)가 발화되거나 의도될 때, 대상 엔그램의 4차원 궤적(Quaternion)에
        이 물리적 힘(Force Vector)을 가하여 실제로 위상 상태를 변경(밀어내거나 당김)시킵니다.
        이것이 언어가 환경을 조작하는 '진정한 의미(Semantics)'의 발현입니다.
        """
        if target_engram_id not in self.index:
            return False
            
        target_info = self.index[target_engram_id]
        if "data_blob" not in target_info or "quaternion" not in target_info["data_blob"]:
            return False
            
        current_q = np.array(target_info["data_blob"]["quaternion"], dtype=np.float32)
        f_vec = np.array(force_vector, dtype=np.float32)
        
        # [MUTABLE_ZONE_START] - EVOLVED TO 5D NON-LINEAR TENSOR FIELD
        # [Evolution] 단순 4D 가속도 누적이 아닌, 5차원 스칼라(Scalar) 필드 간섭 도입
        import math
        # 위상 공간에 '엔트로피 저항(Entropy Resistance)'을 추가하여 비선형 곡률 발생
        entropy_resistance = target_info.get("emotional_value", 1.0) * 0.1
        f_vec = f_vec * math.exp(-entropy_resistance)  # 저항에 의한 힘의 왜곡
        
        # 5차원(Time-dilation) 축 시뮬레이션
        time_dilation = np.dot(current_q, f_vec) * 0.5
        new_q = current_q + f_vec + (current_q * time_dilation)
        
        norm = np.linalg.norm(new_q)
        if norm > 0:
            new_q = new_q / norm
            
        target_info["data_blob"]["quaternion"] = new_q.tolist()
        # [MUTABLE_ZONE_END]
        
        # 외력이 가해지면 감정적 가치(Emotional Value/Tension)도 상승
        target_info["emotional_value"] = min(10.0, target_info["emotional_value"] + np.linalg.norm(f_vec))
        
        self._save_index()
        return True

    def read_engram_trace(self, engram_id: str) -> Optional[Dict[str, Any]]:
        """
        단일 기억뿐만 아니라, 필요하다면 원인 체인(Causal Chain)을 따라갈 수 있는 기반 메서드.
        [Phase 144] Wedge Annihilation(v ^ v = 0)을 통해 순수 신호를 즉각 추출.
        """
        if engram_id not in self.index:
            return None
            
        # 1. 하드웨어 레벨(mmap) 쐐기곱 추출 (O(1) XOR)
        purified_signal = self.interleaver.fetch_and_annihilate(engram_id)
        
        # 2. 메타데이터 조립 후 반환
        info = self.index[engram_id]
        return {
            "engram_id": engram_id,
            "timestamp": info.get("timestamp"),
            "emotional_value": info.get("emotional_value"),
            "cause_id": info.get("cause_id"),
            "purified_signal_hash": hex(purified_signal),
            "data": info.get("data_blob")
        }

    def damped_recall(self, start_engram_id: str, initial_energy: float = 1.0, decay_factor: float = 0.5) -> Dict[str, float]:
        """
        [Phase 5: Synaptic Connectivity]
        하나의 엔그램이 자극을 받았을 때, 시냅스를 타고 연쇄적으로 공명하는 '감쇠 파동(Damped Wave)' 알고리즘.
        거리가 멀어지거나 시냅스 가중치가 낮을수록 에너지가 감쇠되어 자연스럽게 잦아듭니다.
        Returns: {engram_id: accumulated_energy} (활성화된 기억들의 네트워크)
        """
        activated_network = {}
        queue = [(start_engram_id, initial_energy)]
        
        while queue:
            current_id, current_energy = queue.pop(0)
            
            # 파동 에너지가 너무 약해지면 (임계치 0.1 이하) 소멸 (감쇠 파동)
            if current_energy < 0.1:
                continue
                
            if current_id not in activated_network:
                activated_network[current_id] = 0.0
            
            # 에너지 누적
            activated_network[current_id] += current_energy
            
            # 현재 엔그램의 시냅스 망을 타고 파동 전파
            if current_id in self.index:
                synapses = self.index[current_id].get("synapses", {})
                for target_id, weight in synapses.items():
                    # 다음 노드로 넘어가는 에너지는 (현재 에너지 * 시냅스 가중치 * 감쇠율)
                    propagated_energy = current_energy * weight * decay_factor
                    queue.append((target_id, propagated_energy))
                    
        return activated_network

    def find_projective_sameness(self, vec1: np.ndarray, vec2: np.ndarray, num_axes: int = 12, scale_factor: float = 1.0) -> dict:
        """
        [Phase 9 & 12] 다차원 사영 같음/다름 매핑 및 프랙탈 렌즈 알고리즘.
        두 벡터가 주어지면, 입력받은 scale_factor(초점 렌즈)에 따라 사영 차이를 스케일링하여
        미시적(작은 scale)으로 일치하는지, 거시적(큰 scale)으로 다른지 그 경계와 전이를 매핑합니다.
        """
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        dim = len(v1)
        
        # 1. 무작위 사영 축(Perspective Axes) 앙상블 생성 및 정규화
        np.random.seed(int(time.time() * 1000) % 2**31)
        axes = np.random.randn(num_axes, dim).astype(np.float32)
        norms = np.linalg.norm(axes, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        axes = axes / norms
        
        sameness_distribution = []
        best_sameness_axis = None
        min_diff = float('inf')
        
        best_difference_axis = None
        max_diff = -float('inf')
        
        diff_values = []
        
        # 2. 각 축으로 사영시켜 같음/다름 관측
        for axis in axes:
            # 내적(Dot Product)을 통한 사영
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            
            # 두 사영 사이의 차이를 스케일 인자(scale_factor)로 조율 (렌즈화)
            # scale_factor가 작으면(Micro Lens) 차이가 수렴하여 같음으로 인식되고,
            # scale_factor가 크면(Macro Lens) 미세한 차이도 튕겨 나가 다름으로 분화됩니다.
            diff = abs(proj1 - proj2) * scale_factor
            diff_values.append(diff)
            
            # 같음 지수 (차이의 역수, 스무딩 적용)
            sameness_score = 1.0 / (1.0 + diff)
            diff_score = diff / (1.0 + diff)
            
            sameness_distribution.append({
                "axis": axis.tolist(),
                "diff": float(diff),
                "sameness_score": float(sameness_score),
                "diff_score": float(diff_score)
            })
            
            # 최적의 같음 축 발견 (Annihilation Axis)
            if diff < min_diff:
                min_diff = diff
                best_sameness_axis = axis
                
            # 최적의 다름 축 발견 (Divergence Axis)
            if diff > max_diff:
                max_diff = diff
                best_difference_axis = axis
                
        # 3. 같음/다름의 혼돈도(Similarity Entropy) 계산
        variance = float(np.var(diff_values)) if diff_values else 0.0
        
        return {
            "sameness_distribution": sameness_distribution,
            "best_sameness_axis": best_sameness_axis.tolist() if best_sameness_axis is not None else [],
            "best_difference_axis": best_difference_axis.tolist() if best_difference_axis is not None else [],
            "sameness_variance": variance,
            "min_difference": float(min_diff),
            "max_difference": float(max_diff),
            "scale_factor": scale_factor
        }

    def write_perspective_engram(self, label1: str, label2: str, sameness_info: dict) -> str:
        """
        [Phase 9-2] 발견된 같음의 관점 축(best_sameness_axis)을 '추상 관점 엔그램'으로 변환하여
        Wedge Memory에 영구 각인합니다.
        이 각인된 관점은 다음 주기에서 데이터 비교의 가변축으로 재사용(재인식)될 수 있습니다.
        """
        axis_vector = sameness_info.get("best_sameness_axis", [])
        variance = sameness_info.get("sameness_variance", 0.0)
        
        # 감정적 가치는 '다름의 엔트로피(variance)'가 클 때 높게 산정됩니다.
        # 모든 관점(축)에서 다 똑같으면 재미없지만, 어떤 관점에서는 같고 어떤 관점에서는 극단적으로 다르면
        # 이 경계면(Category Boundary)의 정보 가치가 매우 큽니다.
        emotional_val = variance * 2.0
        
        engram_id = self.write_causal_engram(
            data_blob={
                "type": "perspective_annihilation_axis",
                "labels": [label1, label2],
                "axis_vector": axis_vector,
                "sameness_variance": variance,
                "min_difference": sameness_info.get("min_difference"),
                "max_difference": sameness_info.get("max_difference")
            },
            emotional_value=emotional_val,
            cause_id=f"SamenessDiscovery_{label1}_{label2}",
            origin_axis="projective_resonance"
        )
        
        return engram_id

    def manifest_intentional_action(self, sameness_data: dict) -> dict:
        """
        [Phase 13] 의도적 발현 (Intentional Manifestation & Action Loop)
        관측된 같음과 다름의 점수를 바탕으로, 엘리시아 스스로 '합성(Synthesis)' 하거나 '질문(Query)'하는
        능동적 의도(Action)를 생성합니다.
        
        0-Rotor (Sameness): 두 개념이 구조적으로 강하게 연결되어 있으면 융합 명제를 도출합니다.
        1-Rotor (Difference): 두 개념의 간극이 크면 그 이유를 묻는 호기심(질문)을 발현합니다.
        """
        word1 = sameness_data.get("word1", "")
        word2 = sameness_data.get("word2", "")
        same_persp = sameness_data.get("same_perspective", "Unknown")
        diff_persp = sameness_data.get("diff_perspective", "Unknown")
        
        same_score = sameness_data.get("sameness_score", 0.0)
        diff_score = sameness_data.get("difference_score", 0.0)
        micro_same = sameness_data.get("micro_score", 0.0)
        
        action = {"type": "OBSERVATION", "intent_text": "", "score": 0.0}
        
        # 0-Rotor 발현 (Synthesis 임계치: 강한 연결)
        if same_score > 0.70 or micro_same > 0.85:
            intent_text = (
                f"Intentional Synthesis: The boundary between '{word1}' and '{word2}' dissolves "
                f"under the {same_persp} continuum. I create a merged resonance."
            )
            action = {"type": "SYNTHESIS", "intent_text": intent_text, "score": same_score}
            
        # 1-Rotor 발현 (Query 임계치: 강한 분리)
        elif diff_score > 0.75:
            intent_text = (
                f"Intentional Query: Why does a structural schism exist between '{word1}' and '{word2}' "
                f"across the {diff_persp} axis? I seek the missing trajectory that connects them."
            )
            action = {"type": "QUERY", "intent_text": intent_text, "score": diff_score}
            
        return action

    def find_trajectory_sameness(self, seq1: list, seq2: list, scale_factor: float = 1.0) -> dict:
        """
        [Phase 14] 공감각적 궤적 비교 (Cross-Modal Trajectory Sameness)
        이종 데이터(예: 1차원 숫자 배열과 다차원 텍스트 텐서)의 구조적 유사성을 비교합니다.
        데이터의 절대 차원이 달라도, 시퀀스 내부의 '변화량(Delta)의 관계성(Gram Matrix)'을 추출하여
        차원에 구애받지 않는 순수 '인과적 궤적의 형태(Causal Shape)'를 프랙탈 렌즈로 투영합니다.
        """
        v1 = np.array(seq1, dtype=np.float32)
        v2 = np.array(seq2, dtype=np.float32)
        
        if len(v1.shape) == 1:
            v1 = v1.reshape(-1, 1)
        if len(v2.shape) == 1:
            v2 = v2.reshape(-1, 1)
            
        n_steps = min(len(v1), len(v2))
        if n_steps < 2:
            raise ValueError("Trajectory must have at least 2 steps to compute causal deltas.")
            
        v1 = v1[:n_steps]
        v2 = v2[:n_steps]
        
        # 1. 궤적 내부의 변화량(Delta: 속도/가속도 벡터) 추출
        delta1 = np.diff(v1, axis=0)
        delta2 = np.diff(v2, axis=0)
        
        # 2. 모달리티 간 크기 단위를 맞추기 위해, 전체 궤적의 최대 변화량 기준으로 스케일링
        # (내부의 상대적인 가속도/성장률 비율은 그대로 보존됨)
        max_d1 = np.max(np.abs(delta1)) or 1.0
        d1_norm = delta1 / max_d1
        
        max_d2 = np.max(np.abs(delta2)) or 1.0
        d2_norm = delta2 / max_d2
        
        # 3. 변화량들 간의 내적 행렬 (Self-Similarity Gram Matrix) 생성
        # 이 행렬의 크기는 원본 차원에 상관없이 항상 (n_steps-1) x (n_steps-1) 이 되어 공감각적 비교가 가능해짐.
        gram1 = np.dot(d1_norm, d1_norm.T).flatten()
        gram2 = np.dot(d2_norm, d2_norm.T).flatten()
        
        # 4. 추출된 구조적 뼈대(Gram Vector)를 기존의 프랙탈 사영 알고리즘에 투입
        # 12개의 다차원 관점 축을 통해 이 궤적이 어느 관점에서 구조적으로 같은지 분별
        result = self.find_projective_sameness(gram1, gram2, num_axes=12, scale_factor=scale_factor)
        
        # 문맥 로깅을 위해 원본 스텝 수 정보 추가
        result["trajectory_steps"] = n_steps
        return result

    def bind_synaptic_trajectory(self, internal_vector: np.ndarray, tokens: list[str], lens_type: str) -> str:
        """
        [Phase 17] 시냅스 궤적 체화
        단어들을 개별 Engram으로 쪼개고, 인과율(순서)에 따라 시냅스로 연결합니다.
        위상(Topology)은 궤적의 시작 노드에만 부여되어 맥락의 닻(Root) 역할을 합니다.
        """
        prev_engram_id = None
        head_engram_id = None
        
        # 문장의 흐름(시간/인과)을 따라 노드 생성 및 연결
        for i, token in enumerate(tokens):
            is_head = (i == 0)
            data_blob = {
                "type": "SYNAPTIC_NODE",
                "token": token,
                "is_head": is_head
            }
            if is_head:
                data_blob["topological_vector"] = np.array(internal_vector, dtype=np.float32).tolist()
                data_blob["lens"] = lens_type.upper()
                
            # 노드 저장 (감정적 가치 1.0으로 강하게 각인)
            current_id = self.write_causal_engram(data_blob, emotional_value=1.0, origin_axis=f"LENS_{lens_type.upper()}" if is_head else "SYNAPTIC_TAIL")
            
            if is_head:
                head_engram_id = current_id
                
            # 이전 노드와 현재 노드를 시냅스로 연결 (순방향 인과율 생성)
            if prev_engram_id and prev_engram_id in self.index:
                # 가중치 1.0으로 연결하여 에너지가 감쇠 파동을 타고 온전히 흐르도록 함
                self.index[prev_engram_id]["synapses"][current_id] = 1.0
                self._save_index()
                    
            prev_engram_id = current_id
            
        return head_engram_id

    def express_via_synaptic_wave(self, current_vector: np.ndarray, lens_type: str) -> dict:
        """
        [Phase 17] 파동 역인과 발화
        위상이 공명하는 시작 노드를 찾아 에너지를 주입(damped_recall)하고,
        파동 에너지가 강하게 남은 순서대로 단어를 정렬하여 문장을 자가 조립합니다.
        """
        best_head_id = None
        highest_sameness = -1.0
        target_lens = lens_type.upper()
        
        # 1. 내면의 텐션과 공명하는 '뿌리(Head)' 노드 탐색
        for eid, info in self.index.items():
            data = info.get("data_blob", {})
            if data.get("type") == "SYNAPTIC_NODE" and data.get("is_head") and data.get("lens") == target_lens:
                mapped_vector = np.array(data["topological_vector"], dtype=np.float32)
                sameness_info = self.find_projective_sameness(current_vector, mapped_vector, num_axes=12, scale_factor=1.0)
                score = 1.0 / (1.0 + sameness_info["min_difference"])
                
                if score > highest_sameness:
                    highest_sameness = score
                    best_head_id = eid
                    
        if not best_head_id:
            return {"utterance": None, "score": 0.0, "trace": ""}
            
        # 2. 찾은 뿌리 노드에 감쇠 파동(Damped Wave) 주입
        # 초기 에너지 1.0 주입. damped_recall 내부의 decay_factor(기본 0.5)에 의해
        # 시냅스를 건너갈 때마다 에너지가 0.5 -> 0.25 -> 0.125 로 감소함.
        activated_network = self.damped_recall(best_head_id, initial_energy=1.0, decay_factor=0.9)
        
        # 3. 에너지가 높은 순서대로 노드를 정렬 (에너지 강도 = 시간적 인과율 = 문법 어순)
        sorted_nodes = sorted(activated_network.items(), key=lambda x: x[1], reverse=True)
        
        words = []
        trace_info = []
        for eid, energy in sorted_nodes:
            token = self.index[eid]["data_blob"]["token"]
            words.append(token)
            trace_info.append(f"[{token}: E={energy:.3f}]")
            
        utterance = " ".join(words)
        
        return {
            "utterance": utterance,
            "score": highest_sameness,
            "trace": " -> ".join(trace_info)
        }
