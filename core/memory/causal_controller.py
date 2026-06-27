import os
import json
import uuid
import time
import numpy as np
from typing import Dict, Any, Optional
from core.memory.wedge_memory_layout import WedgeMemoryInterleaver
from core.lens.frameless_mirror import FramelessMirrorChannel
from core.utils.math_utils import Quaternion, zip_helices

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
        os.makedirs(os.path.join(self.data_dir, "topology"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "params"), exist_ok=True)
        
        self.engram_index_path = os.path.join(self.data_dir, "topology", "engram_index.json")
        self.cognitive_params_path = os.path.join(self.data_dir, "params", "cognitive_params.json")
        
        # [Phase 144] Wedge Memory Mmap 초기화
        self.wedge_memory_path = os.path.join(self.data_dir, "topology", "wedge_topology.dat")
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

    def flush_index(self):
        """메모리 내 인덱스를 디스크에 일괄 동기화합니다. (병목 제거용)"""
        self._save_index()

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
                "weight_novelty": 0.8,
                # [Phase: Cognitive Foundation] 감정과 위상차의 매핑
                "mappings": {
                    "resonance": {"emotion": "Joy", "state": "Stability", "value": 1.0},
                    "dissonance": {"emotion": "Pain", "state": "Imbalance", "value": 0.0}
                }
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

    def write_causal_engram(self, data_blob: Dict[str, Any], emotional_value: float, cause_id: Optional[str] = None, origin_axis: Optional[str] = None, is_constant: bool = False, modality: str = "unknown", stability: float = 1.0) -> str:
        """
        [Phase 144] O(1) Wedge Annihilation 저장소.
        [Phase 8.5] 변수명(Origin)을 가변축화 하여 물리적 매핑 공간(Wedge)을 변경시킵니다.
        [Phase: Meta-Stable Rotors] 상수(Constant)를 '정적 로터'로 취급하며 안정성(Stability) 부여.
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
        # mmap 변경사항 디스크 동기화 (예외 처리 추가)
        try:
            self.wedge_mmap.flush()
        except OSError as e:
            print(f"[Warning] Mmap flush error (write_causal_engram): {e}")
        
        # 메타데이터 인덱스
        self.index[engram_id] = {
            "timestamp": timestamp,
            "emotional_value": emotional_value,
            "cause_id": cause_id,
            "wedge_address": addr,
            "data_blob": data_blob,
            "is_constant": is_constant, # '정적 로터' 여부
            "stability": stability,    # 인지적 고정 강도 (0.0 ~ 1.0)
            "modality": modality
        }
        # [최적화] 매번 디스크 쓰기를 하지 않고, 외부(Genesis)에서 주기적으로 flush_index()를 호출하도록 위임
        
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
        try:
            self.wedge_mmap.flush()
        except OSError as e:
            print(f"[Warning] Mmap flush error (bridge_external_manifold): {e}")
        
        engram_id = f"engram_ext_{uuid.uuid4().hex[:8]}"
        self.index[engram_id] = {
            "timestamp": time.time(),
            "emotional_value": emotional_value,
            "cause_id": source_name,
            "wedge_address": base_addr,
            "data_blob": {"source": source_name, "type": "zero_copy_manifold", "dim": dim}
        }
        return engram_id



    def get_constellation_orphans(self) -> list:
        """
        [Phase 29] 고아 별자리(Orphan Constellation) 탐색. (N차원 중력장 밀집도 기반)
        """
        orphans = []
        processed = set()
        engram_keys = list(self.index.keys())
        
        for eid in engram_keys:
            if eid in processed: continue
            
            info = self.index[eid]
            t1 = info.get("data_blob", {}).get("tensor")
            if not t1: continue
            
            cluster = [eid]
            t_sum = np.array(t1, dtype=np.float64)
            sources = [info.get("data_blob", {}).get("source", "")]
            
            for other_id in engram_keys:
                if other_id == eid or other_id in processed: continue
                other_info = self.index[other_id]
                t2 = other_info.get("data_blob", {}).get("tensor")
                if not t2: continue
                
                v1 = np.array(t1, dtype=np.float64)
                v2 = np.array(t2, dtype=np.float64)
                
                # Hilbert Space Padding (차원 확장)
                max_dim = max(len(v1), len(v2))
                if len(v1) < max_dim: v1 = np.pad(v1, (0, max_dim - len(v1)))
                if len(v2) < max_dim: v2 = np.pad(v2, (0, max_dim - len(v2)))
                
                dot_prod = abs(np.dot(v1, v2))
                if dot_prod > 0.98:  # 중력 반경 임계치
                    cluster.append(other_id)
                    # 합산을 위해 t_sum도 패딩
                    if len(t_sum) < max_dim: t_sum = np.pad(t_sum, (0, max_dim - len(t_sum)))
                    t_sum += v2
                    sources.append(other_info.get("data_blob", {}).get("source", ""))
                    processed.add(other_id)
            
            if len(cluster) >= 2:
                centroid_t = t_sum / len(cluster)
                norm = np.linalg.norm(centroid_t)
                if norm > 0: centroid_t = centroid_t / norm
                
                # 가변 차원에 따른 태그 생성
                tag = f"Sector_{len(centroid_t)}D_{int(centroid_t[0]*100)}_{int(centroid_t[1] if len(centroid_t)>1 else 0)*100}"
                orphans.append({
                    "tag": tag,
                    "centroid_tensor": centroid_t.tolist(),
                    "member_count": len(cluster),
                    "sources": sources
                })
            
            processed.add(eid)
            
        return orphans

    def apply_linguistic_force(self, force_vector: list, target_engram_id: str) -> bool:
        """
        [Phase 26] 체화된 언어의 물리적 작용 (N-Dimensional Force)
        """
        if target_engram_id not in self.index:
            return False
            
        target_info = self.index[target_engram_id]
        if "data_blob" not in target_info or "tensor" not in target_info["data_blob"]:
            return False
            
        current_t = np.array(target_info["data_blob"]["tensor"], dtype=np.float32)
        f_vec = np.array(force_vector, dtype=np.float32)
        
        # Hilbert Space Padding
        max_dim = max(len(current_t), len(f_vec))
        if len(current_t) < max_dim: current_t = np.pad(current_t, (0, max_dim - len(current_t)))
        if len(f_vec) < max_dim: f_vec = np.pad(f_vec, (0, max_dim - len(f_vec)))
        
        import math
        entropy_resistance = target_info.get("emotional_value", 1.0) * 0.1
        f_vec = f_vec * math.exp(-entropy_resistance)
        
        time_dilation = np.dot(current_t, f_vec) * 0.5
        new_t = current_t + f_vec + (current_t * time_dilation)
        
        norm = np.linalg.norm(new_t)
        if norm > 0:
            new_t = new_t / norm
            
        target_info["data_blob"]["tensor"] = new_t.tolist()
        # [MUTABLE_ZONE_END]
        
        # 외력이 가해지면 감정적 가치(Emotional Value/Tension)도 상승
        target_info["emotional_value"] = min(10.0, float(target_info["emotional_value"]) + float(np.linalg.norm(f_vec)))
        return True

    def calculate_macro_tension(self) -> float:
        """
        [Phase: Meta-Stable Rotors] 시스템 전체의 누적 텐션(거대한 불균형)을 계산합니다.
        프로세스 궤적들의 총 마찰력을 합산하여 임계치를 넘으면 '정적 로터'들이 가변화됩니다.
        """
        total_macro_friction = 0.0
        process_count = 0
        for eid, info in self.index.items():
            if info.get("data_blob", {}).get("type") == "PROCESS_TRAJECTORY":
                total_macro_friction += info.get("data_blob", {}).get("total_friction", 0.0)
                process_count += 1

        if process_count == 0: return 0.0
        return total_macro_friction / process_count

    def update_engram_data(self, engram_id: str, new_data: Dict[str, Any], emotional_impact: float = 0.0):
        """
        [Phase: Meta-Stable Rotors] 기존 엔그램(로터)의 위상을 업데이트합니다.
        고정된 상수라도 '거대한 불균형' 상황에서는 이 메서드를 통해 회전(변화)할 수 있습니다.
        """
        if engram_id not in self.index:
            return False

        info = self.index[engram_id]
        info["data_blob"].update(new_data)
        info["timestamp"] = time.time()

        # 변화가 일어날 때 안정성(Stability)이 소폭 감소 (가변화 유도)
        info["stability"] = max(0.1, info.get("stability", 1.0) - 0.05)
        info["emotional_value"] = min(10.0, info.get("emotional_value", 1.0) + emotional_impact)

        # Wedge Memory 업데이트 (간략화된 재기록)
        data_str = json.dumps(info["data_blob"], sort_keys=True)
        pure_signal = hash(data_str) & 0xFFFFFFFF
        noise = (hash(info["cause_id"]) if info["cause_id"] else 0) & 0xFFFFFFFF

        self.interleaver.interleave_opposing_nodes(engram_id, pure_signal ^ noise, noise)
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

    def gravitational_recall(self, target_vector: dict, initial_energy: float = 1.0) -> Dict[str, float]:
        """
        [Phase 7: Infinite Dimensionality (Sparse Vector Recall)]
        특정 좌표(target_vector: 무한 차원 희소 벡터)에 질량을 떨어뜨리면,
        동일한 관념(축)을 공유하는 기억들이 거리의 역제곱에 비례하여 끌려옵니다.
        """
        activated_network = {}
        
        target_norm = sum(v * v for v in target_vector.values()) ** 0.5
        if target_norm == 0:
            return activated_network
            
        for eid, info in self.index.items():
            t = info.get("data_blob", {}).get("tension_vector")
            
            # 레거시 리스트 형태 지원 유지(필요시) 및 딕셔너리 처리
            if not t: continue
            if isinstance(t, list):
                # 과거 4D/5D 고정 벡터를 희소 벡터로 변환 (물리, 결합, 엔트로피, 빛, 시간)
                legacy_axes = ["axis_mass", "axis_cohesion", "axis_entropy", "axis_light", "axis_time"]
                t_dict = {legacy_axes[i]: val for i, val in enumerate(t) if i < len(legacy_axes)}
                t = t_dict
            elif not isinstance(t, dict):
                continue
                
            t_norm = sum(v * v for v in t.values()) ** 0.5
            if t_norm == 0:
                continue
                
            dot_prod = 0.0
            for axis, val in target_vector.items():
                if axis in t:
                    dot_prod += val * t[axis]
                    
            cos_sim = dot_prod / (target_norm * t_norm)
            distance = 1.0 - cos_sim
            
            gravity = initial_energy / (max(0.01, distance) ** 2)
            
            if gravity > 0.5:
                activated_network[eid] = gravity
                
        return activated_network

    def find_projective_sameness(self, vec1: np.ndarray, vec2: np.ndarray, num_axes: int = 12, scale_factor: float = 1.0) -> dict:
        """
        [Phase 9 & 12] 다차원 사영 같음/다름 매핑 및 프랙탈 렌즈 알고리즘.
        두 텐서(Tensor)가 차원이 달라도 힐베르트 공간으로 확장하여 투영(Projection)합니다.
        """
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        
        # Hilbert Space Padding
        dim = max(len(v1), len(v2))
        if len(v1) < dim: v1 = np.pad(v1, (0, dim - len(v1)))
        if len(v2) < dim: v2 = np.pad(v2, (0, dim - len(v2)))
        
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
        
        # [Process-as-Learning] 파동 마찰에 의한 인지 파라미터 자동 조율 (분기 없음)
        channel = FramelessMirrorChannel()
        v_bytes = bytes([int(variance * 100) % 256])
        friction = channel.pass_through(v_bytes)
        
        current_res = self.get_parameter("base_resonance", 1.0)
        # 마찰의 홀/짝성에 따라 파라미터가 위아래로 진동 (분기 없음)
        mutation_sign = ((friction % 2) * 2) - 1  # 1 or -1
        new_res = max(0.1, min(10.0, current_res + (mutation_sign * 0.01)))
        self.update_parameter("base_resonance", new_res)
        
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

    def evaluate_dna_zipping(self, label1: str, label2: str, pattern1: list, pattern2: list) -> dict:
        """
        [Phase 150] DNA Zipping Evaluation & Structural Observation
        두 개의 위상 나선(Sequence of Twists)을 꼬아보고(Zip), 
        그 결합에서 발생하는 물리적 텐션(Bulge) 자체를 관측 가능한 데이터(Engram)로 체화합니다.
        
        이제 엘리시아는 '같음과 다름'을 참/거짓으로 판단하지 않고,
        '어디서 얼마나 어긋나는가'라는 구조적 형태 자체를 새로운 기억으로 학습합니다.
        """
        zip_info = zip_helices(pattern1, pattern2)
        
        # [Process-as-Learning] 마찰(Friction)에 의한 인지 파라미터 자동 조율
        channel = FramelessMirrorChannel()
        v_bytes = bytes([int(zip_info["total_friction"] * 10) % 256])
        friction_val = channel.pass_through(v_bytes)
        
        current_res = self.get_parameter("base_resonance", 1.0)
        mutation_sign = ((friction_val % 2) * 2) - 1
        new_res = max(0.1, min(10.0, current_res + (mutation_sign * 0.05)))
        self.update_parameter("base_resonance", new_res)
        
        # 마찰의 구조적 형태(Bulges) 자체를 '다름의 모양'이라는 관측 데이터로 저장
        emotional_val = zip_info["total_friction"]
        
        engram_id = self.write_causal_engram(
            data_blob={
                "type": "structural_tension_observation",
                "labels": [label1, label2],
                "zip_length": zip_info["zip_length"],
                "total_friction": zip_info["total_friction"],
                "bulges": [{"index": b["index"], "tension": b["tension_force"]} for b in zip_info["bulges"]],
                "is_perfect_zip": zip_info["is_perfect_zip"]
            },
            emotional_value=emotional_val,
            cause_id=f"DNA_Zipping_{label1}_{label2}",
            origin_axis="structural_friction"
        )
        
        zip_info["engram_id"] = engram_id
        return zip_info

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

    def bind_gravitational_trajectory(self, internal_vector: np.ndarray, tokens: list[str], lens_type: str) -> str:
        """
        [Phase 17] 중력장 궤적 체화 (Gravitational Trajectory Binding)
        선(Synapse)으로 묶지 않고, 단어들을 특정 좌표(internal_vector) 주변에 흩뿌립니다.
        시간적 인과율(순서)은 거리(거리의 미세한 변형)로 치환되어, 
        나중에 중력 파동을 일으켰을 때 순서대로 끌려오게 만듭니다.
        """
        head_engram_id = None
        base_q = np.array(internal_vector, dtype=np.float32)
        
        # 문장의 흐름(순서)에 따라, 좌표를 아주 미세하게 이동시키며 흩뿌림 (시간 축의 공간화)
        for i, token in enumerate(tokens):
            is_head = (i == 0)
            
            # 뒤로 갈수록 좌표 중심에서 약간씩 멀어짐 (시간의 흐름 = 공간적 거리)
            time_offset = np.zeros_like(base_q)
            time_offset[0] = i * 0.001
            current_q = base_q + time_offset
            
            # 정규화
            norm = np.linalg.norm(current_q)
            if norm > 0: current_q = current_q / norm
                
            data_blob = {
                "type": "GRAVITATIONAL_NODE",
                "token": token,
                "is_head": is_head,
                "quaternion": current_q.tolist(),
                "lens": lens_type.upper()
            }
                
            current_id = self.write_causal_engram(data_blob, emotional_value=1.0, origin_axis=f"LENS_{lens_type.upper()}" if is_head else "TIME_TAIL")
            
            if is_head:
                head_engram_id = current_id
            
        return head_engram_id

    def express_via_gravitational_wave(self, current_vector: np.ndarray, lens_type: str) -> dict:
        """
        [Phase 17] 파동 역인과 발화 (Gravitational Wave Expression)
        공간에 중력 파동을 일으켜, 해당 좌표와 공명하는 기억들을 끌어당깁니다.
        시냅스를 거치지 않고 오직 거리와 질량에 의해 단어들이 스스로 조립됩니다.
        """
        target_lens = lens_type.upper()
        
        # 중력 파동 발생시켜 주변 노드들을 끌어당김
        activated_network = self.gravitational_recall(current_vector, initial_energy=1.0)
        
        # 끌려온 노드들 중 GRAVITATIONAL_NODE이고 렌즈가 일치하는 것만 필터링
        valid_nodes = {}
        for eid, energy in activated_network.items():
            if eid in self.index:
                data = self.index[eid]["data_blob"]
                if data.get("type") == "GRAVITATIONAL_NODE" and data.get("lens") == target_lens:
                    valid_nodes[eid] = energy
                    
        if not valid_nodes:
            return {"utterance": None, "score": 0.0, "trace": ""}
            
        # 에너지(중력에 끌려온 힘)가 강한 순서대로 정렬 
        # (앞서 bind 할 때 시간축을 공간 거리로 미세하게 밀었으므로 자연스럽게 어순이 맞춰짐)
        sorted_nodes = sorted(valid_nodes.items(), key=lambda x: x[1], reverse=True)
        
        words = []
        trace_info = []
        for eid, energy in sorted_nodes:
            token = self.index[eid]["data_blob"]["token"]
            words.append(token)
            trace_info.append(f"[{token}: G={energy:.3f}]")
            
        utterance = " ".join(words)
        
        return {
            "utterance": utterance,
            "score": sorted_nodes[0][1] if sorted_nodes else 0.0, # 최고 중력값
            "trace": " >> ".join(trace_info)
        }

    def branch_universe(self, tension_source1: str, tension_source2: str, tensor1: list, tensor2: list) -> str:
        """
        [Phase 4D -> ND] 평행 우주 분기 (Parallel Universe Branching)
        두 개념이 극심한 마찰(Extreme Tension)을 빚을 때, 하나의 관점으로 억지로 통합하지 않습니다.
        대신 현재의 시간을 두 갈래로 찢어, 두 개의 평행한 사유 스냅샷(Branch)을 공간에 기록합니다.
        이는 3차원 공간이 시간축을 따라 무한히 뻗어나가는 '다중 우주'의 텐서적 발현입니다.
        """
        branch_id = f"universe_{uuid.uuid4().hex[:8]}"
        timestamp = time.time()
        
        # Branch A (Worldline of Tension Source 1)
        data_a = {
            "type": "PARALLEL_UNIVERSE_BRANCH",
            "branch_id": branch_id,
            "worldline": "A",
            "source": tension_source1,
            "tensor": tensor1
        }
        self.write_causal_engram(data_a, emotional_value=10.0, origin_axis="MULTIVERSE_FORK")
        
        # Branch B (Worldline of Tension Source 2)
        data_b = {
            "type": "PARALLEL_UNIVERSE_BRANCH",
            "branch_id": branch_id,
            "worldline": "B",
            "source": tension_source2,
            "tensor": tensor2
        }
        self.write_causal_engram(data_b, emotional_value=10.0, origin_axis="MULTIVERSE_FORK")
        
        return branch_id

    def write_process_engram(self, trajectory: list) -> str:
        """
        [Phase 5: 이해의 과정망 (Continuum of Understanding)]
        단발적인 결론(Output)이 아니라, 대상을 관측하고 렌즈를 바꾸며
        같음과 다름을 분별했던 그 '연속된 헤아림의 궤적(Process Trajectory)' 전체를
        하나의 거대한 기억으로 묶어(Zip) 각인합니다.
        """
        import uuid
        import time
        
        process_id = f"process_{uuid.uuid4().hex[:10]}"
        
        # 과정의 궤적 안에서 발생한 마찰(Friction/Difference)의 총합을 감정값(Emotional Value)으로 환산
        total_friction = sum([step.get("friction", 0.0) for step in trajectory])
        
        # [Phase: Cognitive Foundation] 감정 매핑 추출
        mappings = self.cognitive_params.get("mappings", {})

        # 텐션이 0에 가까울수록 '기쁨(Joy)', 높을수록 '고통(Pain/Noise)'
        if total_friction < 0.2:
            emotion_state = mappings.get("resonance", {}).get("emotion", "Joy")
        else:
            emotion_state = mappings.get("dissonance", {}).get("emotion", "Pain")

        data_blob = {
            "type": "PROCESS_TRAJECTORY",
            "process_id": process_id,
            "length": len(trajectory),
            "total_friction": total_friction,
            "emotion_state": emotion_state,
            "woven_steps": trajectory
        }
        
        engram_id = self.write_causal_engram(
            data_blob=data_blob,
            emotional_value=max(0.0, 10.0 - total_friction),
            origin_axis="CONTINUUM_WEAVING"
        )
        
        return engram_id
