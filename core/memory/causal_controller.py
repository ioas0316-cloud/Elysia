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
