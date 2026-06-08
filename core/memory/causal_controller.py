import os
import json
import uuid
import time
from typing import Dict, Any, Optional

class CausalMemoryController:
    """
    가상 SSD (data/ 폴더)를 담당하는 인과적 기억 컨트롤러.
    단순한 파일 저장이 아닌, 감정적 가치(Emotional Value)와 원인(Cause)을 메타데이터로 묶어 'Engram' 형태로 저장합니다.
    """
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)
        self.engram_index_path = os.path.join(self.data_dir, "engram_index.json")
        self.cognitive_params_path = os.path.join(self.data_dir, "cognitive_params.json")
        self._load_index()
        self._load_cognitive_params()

    def _load_index(self):
        if os.path.exists(self.engram_index_path):
            with open(self.engram_index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
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

    def write_causal_engram(self, data_blob: Dict[str, Any], emotional_value: float, cause_id: Optional[str] = None, tags: list = None, synapses: Dict[str, float] = None) -> str:
        """
        주관적 판단에 의해 '기억할 가치가 있다'고 여겨진 데이터를 SSD에 각인시킵니다.
        단독 파일이 아니라, 기존 기억들과의 기하학적 연결성(Synapses)을 부여하여 거대한 신경망 노드로 만듭니다.
        """
        engram_id = f"engram_{uuid.uuid4().hex[:12]}"
        timestamp = time.time()
        
        engram = {
            "engram_id": engram_id,
            "timestamp": timestamp,
            "emotional_value": emotional_value,
            "cause_id": cause_id,  # 이 기억을 파생시킨 이전 기억이나 자극의 ID
            "tags": tags or [],
            "synapses": synapses or {}, # {target_engram_id: weight} 시냅스 그물망
            "data": data_blob
        }
        
        filepath = os.path.join(self.data_dir, f"{engram_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(engram, f, indent=4, ensure_ascii=False)
            
        # 인덱스 업데이트 (빠른 역추적을 위해 시냅스 정보도 캐싱)
        self.index[engram_id] = {
            "timestamp": timestamp,
            "emotional_value": emotional_value,
            "cause_id": cause_id,
            "synapses": synapses or {},
            "filepath": filepath
        }
        self._save_index()
        
        return engram_id

    def read_engram_trace(self, engram_id: str) -> Optional[Dict[str, Any]]:
        """
        단일 기억뿐만 아니라, 필요하다면 원인 체인(Causal Chain)을 따라갈 수 있는 기반 메서드.
        """
        if engram_id not in self.index:
            return None
            
        filepath = self.index[engram_id]["filepath"]
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

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
