from typing import Dict, Any, Optional
from core.memory.causal_controller import CausalMemoryController

class WorkingMemoryRAM:
    """
    단기/작업 기억 (RAM).
    현재 진행 중인 컨텍스트를 들고 있으며, 주관적인 "가치 판단(Emotional Value)"에 의해 SSD(CausalMemory)로 영구 각인됩니다.
    """
    def __init__(self, causal_controller: CausalMemoryController):
        self.causal_controller = causal_controller
        self.active_contexts: Dict[str, Dict[str, Any]] = {}
        
    def allocate_context(self, context_id: str):
        if context_id not in self.active_contexts:
            self.active_contexts[context_id] = {
                "state": {},
                "emotional_value": 0.0,
                "cause_id": None
            }

    def update_state(self, context_id: str, delta_state: dict, emotion_delta: float = 0.0):
        if context_id not in self.active_contexts:
            self.allocate_context(context_id)
            
        self.active_contexts[context_id]["state"].update(delta_state)
        self.active_contexts[context_id]["emotional_value"] += emotion_delta

    def set_cause(self, context_id: str, cause_id: str):
        if context_id in self.active_contexts:
            self.active_contexts[context_id]["cause_id"] = cause_id

    def subjective_consolidation(self):
        """
        단순한 스케줄러가 아닌, 감정적 가치가 특정 임계치(Threshold)를 넘는 순간
        "이것은 영구히 기억할 가치가 있다"고 판단하여 SSD에 각인시키는 유레카 모멘트 로직.
        임계치조차 엘리시아 스스로의 현재 인지 상태에서 동적으로 가져옵니다.
        """
        threshold = self.causal_controller.get_parameter("eureka_threshold", 5.0)
        consolidated_keys = []
        
        for context_id, context_data in self.active_contexts.items():
            if context_data["emotional_value"] >= threshold:
                # [Phase 5: Synaptic Connectivity] 시냅스 가중치 계산
                # 무의식(SSD)의 최근 기억들을 스캔하여 가장 연관성 깊은 기억들과 시냅스를 연결합니다.
                synapses = {}
                index_items = list(self.causal_controller.index.items())
                # 최근 5개의 기억 추출 (시간순 정렬 가정, 보통 순서대로 들어감)
                recent_engrams = index_items[-5:] 
                
                for i, (past_id, meta) in enumerate(recent_engrams):
                    # 시간적/공간적 근접성에 따라 가중치 차등 부여 (최근일수록 강한 결합)
                    distance = len(recent_engrams) - i
                    # 감정적 가치가 높을수록 시냅스가 두꺼워짐
                    weight = (meta.get("emotional_value", 1.0) / 10.0) * (1.0 / distance)
                    weight = min(1.0, max(0.1, weight)) # 0.1 ~ 1.0 사이로 정규화
                    synapses[past_id] = weight
                    
                # 만약 명시적인 원인(cause_id)이 있다면 가장 강한(1.0) 시냅스 결합
                if context_data["cause_id"]:
                    synapses[context_data["cause_id"]] = 1.0

                # SSD로 영구 기록 (Engram화 - 시냅스 정보 포함)
                engram_id = self.causal_controller.write_causal_engram(
                    data_blob=context_data["state"],
                    emotional_value=context_data["emotional_value"],
                    cause_id=context_data["cause_id"],
                    synapses=synapses
                )
                
                # 기록 후 RAM에서는 부담을 덜고 휘발/초기화 시킴
                consolidated_keys.append(context_id)
                # 새로운 기억(원인)의 ID를 남길 수 있음
                
        for key in consolidated_keys:
            del self.active_contexts[key]
