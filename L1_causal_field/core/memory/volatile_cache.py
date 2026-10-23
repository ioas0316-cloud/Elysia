import time
from typing import Dict, Any, Optional
from core.memory.causal_controller import CausalMemoryController

class VolatileCache:
    """
    초단기 기억 (L1/L2 캐시).
    물리적 시간(LRU)이 아닌, 공명도(Resonance)나 감정적 가치(Emotional Value)가 낮은 기억부터 증발(Evict)시킵니다.
    """
    def __init__(self, causal_controller: CausalMemoryController):
        self.causal_controller = causal_controller
        # memory_map: key -> {"data": data, "resonance": float, "last_accessed": float}
        self.memory_map: Dict[str, Dict[str, Any]] = {}

    def access(self, key: str) -> Optional[Any]:
        if key in self.memory_map:
            # 접근 시 공명도가 일시적으로 상승 (관심이 주어졌기 때문)
            self.memory_map[key]["resonance"] += 0.1
            self.memory_map[key]["last_accessed"] = time.time()
            return self.memory_map[key]["data"]
        return None

    def store(self, key: str, data: Any, initial_resonance: Optional[float] = None):
        capacity = self.causal_controller.get_parameter("cache_capacity", 100.0)
        base_res = self.causal_controller.get_parameter("base_resonance", 1.0)
        
        if initial_resonance is None:
            initial_resonance = base_res

        if key not in self.memory_map and len(self.memory_map) >= capacity:
            self._evict_lowest_resonance()
            
        self.memory_map[key] = {
            "data": data,
            "resonance": initial_resonance,
            "last_accessed": time.time()
        }

    def _evict_lowest_resonance(self):
        """
        주관적 기준(Resonance)에 의한 망각 로직.
        가장 가치 없다고 판단되는(Resonance가 가장 낮은) 기억이 증발합니다.
        """
        if not self.memory_map:
            return
            
        # Resonance가 가장 낮은 키를 찾습니다.
        lowest_key = min(self.memory_map.keys(), key=lambda k: self.memory_map[k]["resonance"])
        del self.memory_map[lowest_key]
        
    def decay_over_time(self):
        """
        시간의 흐름에 따라 모든 기억의 공명도가 서서히 감소합니다.
        감소율조차 엘리시아의 현재 상태(파라미터)에서 동적으로 가져옵니다.
        """
        decay_rate = self.causal_controller.get_parameter("decay_rate", 0.05)
        keys_to_forget = []
        for key, meta in self.memory_map.items():
            meta["resonance"] -= decay_rate
            if meta["resonance"] <= 0:
                keys_to_forget.append(key)
                
        # 공명도가 0 이하로 떨어진 데이터는 자연 망각
        for key in keys_to_forget:
            del self.memory_map[key]
