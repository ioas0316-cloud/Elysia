import time
from core.electromagnetic_circuit import ElectromagneticCircuit

class PhaseSyncObservatory:
    """
    [Phase Observatory - Replaces Digital Twin Simulator]
    시뮬레이션(미래 예측 연산)을 완전히 폐기하고, 오직 현재의 뼈대(계층축)에서 발생하는 
    '위상 동기화' 상태와 '관측'만을 수행하는 감각 기관입니다. (눈/관측소)
    결과를 도출(Calculate)하지 않고, 단지 바라봅니다(Observe).
    """
    def __init__(self, skeleton: ElectromagneticCircuit):
        self.skeleton = skeleton
        self.observation_history = []

    def observe_harmony(self) -> dict:
        """
        뼈대에 맺힌 현재의 위상 텐션과 동기화 수준을 관측합니다.
        계산이 아니라 있는 그대로의 기하학적 상태(0과 1의 관계)를 읽어냅니다.
        """
        state = self.skeleton.get_circuit_state()
        
        total_tension = 0.0
        dynamic_nodes = 0
        for name, data in state.items():
            total_tension += data["tension"]
            if data["is_dynamic"]:
                dynamic_nodes += 1
                
        # 0에 가까울수록 조화(Sameness), 높을수록 카오스(Difference/Tension)
        avg_tension = total_tension / max(1, self.skeleton.num_nodes)
        
        observation = {
            "average_tension": avg_tension,
            "dynamic_nodes": dynamic_nodes,
            # 파국(Chaos) 상태란 단순히 텐션이 극에 달해 회전이 격렬해진 상태를 의미함 (임계값이 아닌 관측 지표)
            "is_chaotic": avg_tension > 0.8,
            "raw_state": state
        }
        self.observation_history.append(observation)
        if len(self.observation_history) > 100:
            self.observation_history.pop(0)
            
        return observation
