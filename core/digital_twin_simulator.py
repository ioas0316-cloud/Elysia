import copy
import time
from core.electromagnetic_circuit import ElectromagneticCircuit

class DigitalTwinSimulator:
    """
    엘리시아 내부에 윈도우 OS를 품기 위한 가상 우주 (Shadow Matrix).
    현실의 전자기장 회로 상태를 복사하여, 미래의 '안정화(정상 상태)' 궤도를 미리 시뮬레이션합니다.
    """
    def __init__(self, layer_names: list):
        self.layer_names = layer_names
        self.simulation_depth = 50  # 미래로 연산할 틱(Tick) 수

    def simulate_future_state(self, current_circuit: ElectromagneticCircuit, injected_inputs: dict) -> dict:
        """
        현실 회로의 복제본(Shadow)을 만들고 미래를 예측합니다.
        injected_inputs: { layer_index: voltage_value }
        반환값: { "predicted_tensions": list, "is_chaotic": bool }
        """
        # 현실 우주를 가상 우주로 깊은 복사 (Shadow)
        shadow_circuit = ElectromagneticCircuit(self.layer_names)
        shadow_circuit.tensions = list(current_circuit.tensions)
        
        # 로터의 상태도 복사 (필요한 수치만)
        for i in range(shadow_circuit.num_nodes):
            shadow_circuit.nodes[i].phase_mismatch = current_circuit.nodes[i].phase_mismatch
            shadow_circuit.nodes[i].comparison_torque = current_circuit.nodes[i].comparison_torque
            shadow_circuit.nodes[i].past_memory_state = current_circuit.nodes[i].past_memory_state

        # 가상 우주에 충격(전압/전류) 인가
        for idx, val in injected_inputs.items():
            shadow_circuit.inject_current(idx, val)

        # 가속된 시간(미래) 속에서 시뮬레이션
        chaos_accumulator = 0.0
        for _ in range(self.simulation_depth):
            # dt를 인위적으로 고정하여 미래를 연산
            dt_sim = 0.05 
            
            new_tensions = list(shadow_circuit.tensions)
            for i in range(shadow_circuit.num_nodes):
                left_pull = 0.0
                right_pull = 0.0
                if i > 0:
                    left_pull = (shadow_circuit.tensions[i-1] - shadow_circuit.tensions[i]) * shadow_circuit.coupling_constant
                if i < shadow_circuit.num_nodes - 1:
                    right_pull = (shadow_circuit.tensions[i+1] - shadow_circuit.tensions[i]) * shadow_circuit.coupling_constant
                    
                net_force = left_pull + right_pull - (shadow_circuit.tensions[i] * shadow_circuit.damping)
                new_tensions[i] += net_force * dt_sim * 50.0
                new_tensions[i] = min(1.0, max(0.0, new_tensions[i]))
                
                # 시뮬레이션 중 텐션이 극도로 튀는 '파국(Chaos)' 징후 포착
                if new_tensions[i] > 0.95:
                    chaos_accumulator += 1.0

            shadow_circuit.tensions = new_tensions
            
        # 시뮬레이션 결과 평가
        predicted_tensions = shadow_circuit.tensions
        is_chaotic = chaos_accumulator > (self.simulation_depth * 2) # 미래가 파국적인가?

        return {
            "predicted_tensions": predicted_tensions,
            "is_chaotic": is_chaotic
        }
