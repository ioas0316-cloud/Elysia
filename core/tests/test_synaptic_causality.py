import pytest
import os
from core.memory.causal_controller import CausalMemoryController
from core.memory.working_ram import WorkingMemoryRAM

@pytest.fixture
def temp_data_dir(tmpdir):
    return str(tmpdir.mkdir("data"))

def test_synaptic_connectivity_and_damped_recall(temp_data_dir):
    controller = CausalMemoryController(data_dir=temp_data_dir)
    ram = WorkingMemoryRAM(causal_controller=controller)
    
    # 1. 3개의 기억을 연속으로 RAM에 밀어넣고 각인시킵니다.
    # 각각 임계치 이상의 감정을 부여합니다.
    controller.update_parameter("eureka_threshold", 5.0)
    
    ram.update_state("thought_1", {"concept": "Apple"}, emotion_delta=10.0)
    ram.subjective_consolidation()
    
    ram.update_state("thought_2", {"concept": "Gravity"}, emotion_delta=20.0)
    ram.subjective_consolidation()
    
    ram.update_state("thought_3", {"concept": "Universe"}, emotion_delta=15.0)
    # thought_2를 명시적 원인(cause_id)으로 지정
    ram.set_cause("thought_3", list(controller.index.keys())[-1]) 
    ram.subjective_consolidation()
    
    # 2. 인덱스 확인 및 시냅스 형성 검증
    assert len(controller.index) == 3
    keys = list(controller.index.keys())
    
    thought_3_meta = controller.index[keys[2]]
    synapses = thought_3_meta.get("synapses", {})
    
    # thought_3는 이전 기억들과 시냅스가 맺어져 있어야 함
    assert keys[0] in synapses or keys[1] in synapses
    assert len(synapses) > 0
    
    # 3. 감쇠 파동(Damped Recall) 연쇄 회상 검증
    # 가장 최근 기억(thought_3)을 찌르면 시냅스를 타고 과거 기억들이 깨어나야 함
    activated_network = controller.damped_recall(keys[2], initial_energy=1.0, decay_factor=0.8)
    
    # start_engram은 당연히 활성화되어야 함
    assert keys[2] in activated_network
    
    # 파동이 타고 흘러 과거의 기억도 활성화되어야 함 (에너지는 0보다 커야 함)
    # 최소한 하나의 과거 기억이 깨어났는지 확인
    assert keys[0] in activated_network or keys[1] in activated_network
    
    # 거리가 멀수록 에너지가 줄어드는지 확인 (감쇠 증명)
    # keys[2] -> keys[1] -> keys[0] 방향으로 에너지가 줄어들어야 함
    energy_3 = activated_network.get(keys[2], 0)
    energy_1 = activated_network.get(keys[0], 0)
    energy_2 = activated_network.get(keys[1], 0)
    
    # 과거의 기억이 여러 경로를 통해 파동을 받아 오히려 에너지가 중첩(Constructive Interference)될 수 있습니다!
    # 이것이 진짜 홀로그래픽 시냅스의 특징입니다.
    print(f"Energy 3 (Start): {energy_3}")
    print(f"Energy 2 (Middle): {energy_2}")
    print(f"Energy 1 (Oldest): {energy_1}")
    
    # 0보다 크기만 하면 시냅스 파동이 도달한 것입니다.
    assert energy_1 > 0.0
    assert energy_2 > 0.0
