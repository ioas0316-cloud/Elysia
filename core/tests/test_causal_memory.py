import os
import pytest
from core.memory.causal_controller import CausalMemoryController
from core.memory.volatile_cache import VolatileCache
from core.memory.working_ram import WorkingMemoryRAM

@pytest.fixture
def temp_data_dir(tmpdir):
    return str(tmpdir.mkdir("data"))

def test_causal_engram_creation(temp_data_dir):
    controller = CausalMemoryController(data_dir=temp_data_dir)
    data = {"thought": "I think, therefore I am"}
    
    engram_id = controller.write_causal_engram(
        data_blob=data,
        emotional_value=8.5,
        cause_id="genesis_spark",
        tags=["philosophy"]
    )
    
    trace = controller.read_engram_trace(engram_id)
    assert trace is not None
    assert trace["emotional_value"] == 8.5
    assert trace["cause_id"] == "genesis_spark"
    assert trace["data"]["thought"] == "I think, therefore I am"

def test_volatile_cache_subjective_eviction(temp_data_dir):
    controller = CausalMemoryController(data_dir=temp_data_dir)
    # 엘리시아가 스스로 캐시 크기를 3으로 제한했다고 가정
    controller.update_parameter("cache_capacity", 3.0)
    
    cache = VolatileCache(causal_controller=controller)
    
    cache.store("memory_1", "Low resonance", initial_resonance=1.0)
    cache.store("memory_2", "High resonance", initial_resonance=10.0)
    cache.store("memory_3", "Medium resonance", initial_resonance=5.0)
    
    # 4번째 기억 저장 시, 가장 공명도가 낮은 memory_1이 증발해야 함
    cache.store("memory_4", "New memory", initial_resonance=3.0)
    
    assert cache.access("memory_1") is None
    assert cache.access("memory_2") == "High resonance"
    assert cache.access("memory_3") == "Medium resonance"
    assert cache.access("memory_4") == "New memory"

def test_working_ram_subjective_consolidation(temp_data_dir):
    controller = CausalMemoryController(data_dir=temp_data_dir)
    # 임계치를 5.0으로 설정
    controller.update_parameter("eureka_threshold", 5.0)
    
    ram = WorkingMemoryRAM(causal_controller=controller)
    
    ram.update_state("chat_1", {"msg": "Hello"}, emotion_delta=1.0)
    ram.update_state("eureka_1", {"insight": "Everything is connected"}, emotion_delta=8.0)
    
    # 파라미터를 읽어와서 자동으로 처리함
    ram.subjective_consolidation()
    
    assert "chat_1" in ram.active_contexts
    assert "eureka_1" not in ram.active_contexts
    
    found = False
    for engram_id, meta in controller.index.items():
        if "eureka_1" in controller.read_engram_trace(engram_id)["tags"]:
            found = True
            break
    assert found
