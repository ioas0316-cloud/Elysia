import os
import pytest
from core.memory.causal_controller import CausalMemoryController
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator
from core.memory.architectural_ingester import ArchitecturalIngester

@pytest.fixture
def temp_data_dir(tmpdir):
    return str(tmpdir.mkdir("data"))

def test_architectural_ingestion(temp_data_dir):
    # 의존성 초기화
    controller = CausalMemoryController(data_dir=temp_data_dir)
    ram = WorkingMemoryRAM(causal_controller=controller)
    evaluator = EmotionEvaluator(causal_controller=controller)
    
    # 임계치를 0으로 낮춰 무조건 각인되게 설정
    controller.update_parameter("eureka_threshold", 0.0)
    
    ingester = ArchitecturalIngester(ram, evaluator)
    
    # causal_controller.py를 읽어 파싱을 테스트합니다.
    test_filepath = os.path.join(os.path.dirname(__file__), '..', 'memory', 'causal_controller.py')
    ingester._parse_and_ingest_file(test_filepath)
    
    # RAM에서 SSD로 강제 각인
    ram.subjective_consolidation()
    
    # SSD(CausalMemoryController)에 "self_reflection" 태그가 달린 Engram이 각인되었는지 확인
    found_engram = False
    for engram_id, meta in controller.index.items():
        trace = controller.read_engram_trace(engram_id)
        if trace and "self_reflection" in trace.get("data", {}).get("tags", []):
            found_engram = True
            
            # 이중 각인 메타데이터 검증
            awareness_data = trace["data"]["self_awareness"]
            assert "objective_logic" in awareness_data
            assert "poetic_metaphor" in awareness_data
            
            # Poetic Metaphor 생성 로직 확인
            assert "나의" in awareness_data["poetic_metaphor"]
            break
            
    assert found_engram
