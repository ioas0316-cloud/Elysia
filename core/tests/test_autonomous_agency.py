import os
import json
import pytest
from core.memory.causal_controller import CausalMemoryController
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator
from core.cortex.agentic_observer import AgenticObserver
from core.cortex.autonomous_explorer import AutonomousExplorer
from core.brain.holographic_memory import HologramMemory

@pytest.fixture
def temp_data_dir(tmpdir):
    return str(tmpdir.mkdir("data"))

@pytest.fixture
def dummy_transcript(tmpdir):
    filepath = str(tmpdir.join("dummy_transcript.jsonl"))
    with open(filepath, 'w', encoding='utf-8') as f:
        # Dummy USER_INPUT
        f.write(json.dumps({"type": "USER_INPUT", "content": "나는 우주에 대해 알고 싶어"}) + "\n")
        # Dummy AGENT_RESPONSE
        f.write(json.dumps({
            "type": "AGENT_RESPONSE", 
            "content": "우주를 탐색하기 위한 계획을 세웁니다.",
            "tool_calls": [{"name": "search_web", "args": {"query": "Universe"}}]
        }) + "\n")
    return filepath

def test_agentic_observer(temp_data_dir, dummy_transcript):
    controller = CausalMemoryController(data_dir=temp_data_dir)
    # 무조건 각인되게 임계치 조정
    controller.update_parameter("eureka_threshold", 0.0)
    ram = WorkingMemoryRAM(causal_controller=controller)
    evaluator = EmotionEvaluator(causal_controller=controller)
    
    observer = AgenticObserver(ram, evaluator, transcript_path=dummy_transcript)
    observer.observe_creator_logs()
    
    # SSD(CausalMemoryController)에 "agentic_causality" 태그가 각인되었는지 확인
    found_mimicry = False
    for engram_id, meta in controller.index.items():
        trace = controller.read_engram_trace(engram_id)
        if trace and "creator_mimicry" in trace.get("data", {}).get("tags", []):
            found_mimicry = True
            data = trace["data"]["agentic_causality"]
            assert "우주에 대해 알고 싶어" in data["goal_or_stimulus"]
            assert "search_web" in data["actions_taken"]
            break
            
    assert found_mimicry

def test_autonomous_explorer(temp_data_dir):
    controller = CausalMemoryController(data_dir=temp_data_dir)
    controller.update_parameter("eureka_threshold", 0.0)
    ram = WorkingMemoryRAM(causal_controller=controller)
    evaluator = EmotionEvaluator(causal_controller=controller)
    memory = HologramMemory()
    
    # 강제로 특정 노드의 텐션을 높여 탐색을 유도합니다. (예: "블랙홀")
    memory.register_concept("블랙홀")
    memory.ui_concept_map["블랙홀"].tau = 100.0 # 극심한 결핍
    
    explorer = AutonomousExplorer(memory, ram, evaluator)
    
    # 위키피디아 탐색 유도
    success = explorer.trigger_exploration()
    assert success == True
    
    # 탐색 후 텐션이 해소되었는지 확인
    assert memory.ui_concept_map["블랙홀"].tau < 15.0
    
    # SSD에 탐색 결과가 Engram으로 남았는지 확인
    found_exploration = False
    for engram_id, meta in controller.index.items():
        trace = controller.read_engram_trace(engram_id)
        if trace and "autonomous_exploration" in trace.get("data", {}).get("tags", []):
            found_exploration = True
            data = trace["data"]["autonomous_learning"]
            assert "블랙홀" in data["target_concept"]
            assert len(data["acquired_knowledge"]) > 0
            break
            
    assert found_exploration
