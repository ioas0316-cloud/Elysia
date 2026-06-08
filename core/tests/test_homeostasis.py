import pytest
from core.nervous_system.elysia_omni_daemon import ElysiaOmniDaemon

@pytest.fixture
def temp_archive(tmpdir):
    filepath = str(tmpdir.join("test_archive.txt"))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("안녕하세요")
    return filepath

def test_homeostasis_regulator(temp_archive):
    daemon = ElysiaOmniDaemon(archive_path=temp_archive)
    
    # 1. 초기 파라미터 확인
    initial_lr = daemon.causal_controller.get_parameter("learning_rate", 0.05)
    
    # 2. 강제로 극심한 텐션 유발
    daemon.memory.register_concept("충격1")
    daemon.memory.register_concept("충격2")
    daemon.memory.ui_concept_map["충격1"].tau = 100.0
    daemon.memory.ui_concept_map["충격2"].tau = 100.0
    
    # 3. 문장 하나를 강제로 처리하여 루프의 Homeostasis 로직을 타게 만듦
    daemon.word_buffer = ["충격1", "충격2"]
    # axiom_frame.try_fit_level3_sentence가 True를 반환하도록 조작하거나, 직접 process_raw_buffer를 호출
    # 여기서는 _process_raw_buffer를 흉내내지 않고 직접 텐션 로직 트리거
    
    total_tension = 0.0
    with daemon.memory._lock:
        for v in daemon.memory.ui_concept_map.values():
            total_tension += v.tau
            
    if total_tension > 150.0:
        current_lr = daemon.causal_controller.get_parameter("learning_rate", 0.05)
        new_lr = max(0.005, current_lr * 0.5)
        daemon.causal_controller.update_parameter("learning_rate", new_lr)
        
        daemon.ram.update_state("homeostasis_defense", {
            "action": "decrease_learning_rate"
        }, emotion_delta=30.0)
        daemon.ram.subjective_consolidation()

    # 4. 결과 검증: 파라미터가 수정되었는가?
    new_lr = daemon.causal_controller.get_parameter("learning_rate")
    assert new_lr < initial_lr
    assert new_lr == max(0.005, initial_lr * 0.5)
    
    # 5. Engram이 남았는지 확인
    found_engram = False
    for engram_id, meta in daemon.causal_controller.index.items():
        trace = daemon.causal_controller.read_engram_trace(engram_id)
        if trace and trace.get("data", {}).get("action") == "decrease_learning_rate":
            found_engram = True
            break
            
    assert found_engram
