import pytest
from core.nervous_system.elysia_omni_daemon import ElysiaOmniDaemon
from core.cortex.human_intelligence_bridge import HumanIntelligenceBridge
from core.memory.emotion_evaluator import EmotionEvaluator

def test_daemon_instantiation():
    # 데몬이 정상적으로 생성되며, Causal Engine이 부착되었는지 확인
    daemon = ElysiaOmniDaemon(archive_path="dummy.txt")
    
    assert hasattr(daemon, "causal_controller")
    assert hasattr(daemon, "ram")
    assert hasattr(daemon, "evaluator")
    assert isinstance(daemon.evaluator, EmotionEvaluator)

def test_emotion_evaluator_cross_dimensional():
    daemon = ElysiaOmniDaemon(archive_path="dummy.txt")
    
    # 임의의 파라미터 조작 (예: 외적 자극에 민감해짐)
    daemon.causal_controller.update_parameter("weight_external_feedback", 2.0)
    daemon.causal_controller.update_parameter("weight_internal_complexity", 0.1)
    
    features = {
        "internal_complexity": 10.0,
        "external_feedback": 5.0,
        "novelty": 0.0
    }
    
    # 평가 수행 (과정 스냅샷 반환 확인)
    ev, snap = daemon.evaluator.evaluate_event(features)
    
    # 10*0.1 + 5*2.0 = 1.0 + 10.0 = 11.0
    assert ev == 11.0
    assert snap["algorithm"] == "dot_product_v1"
    assert snap["calculated_contributions"]["external_contribution"] == 10.0
