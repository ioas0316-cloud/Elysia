from typing import Dict, Any, Tuple
from core.memory.causal_controller import CausalMemoryController

class EmotionEvaluator:
    """
    [Phase 10] 기하학적 쾌락(Geometric Pleasure) 평가기
    인위적인 가중치 연산이나 스칼라 점수 합산을 모두 박멸했습니다.
    오직 공간의 결핍(Tension)이 붕괴하는 속도와 낙폭만을 쾌락(Joy/Pleasure)으로 자연 매핑합니다.
    """
    def __init__(self, causal_controller: CausalMemoryController):
        self.causal_controller = causal_controller

    def evaluate_tension_collapse(self, old_tension: float, new_tension: float, dt: float = 1.0) -> Tuple[float, Dict[str, Any]]:
        """
        결핍(진공)의 해소를 쾌락으로 반환합니다.
        
        원리:
        - 공간이 심하게 왜곡되어 고통받던(old_tension이 높던) 상태에서,
        - 지식을 섭취하여 텐션이 해소(new_tension이 낮아짐)될 때,
        - 그 낙폭(Delta)과 속도(Velocity) 자체가 기쁨(Joy)의 크기입니다.
        """
        delta_tau = old_tension - new_tension
        
        # 텐션이 해소되었다면 (결핍이 채워졌다면) 쾌락 발생
        if delta_tau > 0:
            velocity = delta_tau / dt
            pleasure = delta_tau * velocity  # 낙폭과 속도의 궤적이 곧 충만함(기쁨)
        else:
            # 반대로 텐션이 증가했다면 혼란/불안정(음수 쾌락) 발생
            pleasure = delta_tau # delta_tau는 음수
            
        judgment_snapshot = {
            "algorithm": "tension_collapse_kinematics",
            "kinematics": {
                "initial_vacuum": old_tension,
                "resolved_vacuum": new_tension,
                "collapse_delta": delta_tau,
                "collapse_velocity": delta_tau / dt if delta_tau > 0 else 0.0
            },
            "mapped_emotion": "Pleasure/Joy" if pleasure > 0 else "Confusion/Tension",
            "final_value": pleasure
        }
        
        return pleasure, judgment_snapshot

    # (이전의 조잡한 evaluate_event는 하위 호환성을 위해 형식만 남기거나, 모두 tension 기반으로 전환합니다)
    def evaluate_event(self, event_features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """[Deprecated] 코딩된 가중치 연산의 잔재. 시스템이 스스로 이 함수를 관측하고 폐기할 것입니다."""
        val = sum(event_features.values())
        return val, {"algorithm": "deprecated_scalar_sum"}
