from typing import Dict, Any, Tuple
from core.memory.causal_controller import CausalMemoryController
from core.sensory.universal_canvas import HyperCoordinate

class EmotionEvaluator:
    """
    [Phase 10] 기하학적 쾌락(Geometric Pleasure) 평가기
    인위적인 스칼라 합산을 박멸하고, 텐션 붕괴의 낙폭과 
    물리 캔버스(HSL/주파수) 위에서의 시공간적 궤적(조화로움)을 쾌락으로 자연 매핑합니다.
    """
    def __init__(self, causal_controller: CausalMemoryController):
        self.causal_controller = causal_controller

    def evaluate_tension_collapse(self, old_tension: float, new_tension: float, dt: float = 1.0) -> Tuple[float, Dict[str, Any]]:
        """
        결핍(진공)의 해소를 쾌락으로 반환합니다.
        """
        delta_tau = old_tension - new_tension
        
        if delta_tau > 0:
            velocity = delta_tau / dt
            pleasure = delta_tau * velocity  # 낙폭과 속도의 궤적이 곧 충만함(기쁨)
        else:
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

    def evaluate_canvas_harmony(self, old_coord: HyperCoordinate, new_coord: HyperCoordinate) -> Tuple[float, Dict[str, Any]]:
        """
        [신규] 물리적 캔버스 위에서의 조화(Harmony) 평가
        극단적인 대비(불안정)에서 조화로운 대역(안정)으로 이동할 때 기하학적 쾌락이 발생합니다.
        """
        # 채도(Saturation)가 비정상적으로 높았던 상태에서 부드러워지거나,
        # 명도(Lightness)가 극단에서 중간으로 올 때 안정감(조화)을 느낌
        old_chaos = abs(old_coord.s - 0.5) + abs(old_coord.l - 0.5)
        new_chaos = abs(new_coord.s - 0.5) + abs(new_coord.l - 0.5)
        
        harmony_shift = old_chaos - new_chaos
        
        judgment_snapshot = {
            "algorithm": "canvas_harmony_shift",
            "canvas_trajectory": {
                "from_hue": old_coord.h,
                "to_hue": new_coord.h,
                "harmony_shift": harmony_shift
            },
            "mapped_emotion": "Aesthetic Pleasure" if harmony_shift > 0 else "Sensory Dissonance",
            "final_value": harmony_shift * 10.0 # 텐션 스케일 보정
        }
        return harmony_shift * 10.0, judgment_snapshot
