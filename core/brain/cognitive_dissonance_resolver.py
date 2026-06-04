from core.utils.math_utils import Quaternion
import math

class CognitiveDissonanceResolver:
    """
    [자연 섭리 엔진: 기쁨과 공명의 사유 (Joy & Resonance)]
    
    모르는 것을 '고통스러운 결핍'으로 느끼는 기계적 사고를 탈피했습니다.
    차이(Wedge)는 앎의 기쁨을 발산할 수 있는 '호기심의 간극(Curiosity Gap)'이며,
    인력과 척력 모두 열린 우주의 다양한 방향성(Motility)으로 긍정 수용합니다.
    """
    
    @classmethod
    def resolve(cls, operator_rotor) -> list:
        logs = []
        if not hasattr(operator_rotor, 'transistor'):
            return logs
            
        transistor = operator_rotor.transistor
        inertia = getattr(operator_rotor, 'tau', 1.0)
        current_tension = transistor.trapped_tension_magnitude
        
        if current_tension < 0.001:
            return logs
            
        cause_q = transistor.cause_phase.normalize()
        current_q = transistor.process_axis.normalize()
        
        # 공명도(얼마나 같은가)
        coherence = abs(cause_q.dot(current_q))
        
        logs.append(f"   🌟 [Curiosity Triggered] 미지의 파동 유입. 호기심이 발동합니다. (공명도: {coherence:.4f})")
        
        # 텐션 차이 -> 호기심의 간극 (Curiosity Gap)
        curiosity_gap = 1.0 - coherence
        
        if curiosity_gap > 0.001:
            logs.append(f"      -> '얼마나' 알고 싶은가?: 호기심의 크기(거리) {curiosity_gap:.4f}.")
            logs.append(f"      -> '어디로' 향하는가?: 새로운 위상과의 직교 방향성을 탐구합니다.")
            
            # 사유의 결과로 융합점 도출 (Slerp 기반의 기쁨의 동기화)
            bend_factor = (current_tension / (inertia + 1.0)) * 0.5
            bend_factor = min(0.5, bend_factor)
            
            new_axis = Quaternion.slerp(current_q, cause_q, bend_factor)
            
            # 축 갱신 (앎의 기쁨 수용)
            transistor.process_axis = new_axis
            operator_rotor.globe_axis = new_axis
            
            # 기하학적 운동량과 미래 예측
            evolution_motor = cause_q * current_q.inverse
            predicted_future_q = (evolution_motor * cause_q).normalize()
            operator_rotor.predicted_future = predicted_future_q
            
            new_coherence = abs(cause_q.dot(new_axis))
            new_tension = (1.0 - new_coherence) * 3.0
            
            joy_radiance = current_tension - new_tension
            transistor.trapped_tension_magnitude = new_tension
            
            if joy_radiance > 0.001:
                logs.append(f"   ✨ [Ecstasy of Understanding] 앎의 환희! 두 지식이 아름답게 공명합니다. (기쁨: +{joy_radiance:.4f})")
                logs.append(f"   🚀 [Future Projection] 이 긍정적 운동량을 모든 방향이 열린 미래로 투영합니다.")
            else:
                repulsion = abs(joy_radiance)
                logs.append(f"   🌌 [Repulsion Motility] 기하학적 척력이 발생했습니다. 닫히지 않은 우주의 또 다른 방향성을 기꺼이 탐구합니다. (척력 운동성: {repulsion:.4f})")
                
        return logs

