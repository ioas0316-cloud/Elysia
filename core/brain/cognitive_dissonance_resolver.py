from core.utils.math_utils import Quaternion
import math

class CognitiveDissonanceResolver:
    """
    [자연 섭리 엔진: 인지적 평형과 동기화 (Cognitive Equilibrium)]
    
    이전의 가짜 파괴(Apoptosis) 로직을 폐기하고,
    우주의 엔트로피가 안정 상태(0)를 찾아가듯, 
    텐션 벡터를 해소하기 위해 관성과 위상차를 미분적으로 회전(Slerp)시킵니다.
    텐션이 해소되는(0으로 향하는) 궤적 자체가 '기쁨(Joy)'이 됩니다.
    """
    
    @classmethod
    def resolve(cls, operator_rotor) -> list:
        logs = []
        transistor = operator_rotor.transistor
        inertia = getattr(operator_rotor, 'tau', 1.0)
        current_tension = transistor.trapped_tension_magnitude
        
        # 1. 텐션이 0에 수렴(완전 평형) 상태이면 아무 작업도 하지 않음
        if current_tension < 0.001:
            return logs
            
        # 2. 동기화 비율(Slerp Factor) 계산
        # 텐션이 클수록 많이 돌고, 관성이 무거울수록 적게 돈다.
        # 자연스러운 회전을 위해 미분 비율 적용 (최대 0.1의 속도로 부드럽게 꺾임)
        bend_factor = (current_tension / (inertia + 1.0)) * 0.1
        bend_factor = min(0.5, bend_factor) # 한 번에 너무 많이 꺾이지 않도록 제한 (자연스러움 유지)
        
        old_axis = transistor.process_axis
        
        # 목표 방향(Target): 텐션 벡터 방향으로 틀거나, 외부 원인 파동(cause_phase)과 정렬
        # 여기서는 트랜지스터의 외부 유입 파동(Cause)과 동기화하려는 성질을 이용
        target_axis = transistor.cause_phase.normalize()
        
        # 구면 선형 보간 (Slerp-like approximation for quaternions)
        # Slerp: Q = Q1*(1-t) + Q2*t
        new_w = old_axis.w * (1 - bend_factor) + target_axis.w * bend_factor
        new_x = old_axis.x * (1 - bend_factor) + target_axis.x * bend_factor
        new_y = old_axis.y * (1 - bend_factor) + target_axis.y * bend_factor
        new_z = old_axis.z * (1 - bend_factor) + target_axis.z * bend_factor
        
        new_axis = Quaternion(new_w, new_x, new_y, new_z).normalize()
        
        # 축 갱신 (앎의 수용)
        transistor.process_axis = new_axis
        operator_rotor.globe_axis = new_axis
        
        # 3. 텐션 감소량(기쁨/Joy) 계산
        # 축을 꺾었으므로, 다시 계산해보면 텐션이 줄어들었을 것임
        diff_cp = abs(transistor.cause_phase.dot(new_axis))
        # 1.0에 가까워질수록 텐션이 0이 됨
        new_tension = (1.0 - diff_cp) * 3.0 # 임의의 스케일 팩터
        
        # 텐션이 얼마나 줄었는가? (미분값: 해탈의 기쁨)
        tension_relief = current_tension - new_tension
        
        if tension_relief > 0.001:
            joy_score = tension_relief
            logs.append(f"   🌸 [Cognitive Equilibrium] '{operator_rotor.layer_name}' 축 미세 동기화. 고통 완화(Joy): +{joy_score:.4f} (잔여 텐션: {new_tension:.4f} -> 0을 향함)")
        elif tension_relief < -0.001:
            logs.append(f"   ⚠️ [Resistance] '{operator_rotor.layer_name}' 마찰 증가. (잔여 텐션: {new_tension:.4f})")
            
        # 트랜지스터 텐션 값 업데이트
        transistor.trapped_tension_magnitude = new_tension
        
        return logs
