"""
Elysia Causality Wave Engine (시공간 인과 파동 엔진)
=====================================================
정적인 상태의 비교를 넘어, 상태 A(원인)가 상태 B(결과)로 변환되는 
'과정(Process)' 자체를 기하학적 파동(Quaternion)으로 추출해냅니다.
Q_cause * Q_process = Q_result 의 원리를 이용하여,
코드의 연산이나 물리적 변화를 4차원 시공간 축 위의 각운동량(Tension)으로 사유합니다.
"""

from core.utils.math_utils import Quaternion
from core.brain.fractal_rotor import FractalRotor

class CausalityWave:
    def __init__(self):
        pass

    def extract_process_wave(self, cause_state: Quaternion, result_state: Quaternion) -> Quaternion:
        """
        원인 상태와 결과 상태를 이어주는 연산/과정(Process) 파동을 추출합니다.
        Q_process = Q_cause_inverse * Q_result
        """
        # 켤레 복소수(Conjugate)를 구하여 역행렬(Inverse)을 계산 (단위 쿼터니언이므로 켤레가 역원임)
        cause_inv = cause_state.conjugate()
        
        # 과정 파동 = 원인의 역원 * 결과
        process_wave = cause_inv * result_state
        return process_wave.normalize()

    def entangle_causality(self, cause_rotor: FractalRotor, result_rotor: FractalRotor) -> Quaternion:
        """
        두 개의 독립적인 로터를 시공간 축(Temporal Axis)으로 얽어냅니다.
        결과 로터는 원인 로터의 미래가 되고, 과정 파동이 그 사이의 텐션으로 기록됩니다.
        """
        process_wave = self.extract_process_wave(cause_rotor.state, result_rotor.state)
        
        # 시공간 얽힘 (Temporal Entanglement)
        cause_rotor.future_result = result_rotor
        result_rotor.past_cause = cause_rotor
        
        # 과정 파동 자체를 텐션(tau)으로 내재화하거나, 별도의 링크 속성으로 저장
        # 여기서는 원인 로터가 미래로 나아가기 위한 장력(process_tension)으로 기록합니다.
        cause_rotor.process_wave = process_wave
        
        return process_wave

    def simulate_temporal_ripple(self, cause_rotor: FractalRotor, perturbation_amount: float):
        """
        과거(원인)에 미세한 변화(Perturbation)가 생겼을 때,
        그 파동이 시간축을 타고 미래(결과)의 궤적을 어떻게 연쇄적으로 비틀어버리는지 시뮬레이션합니다.
        """
        import math
        
        # 과거를 비틂 (텐션 증가 및 위상 회전)
        cause_rotor.apply_perturbation(perturbation_amount)
        # 위상 자체를 강제로 회전 (비틀림)
        q_twist = Quaternion(math.cos(perturbation_amount), math.sin(perturbation_amount), 0.0, 0.0)
        cause_rotor.lens_offset = (cause_rotor.lens_offset * q_twist).normalize()
        
        # 나비효과: 비틀린 과거가 과정 파동(Process Wave)을 거쳐 미래를 새롭게 형성함
        current = cause_rotor
        while hasattr(current, 'future_result') and current.future_result is not None:
            future = current.future_result
            process = current.process_wave
            
            # 새로운 미래 = 비틀린 과거 * 동일한 과정(물리법칙/문맥)
            new_future_state = current.state * process
            
            # 미래 로터의 상태를 업데이트 (렌즈의 고유 위상 수정)
            future.lens_offset = new_future_state.normalize()
            # 미래 로터도 텐션(tau) 타격을 받음 (시공간 얽힘의 파급력)
            future.apply_perturbation(perturbation_amount * 0.618)
            
            # 다음 미래로 연쇄
            current = future
