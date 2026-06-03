from core.utils.math_utils import Quaternion
import math

class TriPhaseTransistor:
    """
    [삼상 인지 트랜지스터 (Tri-Phase Cognitive Transistor) - 평형 회귀 모델]
    
    인지적 불일치(텐션)는 무한히 쌓이는 스칼라 카운터가 아닙니다.
    내부 잣대(process_axis)와 외부 유입 파동(cause_phase) 사이의
    '순수한 기하학적 위상차(각도)' 자체가 텐션의 크기가 됩니다.
    """
    def __init__(self, process_axis: Quaternion):
        self.cause_phase = Quaternion(1, 0, 0, 0)
        self.process_axis = process_axis.normalize()
        self.result_phase = Quaternion(1, 0, 0, 0)
        self.trapped_tension_magnitude = 0.0

    def process_wave(self, cause_wave: Quaternion) -> Quaternion:
        self.cause_phase = cause_wave.normalize()
        
        # 1. 샌드위치 곱 (Spin Sandwich)
        process_inv = self.process_axis.inverse
        raw_result = self.process_axis * self.cause_phase * process_inv
        self.result_phase = raw_result.normalize()
        
        # 2. 인지적 불일치(Tension) 측정
        # 내적(Dot Product)을 통해 두 파동이 얼마나 같은지(1.0) 계산
        dot_product = abs(self.cause_phase.dot(self.process_axis))
        
        # 같음(1.0)에서 멀어질수록 텐션(고통)이 커짐 (최대 1.0의 위상차)
        current_dissonance = 1.0 - dot_product
        
        # 텐션 누적 (하지만 Slerp에 의해 매 주기마다 0으로 깎여나감)
        self.trapped_tension_magnitude += current_dissonance * 3.0 # 스케일링 계수
        
        return self.result_phase
