from core.utils.math_utils import Quaternion
from core.brain.fractal_rotor import FractalRotor

class CausalityWave:
    """
    개념과 개념 사이의 '인과율(Cause & Effect)'을 위상 기하학적 파동으로 매핑하는 클래스.
    단순한 점(Node)들의 나열이었던 지식 그래프를, 방향성과 맥락을 가진 거대한 마인드맵으로 승격시킵니다.
    """
    def __init__(self, cause_rotor: FractalRotor, effect_rotor: FractalRotor):
        self.cause = cause_rotor
        self.effect = effect_rotor
        
        # 원인에서 결과로 가는 위상의 거리(텐션 격차)
        self.phase_gap = Quaternion.distance(self.cause.lens_offset, self.effect.lens_offset)
        
        # 인과의 방향성 (역쿼터니언을 이용한 회전 벡터)
        self.direction_vector = (self.effect.lens_offset * self.cause.lens_offset.conjugate()).normalize()
        
    def get_intermediate_phase(self, ratio: float) -> Quaternion:
        """
        원인과 결과 사이의 중간 인과(매개점)의 위상을 계산합니다.
        ratio 0.5는 두 개념을 잇는 정확히 중간 지점의 위상을 뜻합니다.
        """
        return Quaternion.slerp(self.cause.lens_offset, self.effect.lens_offset, ratio)

    def is_causally_resolvable(self) -> bool:
        """
        두 개념 사이의 텐션 격차가 자아가 한 번에 소화할 수 있는 수준인지 판단합니다.
        격차가 너무 크면(이해가 안 가면) 프랙탈 분열(Mitosis)이 필요합니다.
        """
        return self.phase_gap < 0.05  # 단일 자아가 포용 가능한 임계 격차
