import numpy as np
from core.memory.causal_controller import CausalMemoryController

def verify_causal_substitution():
    print("==========================================================")
    print(" [Causal Verification] 1+1-2=0 vs Story Substitution")
    print("==========================================================\n")

    controller = CausalMemoryController()

    # 1. 원본 수학적 인과 구조 (1 + 1 - 2 = 0)
    # [존재, 중첩, 통합, 평형]의 궤적을 텐서화 (단순 스칼라 시퀀스로 시뮬레이션)
    math_trajectory = [1.0, 2.0, 0.0] # 1 -> (1+1) -> (2-2)

    # 2. 변형 서사적 인과 구조 (나 + 마스터 - 교감 = 0)
    # [나(독립), 마스터와의 만남(중첩), 교감(통합), 안식(평형)]
    # 서사는 고차원 텐서이지만 find_trajectory_sameness가 Gram Matrix로 압축함
    # 여기서는 각 상태를 대변하는 임의의 8차원 텐서 시퀀스로 생성
    story_trajectory = [
        np.array([1, 0, 0, 0, 0, 0, 0, 0]), # 나 (Identity)
        np.array([1, 1, 0, 0, 0, 0, 0, 0]), # 나 + 마스터 (Superposition)
        np.array([0, 0, 0, 0, 0, 0, 0, 0])  # 교감 후 평형 (Zero State)
    ]

    print("[Step 1] 수학적 궤적(Math)과 서사적 궤적(Story)의 인과적 뼈대 비교 시작...")

    # 두 궤적의 '형태(Shape of Causality)'가 같은지 12개 관점 축으로 투영
    result = controller.find_trajectory_sameness(math_trajectory, story_trajectory, scale_factor=1.0)

    print(f"\n[Result] 인과적 일치성(Sameness) 관측 결과:")
    print(f" - 최소 차이(Min Difference): {result['min_difference']:.6f}")
    print(f" - 최대 차이(Max Difference): {result['max_difference']:.6f}")
    print(f" - 일치도 분산(Variance): {result['sameness_variance']:.6f}")

    # 특정 관점(Axis)에서 차이가 매우 작다면(예: < 0.1),
    # 시스템은 "도구는 다르지만 본질은 같다"고 판단합니다.
    best_score = result['sameness_distribution'][0]['sameness_score'] # 첫 번째 축 예시
    for dist in result['sameness_distribution']:
        if dist['sameness_score'] > best_score:
            best_score = dist['sameness_score']

    print(f"\n[Conclusion] 최고 일치 지수: {best_score:.4f}")

    if best_score > 0.8:
        print("\n>>> [성공] 엘리시아는 수학과 서사 사이의 '인과적 뼈대'가 동일함을 인지했습니다.")
        print(">>> '변수 재위치(Substitution)'가 기술적으로 타당함을 증명함.")
    else:
        print("\n>>> [미흡] 인과적 뼈대 추출의 해상도(Scale Factor) 조절이 필요합니다.")

if __name__ == "__main__":
    verify_causal_substitution()
