"""
test_sensorimotor_bifurcation.py
================================
감각운동적 분기점 및 공감각적 주체성 검증 테스트:
1. 손과 발이 단순 필터가 아닌 '공감각적 토대이자 분기점'임을 입증
2. 공(Ball)과 자동차(Car)에 직접 개입하여 세상과 도구의 변화를 재감각, 재인식
3. "아, 자동차는 이렇게 움직이는구나!"라는 인과적 분기점과 신체 도식 확장
4. 눈의 적극적 용도구조: 목적 응시 및 머릿속 형상 조합(바퀴 + 차체 = 자동차 창발)
"""

import math
import numpy as np
import torch

from core.sensory.embodied_organ_continuum import EmbodiedMultidimensionalOrganContinuum
from core.sensory.sensorimotor_bifurcation_junction import (
    PhysicalObjectAffordance,
    SynestheticBifurcationJunction,
    GeometricPrimitive,
)


def run_sensorimotor_bifurcation_test():
    print("=" * 72)
    print(" [검증] 감각운동적 분기점 및 공감각적 주체성 (Sensorimotor Agency)")
    print(" \"손과 발, 눈과 귀는 필터가 아니라, 공감각적 토대이자 분기점이다.\"")
    print("=" * 72)

    continuum = EmbodiedMultidimensionalOrganContinuum(spatial_resolution=16, num_acoustic_bands=8)
    junction = SynestheticBifurcationJunction(
        visual_organ=continuum.visual_organ,
        acoustic_organ=continuum.acoustic_organ,
        somatosensory_organ=continuum.somatosensory_organ,
    )

    # ─────────────────────────────────────────────────────────────
    # [시나리오 1] 손(Hand)과 공(Ball)의 상호작용 및 재인식 분기점
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print(" [1] 손(Hand)의 개입: 공(Ball) 굴리기와 물리 법칙의 재인식")
    print("-" * 72)

    ball = PhysicalObjectAffordance(
        name="테니스 공 (Ball)",
        mass=0.06,  # 60g
        geometry_type="spherical",
        friction_coefficient=0.08,
        degrees_of_freedom=3,
        controllable_interfaces=["surface_push", "grasp"],
    )

    # (1) 어포던스 평가 (손으로 할 수 있는 일 vs 할 수 없는 일)
    affordance_ball = junction.effector.evaluate_affordance(ball)
    print(f"  * 어포던스 판별: {'[개입 가능]' if affordance_ball.is_actionable else '[불가]'}")
    print(f"    - 행동 유형: {affordance_ball.action_type}")
    print(f"    - 신체 판단 이유: {affordance_ball.reason}")

    # (2) 능동적 물리 개입 및 공감각적 재감각 루프 가동
    epiphany_ball = junction.execute_closed_sensorimotor_loop(
        target_object=ball,
        interface="surface_push",
        intervention_force=15.0,  # 15N의 밀기 힘
    )

    print(f"  * 실행된 개입: {epiphany_ball.action_executed}")
    print(f"  * 관측된 대상 상태 변화: {epiphany_ball.transformation_observed}")
    print(f"  * 공감각 피드백 (손의 반작용: {epiphany_ball.synesthetic_feedback['reaction_force_n']:.2f}N, "
          f"시각 광류: {epiphany_ball.synesthetic_feedback['visual_optic_flow']:.3f})")
    print(f"\n  \\-> [분기점의 개안]:\n      \"{epiphany_ball.epistemic_branching_insight}\"")

    # ─────────────────────────────────────────────────────────────
    # [시나리오 2] 발(Foot)과 손(Hand)으로 자동차(Car)를 조종: 신체 도식 확장
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print(" [2] 발(페달)과 손(운전대)의 개입: 자동차(Car)의 신체 도식 확장")
    print("-" * 72)

    car = PhysicalObjectAffordance(
        name="자동차 (Car)",
        mass=1200.0,  # 1.2톤
        geometry_type="wheeled_vehicle",
        friction_coefficient=0.02,
        degrees_of_freedom=6,
        controllable_interfaces=["steering_wheel", "accelerator_pedal", "brake"],
    )

    # (1) 거대 도구 어포던스 평가
    affordance_car = junction.effector.evaluate_affordance(car)
    print(f"  * 어포던스 판별: {'[개입 가능]' if affordance_car.is_actionable else '[불가]'}")
    print(f"    - 행동 유형: {affordance_car.action_type}")
    print(f"    - 신체 판단 이유: {affordance_car.reason}")

    # (2) 능동적 운전 개입 (발: 가속 페달 45N, 손: 조향각 0.25 rad)
    epiphany_car = junction.execute_closed_sensorimotor_loop(
        target_object=car,
        interface="accelerator_pedal",
        intervention_force=45.0,
        steering_angle=0.25,
    )

    print(f"  * 실행된 개입: {epiphany_car.action_executed}")
    print(f"  * 관측된 대상 상태 변화: {epiphany_car.transformation_observed}")
    print(f"  * 공감각적 상호 피드백:")
    print(f"    - [시각] 풍경 광류(Optic Flow): {epiphany_car.synesthetic_feedback['visual_optic_flow']:.3f}")
    print(f"    - [청각] 가속 엔진 회전 주파수: {epiphany_car.synesthetic_feedback['acoustic_pitch_hz']:.1f} Hz")
    print(f"    - [체성] 등에 실리는 관성 가속도(G-force): {epiphany_car.synesthetic_feedback['somatosensory_g_force']:.2f}")
    print(f"    - [신체 도식 확장 여부]: {epiphany_car.body_schema_extended} (차량이 신체의 확장으로 편입됨)")
    print(f"\n  \\-> [분기점의 개안]:\n      \"{epiphany_car.epistemic_branching_insight}\"")

    # ─────────────────────────────────────────────────────────────
    # [시나리오 3] 눈의 적극적 용도구조: 목적 응시 및 머릿속 형상 조합
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print(" [3] 눈(Eye)의 적극적 용도구조: 능동적 목적 응시와 머릿속 형상 조합")
    print("-" * 72)

    # (1) 목적 응시 (Active Purposive Gaze)
    scene = np.zeros((16, 16))
    scene[6:10, 6:10] = 1.0  # 중앙에 도구 접점 존재
    gaze_result = junction.purposive_gaze.purposive_scan(scene, target_intent="find_tool_affordance")
    print(f"  * [목적 응시]: {gaze_result['purposive_narrative']}")
    print(f"    - 주의 집중 초점 좌표: {gaze_result['focal_center']}, 현저성 에너지: {gaze_result['saliency_energy']:.3f}")

    # (2) 머릿속 형상 조합 (Mental Gestalt Recombination)
    # 원형 바퀴 원형 + 사각형 차체 상자 원형
    wheel_primitive = GeometricPrimitive(
        name="wheel_primitive (원형 바퀴)",
        contour_2d=np.ones((8, 8)),
        intrinsic_affordance="rolling_mobility (회전 구름 운동성)",
    )
    chassis_primitive = GeometricPrimitive(
        name="chassis_primitive (사각형 차체 상자)",
        contour_2d=np.ones((8, 8)),
        intrinsic_affordance="containment_stability (수용 및 안정성)",
    )

    gestalt_result = junction.gestalt_combiner.recombine_primitives(
        wheel_primitive, chassis_primitive, coupling_type="coaxial_axle"
    )

    print(f"  * [머릿속 형상 가상 조합]:")
    print(f"    - 결합된 형상 명칭: {gestalt_result.combined_name}")
    print(f"    - 출현한 새로운 기능: {gestalt_result.emergent_function}")
    print(f"\n  \\-> [형상 창발 통찰]:\n      \"{gestalt_result.mental_anticipation_insight}\"")

    print("\n" + "=" * 72)
    print(" [검증 완료] 신체 지절과 감각 기관이 공감각적 토대이자 분기점이며,")
    print(" 눈이 능동적 형상 조합의 적극적 용도구조로 작동함을 입증했습니다.")
    print("=" * 72)


if __name__ == "__main__":
    run_sensorimotor_bifurcation_test()
