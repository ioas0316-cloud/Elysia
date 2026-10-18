"""
core/sensory/sensorimotor_bifurcation_junction.py
=================================================
Sensorimotor Bifurcation Junction & Synesthetic Agency Engine
(감각운동적 분기점 및 공감각적 주체성 엔진)

Philosophy:
"인간에게 손이나 발, 눈과 귀는 그냥 필터가 아니다. 공감각적 토대이자 분기점이다."
- 손으로 할 수 있는 일과 할 수 없는 일 (Affordance Boundary & Kinematic Constraints)
- 자신이 개입하여 세상과 도구, 공, 자동차가 어떻게 변화하는지 재감각하고 재인식하는 루프
- "아, 자동차는 이렇게 움직이는구나!"라는 새로운 인과적 분기점으로의 확장 (Tool-Body Schema Extension)
- 눈의 적극적 용도구조: 보고자 하는 것을 능동적으로 탐색하고(Active Purposive Gaze),
  머릿속에서 형상들을 조합하여 어떤 형태와 기능이 출현할지 가상 합성하는 구조(Mental Gestalt Recombination)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union

from core.sensory.embodied_organ_continuum import (
    EmbodiedVisualOrgan,
    EmbodiedAcousticOrgan,
    EmbodiedSomatosensoryOrgan,
    VisualOrganDiscernment,
    AcousticOrganDiscernment,
    SomatosensoryOrganDiscernment,
)


# =====================================================================
# 1. 물리적 대상 및 어포던스 (Physical Object & Affordance Specification)
# =====================================================================

@dataclass
class PhysicalObjectAffordance:
    """
    [물리적 대상 및 상호작용 어포던스]
    세상에 존재하는 객체(공, 자동차, 도구, 벽돌 등)의 물리적 특성과 상호작용 접점.
    """
    name: str                           # 'ball', 'car', 'stone', 'hammer'
    mass: float                         # 질량 (kg)
    geometry_type: str                  # 'spherical', 'wheeled_vehicle', 'rigid_block'
    friction_coefficient: float         # 마찰계수 μ
    degrees_of_freedom: int             # 자유도 (DOF)
    controllable_interfaces: List[str]  # ['surface_push', 'grasp'] or ['steering_wheel', 'accelerator_pedal', 'brake']
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    internal_drive_force: float = 0.0   # 엔진 등 자체 추진력


@dataclass
class AffordanceVerdict:
    """손/발로 할 수 있는 일과 할 수 없는 일에 대한 물리적 판별"""
    is_actionable: bool                 # 개입 가능 여부
    action_type: str                    # 'direct_push', 'grasp', 'tool_cockpit_control', 'impassable_barrier'
    reason: str                         # 왜 할 수 있고 왜 할 수 없는지에 대한 물리적 이유
    required_force: float               # 필요한 최소 힘
    feasible_interfaces: List[str]      # 조작 가능한 인터페이스 목록


@dataclass
class BifurcationEpiphany:
    """
    [분기점의 개안 (Bifurcation Epiphany)]
    개입 -> 대상 변화 -> 재감각 -> 재인식 루프를 거쳐 도달한 인과적 확장 통찰.
    """
    target_name: str
    action_executed: str
    transformation_observed: str
    synesthetic_feedback: Dict[str, Any]
    body_schema_extended: bool
    epistemic_branching_insight: str   # "아, 자동차는 이렇게 움직이는구나!"


# =====================================================================
# 2. 신체 개입 기관 (Embodied Effector Limb: 손과 발)
# =====================================================================

class EmbodiedEffectorLimb(nn.Module):
    """
    [체화된 개입 지절 (Embodied Effector Limb: 손과 발)]

    단순한 위치 수용기가 아니라, 세상에 능동적으로 개입(Intervention)하여
    물리계의 상태를 바꾸고 그 반응을 되짚어 인과 법칙을 도출하는 기관.
    """

    def __init__(self, max_force: float = 100.0, arm_span: float = 1.0):
        super().__init__()
        self.max_force = max_force      # 팔/다리의 최대 발휘 힘 (N)
        self.arm_span = arm_span        # 도달 가능 거리 (m)

    def evaluate_affordance(self, obj: PhysicalObjectAffordance) -> AffordanceVerdict:
        """
        [할 수 있는 일 vs 할 수 없는 일의 분별]
        객체의 질량, 형상, 접점을 자신의 신체 용량과 비교하여 행동 가능성을 판별.
        """
        # 1. 자동차와 같은 거대 도구 (도구 인터페이스를 통한 개입)
        if obj.geometry_type == "wheeled_vehicle":
            if "steering_wheel" in obj.controllable_interfaces and "accelerator_pedal" in obj.controllable_interfaces:
                return AffordanceVerdict(
                    is_actionable=True,
                    action_type="tool_cockpit_control",
                    reason=(
                        f"{obj.name}의 질량({obj.mass}kg)은 손으로 직접 들 수 없으나, "
                        f"운전대(손)와 페달(발)이라는 도구 접점을 통해 신체 도식을 확장 개입할 수 있습니다."
                    ),
                    required_force=20.0,
                    feasible_interfaces=["steering_wheel", "accelerator_pedal"],
                )

        # 2. 직접 조작 가능한 객체 (공, 돌 등)
        if obj.mass <= self.max_force:
            if obj.geometry_type == "spherical":
                return AffordanceVerdict(
                    is_actionable=True,
                    action_type="direct_push_and_roll",
                    reason=(
                        f"{obj.name}은 질량({obj.mass}kg)이 가볍고 구형 대칭이므로, "
                        f"손으로 감싸 쥐거나 작은 밀기 힘으로도 회전 및 구름 운동을 유발할 수 있습니다."
                    ),
                    required_force=obj.mass * 9.8 * obj.friction_coefficient,
                    feasible_interfaces=["surface_push", "grasp"],
                )
            else:
                return AffordanceVerdict(
                    is_actionable=True,
                    action_type="direct_manipulation",
                    reason=f"{obj.name}은 직접 손으로 들어 올리거나 이동시킬 수 있는 범위 내에 있습니다.",
                    required_force=obj.mass * 9.8,
                    feasible_interfaces=["grasp", "surface_push"],
                )

        # 3. 신체 용량을 초과하는 고정 장벽
        return AffordanceVerdict(
            is_actionable=False,
            action_type="impassable_barrier",
            reason=f"{obj.name}의 질량({obj.mass}kg)과 저항은 신체 한계({self.max_force}N)를 넘어 손발로 직접 변형시킬 수 없습니다.",
            required_force=obj.mass * 9.8,
            feasible_interfaces=[],
        )

    def intervene(
        self,
        obj: PhysicalObjectAffordance,
        interface: str,
        applied_force: float,
        angle_input: float = 0.0,
        dt: float = 0.1,
    ) -> Dict[str, Any]:
        """
        [세상에 대한 능동적 개입 (Intervention)]
        신체의 힘을 특정 접점에 가하여 객체의 상태(속도, 각속도, 위치)를 실제로 변화시킴.
        """
        applied_force = float(np.clip(applied_force, 0.0, self.max_force))
        prev_velocity = obj.velocity.copy()

        # A. 공(Ball) 굴리기 개입
        if obj.geometry_type == "spherical" and interface in ("surface_push", "grasp"):
            # 구름 마찰 저항을 극복하고 가속도 발생
            net_force = max(0.0, applied_force - obj.mass * 9.8 * obj.friction_coefficient)
            accel = net_force / obj.mass
            obj.velocity[0] += accel * dt
            # 회전 각속도 v = r * w
            radius = 0.15
            obj.angular_velocity[2] = obj.velocity[0] / radius
            obj.position += obj.velocity * dt

            reaction_force = applied_force * 0.3  # 손에 전해지는 반작용

            return {
                "action_type": "roll_ball",
                "applied_force": applied_force,
                "reaction_force": reaction_force,
                "delta_velocity": float(np.linalg.norm(obj.velocity - prev_velocity)),
                "new_velocity": obj.velocity.copy(),
                "new_angular_velocity": obj.angular_velocity.copy(),
                "description": f"손으로 {applied_force:.1f}N의 힘을 가하자, 구체가 {obj.angular_velocity[2]:.2f} rad/s로 회전하며 앞으로 굴러갔습니다.",
            }

        # B. 자동차(Car) 조종 개입 (핸들 및 페달)
        elif obj.geometry_type == "wheeled_vehicle":
            engine_multiplier = 35.0  # 페달에 의해 가동되는 엔진 마력 증폭비
            if interface == "accelerator_pedal":
                engine_thrust = applied_force * engine_multiplier
                obj.velocity[0] += (engine_thrust / obj.mass) * dt
            elif interface == "brake":
                brake_force = applied_force * 40.0
                obj.velocity[0] = max(0.0, obj.velocity[0] - (brake_force / obj.mass) * dt)

            # 운전대 회전 각도에 따른 횡방향 조향
            obj.angular_velocity[1] = angle_input * 1.5
            obj.velocity[1] = obj.velocity[0] * math.sin(angle_input)
            obj.position += obj.velocity * dt

            reaction_pedal = applied_force * 0.5  # 페달 스프링 반발력
            return {
                "action_type": "drive_car",
                "applied_force": applied_force,
                "steering_angle": angle_input,
                "reaction_force": reaction_pedal,
                "delta_velocity": float(np.linalg.norm(obj.velocity - prev_velocity)),
                "new_velocity": obj.velocity.copy(),
                "new_angular_velocity": obj.angular_velocity.copy(),
                "description": (
                    f"발로 페달을 {applied_force:.1f}N 밟고 운전대를 {angle_input:.2f} rad 꺾자, "
                    f"엔진이 추진력을 뿜으며 차속이 {obj.velocity[0]:.2f} m/s로 상승하고 차체가 조향되었습니다."
                ),
            }

        return {"action_type": "ineffective", "delta_velocity": 0.0, "reaction_force": 0.0}


# =====================================================================
# 3. 눈의 적극적 용도구조: 목적 응시 및 머릿속 형상 조합기
#    (Active Purposive Gaze & Mental Gestalt Combiner)
# =====================================================================

@dataclass
class GeometricPrimitive:
    """머릿속에서 조합할 수 있는 형상 원형"""
    name: str                   # 'sphere_wheel', 'box_chassis', 'rod_axle', 'wing_plane'
    contour_2d: np.ndarray      # 2D 윤곽선 레벨셋
    intrinsic_affordance: str   # 'rolling_mobility', 'containment_stability', 'directional_joint'


@dataclass
class EmergentGestaltCombination:
    """머릿속 형상 조합 결과 및 출현한 새로운 기능"""
    combined_name: str
    component_names: List[str]
    emergent_contour: np.ndarray
    emergent_function: str
    mental_anticipation_insight: str


class ActivePurposiveGaze:
    """
    [능동적 목적 시각 탐색기 (Active Purposive Gaze)]
    단순히 눈앞의 전체 픽셀을 수동적으로 담는 것이 아니라,
    자신의 의도(Intent)에 따라 특정 접점과 어포던스 영역을 집중 응시(Foveation)하는 구조.
    """

    def purposive_scan(
        self,
        scene_field: np.ndarray,
        target_intent: str = "find_affordance",
    ) -> Dict[str, Any]:
        """
        의도에 따라 시각 필드 내에서 가장 유의미한 윤곽선과 상호작용 접점을 선택 포착.
        """
        arr = np.asarray(scene_field, dtype=np.float64)
        if arr.ndim < 2:
            arr = np.outer(arr, arr) if len(arr) > 1 else np.ones((16, 16))

        # 기울기(Gradient) 에너지 맵을 통한 주의 집중 영역(Saliency Focal Center) 계산
        gy, gx = np.gradient(arr)
        energy_map = gx ** 2 + gy ** 2
        focal_y, focal_x = np.unravel_index(np.argmax(energy_map), energy_map.shape)

        return {
            "intent": target_intent,
            "focal_center": (int(focal_x), int(focal_y)),
            "saliency_energy": float(energy_map[focal_y, focal_x]),
            "purposive_narrative": f"'{target_intent}' 의도에 따라 시선을 좌표 ({focal_x}, {focal_y})의 상호작용 경계면에 정렬했습니다.",
        }


class MentalGestaltCombiner:
    """
    [머릿속 형상 조합기 (Mental Gestalt Combiner)]
    "눈으로 본 것들을 머릿속에서 조합하면 어떤 형상이 나타날지를 생각하는 적극적 용도구조"
    서로 다른 형상 조각들을 머릿속 시공간 샌드박스에서 결합하여
    새로운 구조와 기능의 출현을 사전에 시뮬레이션합니다.
    """

    def recombine_primitives(
        self,
        primitive_a: GeometricPrimitive,
        primitive_b: GeometricPrimitive,
        coupling_type: str = "coaxial_joint",
    ) -> EmergentGestaltCombination:
        """
        두 형상 원형을 머릿속에서 결합하여 출현하는 형태(Gestalt)와 기능을 예측.
        """
        # 머릿속 2D 레벨셋 격자 생성
        grid_size = 32
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        xx, yy = np.meshgrid(x, y)

        # 예: 바퀴(원형) + 차체(사각형) 결합
        if "wheel" in primitive_a.name or "wheel" in primitive_b.name:
            if "chassis" in primitive_a.name or "chassis" in primitive_b.name:
                # 사각형 차체 레벨셋 (중심)
                chassis_mask = np.maximum(np.abs(xx) - 1.0, np.abs(yy) - 0.4)
                # 원형 바퀴 레벨셋 (좌우 하단 2개)
                wheel_left = np.sqrt((xx + 0.9)**2 + (yy + 0.5)**2) - 0.35
                wheel_right = np.sqrt((xx - 0.9)**2 + (yy + 0.5)**2) - 0.35
                wheels = np.minimum(wheel_left, wheel_right)

                # 부드러운 불리언 결합 (Smooth Minimum / Union)
                combined_contour = np.minimum(chassis_mask, wheels)

                name = "Wheeled_Vehicle_Prototype (자동차/수레 원형)"
                func = "공간 적재 안정성(차체) + 마찰 극복 구름 주행성(바퀴)이 결합된 고속 수송체"
                insight = (
                    "머릿속에서 원형 바퀴 2개와 사각형 상자를 결합해 보았습니다. "
                    "단순한 정지된 상자가 아니라, 바퀴의 회전 자유도와 결합됨으로써 "
                    "지면 위를 매끄럽게 질주할 수 있는 '자동차/탈것'의 새로운 형상과 기능이 창발했습니다."
                )

                return EmergentGestaltCombination(
                    combined_name=name,
                    component_names=[primitive_a.name, primitive_b.name],
                    emergent_contour=combined_contour,
                    emergent_function=func,
                    mental_anticipation_insight=insight,
                )

        # 일반 결합 (블록 결합)
        combined_contour = np.sin(xx) * np.cos(yy)
        return EmergentGestaltCombination(
            combined_name=f"{primitive_a.name}_{primitive_b.name}_Assembly",
            component_names=[primitive_a.name, primitive_b.name],
            emergent_contour=combined_contour,
            emergent_function="복합 구조체",
            mental_anticipation_insight=f"{primitive_a.name}과 {primitive_b.name}의 기하학적 접합으로 새로운 안정 구조를 구상했습니다.",
        )


# =====================================================================
# 4. 공감각적 분기점 통합 엔진 (Synesthetic Bifurcation Junction)
# =====================================================================

class SynestheticBifurcationJunction(nn.Module):
    """
    [공감각적 분기점 통합 엔진 (Synesthetic Bifurcation Junction)]

    손, 발, 눈, 귀를 닫힌 감각운동 루프로 묶어,
    '개입 -> 세상의 물리적 변화 -> 공감각적 재감각 -> 신체 도식 확장 및 인과 분기점 확립'
    의 전 과정을 지휘하는 마스터 융합 엔진.
    """

    def __init__(
        self,
        visual_organ: EmbodiedVisualOrgan,
        acoustic_organ: EmbodiedAcousticOrgan,
        somatosensory_organ: EmbodiedSomatosensoryOrgan,
    ):
        super().__init__()
        self.visual_organ = visual_organ
        self.acoustic_organ = acoustic_organ
        self.somatosensory_organ = somatosensory_organ

        self.effector = EmbodiedEffectorLimb(max_force=120.0, arm_span=1.2)
        self.purposive_gaze = ActivePurposiveGaze()
        self.gestalt_combiner = MentalGestaltCombiner()

        # 도구-신체 도식 확장 메모리
        self.extended_body_schema: Dict[str, Any] = {}

    def execute_closed_sensorimotor_loop(
        self,
        target_object: PhysicalObjectAffordance,
        interface: str,
        intervention_force: float,
        steering_angle: float = 0.0,
    ) -> BifurcationEpiphany:
        """
        [닫힌 감각운동 루프 실행 및 분기점 발견]
        1. 어포던스 평가: 손/발로 할 수 있는 일인지 판별
        2. 물리적 개입: 대상을 조작하여 상태를 변화시킴
        3. 공감각적 재감각 (Synesthetic Resensing):
           - 시각: 광류(Optic Flow) 및 위상 속도 변화 포착
           - 청각: 마찰음 및 엔진 회전수(RPM) 주파수 변화 포착
           - 체성감각: 페달/핸들 반발력 및 관성 가속도(G-force) 체감
        4. 분기점 도출: "아, 자동차는/공은 이렇게 움직이는구나!"
        """
        # 1. 어포던스 판별
        verdict = self.effector.evaluate_affordance(target_object)
        if not verdict.is_actionable:
            return BifurcationEpiphany(
                target_name=target_object.name,
                action_executed="aborted_intervention",
                transformation_observed="변화 없음 (신체 한계 초과)",
                synesthetic_feedback={},
                body_schema_extended=False,
                epistemic_branching_insight=f"'{target_object.name}'은 손발의 물리적 용량으로 직접 조작할 수 없는 영역에 있습니다: {verdict.reason}",
            )

        # 2. 물리적 개입 실행
        intervention = self.effector.intervene(
            target_object,
            interface=interface,
            applied_force=intervention_force,
            angle_input=steering_angle,
        )

        speed = float(np.linalg.norm(target_object.velocity))

        # 3. 공감각적 재감각 (Synesthetic Resensing across Organs)

        # (a) 시각 재감각: 속도에 따른 시공간 위상 속도(Optic Flow) 발생
        shift_val = max(1, int(round(speed * 10.0)))
        f1 = np.outer(np.linspace(0.1, 0.9, 16), np.linspace(0.1, 0.9, 16))
        f2 = np.roll(f1, shift=shift_val, axis=1)
        vis_res = self.visual_organ.perceive_optical_field(f1, temporal_sequence=[f1, f2])

        # (b) 청각 재감각: 속도와 엔진 회전에 비례하는 주파수 피치 시프트
        t = np.linspace(0, 0.2, 1600)
        base_freq = 80.0 + speed * 25.0
        engine_sound = np.sin(2 * np.pi * base_freq * t) + 0.4 * np.sin(2 * np.pi * base_freq * 2 * t)
        ac_res = self.acoustic_organ.perceive_acoustic_field(engine_sound, sample_rate=8000)

        # (c) 체성감각 재감각: 관성 가속도 G-force 및 접점 반작용
        g_force = intervention["delta_velocity"] * 1.8
        reaction = intervention["reaction_force"]
        so_res = self.somatosensory_organ.perceive_somatosensory_field(
            stress_input=np.array([reaction * 0.1, g_force * 0.2]),
            hysteresis_permanence=0.2,
        )

        # 4. 신체 도식 확장 및 분기점 개안
        is_tool_extended = target_object.geometry_type == "wheeled_vehicle" and speed > 0.05
        if is_tool_extended:
            self.extended_body_schema[target_object.name] = {
                "extended_dof": target_object.degrees_of_freedom,
                "current_speed": speed,
                "steering_coupling": steering_angle,
            }
            epiphany_text = (
                f"내가 발로 페달을 밟고 손으로 운전대를 꺾자, 차체가 회전하며 전방 시야의 광류(위상 흐름 {vis_res.temporal_phase_flux:.3f})가 빨라지고, "
                f"엔진음 주파수({base_freq:.1f}Hz)가 치솟으며 내 등에 관성 가속도(G-force {g_force:.2f})가 실렸다. "
                f"이것은 단순한 외부 물체가 아니라 나의 신체 운동이 증폭 전달되는 도구이다. "
                f"**'아, 자동차는 이렇게 움직이는구나!'**라는 새로운 인과적 분기점이 열렸습니다."
            )
        elif target_object.geometry_type == "spherical":
            epiphany_text = (
                f"내가 손으로 전단력을 가하자, 구형 대칭 경계가 지면과의 마찰을 회전으로 전환시키며 {target_object.angular_velocity[2]:.2f} rad/s로 굴러갔다. "
                f"손에 닿는 반작용({reaction:.2f}N)과 시각적 구름 운동의 일치를 통해, "
                f"**'아, 공은 이렇게 굴러가는구나!'**라는 물리적 분기점을 온몸으로 재인식했습니다."
            )
        else:
            epiphany_text = f"자신의 개입({intervention['action_type']})으로 세상의 상태가 {speed:.2f} m/s로 갱신되었습니다."

        return BifurcationEpiphany(
            target_name=target_object.name,
            action_executed=intervention["description"],
            transformation_observed=f"위치/속도 상태 갱신: v={speed:.2f} m/s, w={np.linalg.norm(target_object.angular_velocity):.2f} rad/s",
            synesthetic_feedback={
                "visual_optic_flow": vis_res.temporal_phase_flux,
                "acoustic_pitch_hz": base_freq,
                "somatosensory_g_force": g_force,
                "reaction_force_n": reaction,
            },
            body_schema_extended=is_tool_extended,
            epistemic_branching_insight=epiphany_text,
        )
