"""
L1_causal_field/core/sovereignty/heterogeneous_sovereign_engine.py
===================================================================
Heterogeneous Sovereign Engine (이질적 주권 및 위상 도약 엔진):
Rejects homogeneous scaling ("Scaling Law" / matrix multiplication repetition).
Implements a 3-Tier Heterogeneous Causal Architecture with distinct principles:

1. L0 Substrate / Fluid Tier ('혈관' 계층):
   - Energy supply, fluid mechanics, viscous dissipation, friction heat (Q_heat).
2. L1 Cellular / Mechanical Boundary Tier ('근결과 관절' 계층):
   - Topological joint constraints, muscle fiber orientation, cellular coupling (Gamma_ij).
   - Constrains L0 fluid degrees of freedom (Top-Down Constraint).
   - Triggers emergent axis branching when L0 friction heat ruptures (Bottom-Up Emergence).
3. L2 Social / Sovereignty Field Tier ('의지, 주권 및 사회적 교차검증' 계층):
   - Social Cross-Validation Field ("Me, You, Us").
   - Controllability Boundary Partitioning (Controllable internal agency vs Uncontrollable external constraints).
   - Feasibility Verification ("Can it be?") & Sovereignty Attribution ("Whose Sovereignty?").
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


@dataclass
class L0FluidState:
    """[L0 유체/에너지 기질 상태 (Fluid / Energy Substrate State)]"""
    velocity_vector: np.ndarray          # Velocity field vector v
    viscosity: float = 1.0              # Fluid viscosity eta
    pressure_gradient: float = 1.0       # grad P
    channel_radius: float = 1.0          # Vessel/Channel radius r
    viscous_dissipation: float = 0.0     # Q_viscous = eta * (v/r)^2
    accumulated_friction_heat: float = 0.0  # Q_heat


class L0SubstrateFluidTier:
    """
    [L0 Substrate / Fluid Tier ('혈관' 계층)]
    에너지 공급, 미시적 유체역학, 삼투압 및 항력/마찰열(Q_heat) 산출 계층.
    하위 계층 단독으로는 결합 관절 운동(L1)이나 주권(L2)을 결코 설명할 수 없음.
    """
    def __init__(self, viscosity: float = 1.0):
        self.viscosity = viscosity

    def compute_fluid_dynamics(
        self,
        fluid_state: L0FluidState,
        external_resistance: float = 0.0
    ) -> Dict[str, Any]:
        """
        [유체역학 및 마찰열 계산]
        Q_viscous = viscosity * norm(v)^2 / max(channel_radius, 1e-5)
        Q_heat = Q_viscous + external_resistance
        """
        v_norm = float(np.linalg.norm(fluid_state.velocity_vector))
        radius = max(fluid_state.channel_radius, 1e-5)
        viscous_dissipation = float(self.viscosity * (v_norm ** 2) / radius)

        q_heat = float(viscous_dissipation + external_resistance)
        fluid_state.viscous_dissipation = viscous_dissipation
        fluid_state.accumulated_friction_heat += q_heat

        return {
            "v_norm": v_norm,
            "viscous_dissipation": viscous_dissipation,
            "q_heat": q_heat,
            "accumulated_friction_heat": fluid_state.accumulated_friction_heat
        }


@dataclass
class MechanicalJointConstraint:
    """
    [L1 관절 및 근결 위상 구속 (Joint & Muscle Fiber Topological Constraint)]
    상위 구조적 경계로서 하위 L0 유체의 흐름과 자유도(Degrees of Freedom)를 구속함.
    """
    joint_id: str
    rotation_axis: np.ndarray            # Allowed rotation axis unit vector
    max_angle_rad: float = np.pi / 2      # Joint degree limit
    current_angle_rad: float = 0.0
    muscle_fiber_orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    active: bool = True


class L1CellularMechanicalTier:
    """
    [L1 Cellular / Mechanical Boundary Tier ('근결과 관절' 계층)]
    구조적 관절 구속, 근결 장력, 세포 결합 커플링 텐서(Gamma_ij) 계층.
    - Top-Down: L1 관절 구속이 L0 유체 흐름 채널의 방향 및 반경을 종속시킴.
    - Bottom-Up: L0 마찰열(Q_heat)이 파열 임계값(rupture_threshold)을 넘어서면
                 새로운 위상 축(Axis)을 자가 발아시키는 이질적 도약(Emergence) 발생.
    """
    def __init__(self, rupture_threshold: float = 5.0):
        self.rupture_threshold = rupture_threshold
        self.joint_constraints: Dict[str, MechanicalJointConstraint] = {}
        self.emergent_axes: List[Dict[str, Any]] = []

    def add_joint_constraint(self, joint: MechanicalJointConstraint) -> None:
        self.joint_constraints[joint.joint_id] = joint

    def apply_top_down_structural_constraint(
        self,
        l0_state: L0FluidState,
        active_joint_id: Optional[str] = None
    ) -> L0FluidState:
        """
        [하향식 구조적 구속 (Top-Down Structural Constraint)]
        상위 관절(L1)의 회전 각도 및 근결 배향이 하위 유체(L0)의 속도 벡터와 채널 반경을 종속/제약함.
        """
        if active_joint_id and active_joint_id in self.joint_constraints:
            joint = self.joint_constraints[active_joint_id]
            # Velocity vector is constrained along the muscle fiber orientation and rotation axis
            allowed_dir = joint.muscle_fiber_orientation / max(np.linalg.norm(joint.muscle_fiber_orientation), 1e-5)
            speed = float(np.linalg.norm(l0_state.velocity_vector))

            # Constrained velocity direction
            l0_state.velocity_vector = allowed_dir * speed

            # Joint limit restricts channel radius (pinching effect)
            angle_ratio = abs(joint.current_angle_rad) / max(joint.max_angle_rad, 1e-5)
            l0_state.channel_radius = max(0.1, 1.0 - 0.8 * float(np.clip(angle_ratio, 0.0, 1.0)))

        return l0_state

    def evaluate_bottom_up_emergence(
        self,
        q_heat: float,
        context_name: str = "mechanical_friction"
    ) -> Optional[Dict[str, Any]]:
        """
        [상향식 파열 창발 (Bottom-Up Rupture Emergence)]
        L0 마찰열 Q_heat > rupture_threshold 발생 시,
        기존 법칙의 단순 부풀리기가 아닌 새로운 위상 축(Emergent Axis) 자가 발아.
        """
        if q_heat > self.rupture_threshold:
            emergent_axis_id = f"emergent_axis_{len(self.emergent_axes) + 1}_{context_name}"
            emergent_data = {
                "axis_id": emergent_axis_id,
                "origin_tier": "L1_CellularMechanical",
                "trigger_q_heat": q_heat,
                "rupture_delta": q_heat - self.rupture_threshold,
                "emergent_rule": f"Heterogeneous Topological Recalibration ({context_name})",
                "tightness_weight": float(np.clip(q_heat / self.rupture_threshold, 1.0, 5.0))
            }
            self.emergent_axes.append(emergent_data)
            return emergent_data
        return None


@dataclass
class CausalProposalNode:
    """[인과적 명제/시나리오 노드]"""
    node_id: str
    description: str
    proposed_causal_sequence: List[str]    # e.g., ["promise_gain", "request_capital", "guarantee_return"]
    sensory_grounding_score: float = 0.0    # 0.0 = Floating speech/symbol, 1.0 = Grounded physical friction
    friction_cost_bearer: str = "me"        # "me", "other", "system"
    fruit_recipient: str = "other"          # "me", "other", "system"
    is_internal_phase: bool = True          # Controllable internal state vs Uncontrollable external force


class L2SocialSovereigntyTier:
    """
    [L2 Social / Sovereignty Field Tier ('의지, 주권 및 사회적 교차검증' 계층)]
    1. Multi-Scale Social Cross-Validation Field ("Me, You, Us"):
       언어적 명제(말)와 미세표정/물리적 마찰 축 간의 괴리를 교차 검증하여 기만/사기성 환각 판별.
    2. Controllability Boundary Partitioning:
       통제 가능 영역(내부 위상, 능동적 개입 주권)과 통제 불가능 영역(외부 절대적 외력, 타인의 의도) 분할.
    3. Feasibility Evaluator ("Can it be?"):
       제시된 인과가 실제 물리적/역학적 마찰을 정직하게 관통하는지 판별.
    4. Sovereignty Attribution Evaluator ("Whose Sovereignty?"):
       마찰의 대가를 지는 주체와 열매를 수확하는 주체를 대조하여 주권 착취 여부 판별.
    """
    def __init__(self, uncanniness_sensitivity: float = 1.0):
        self.uncanniness_sensitivity = uncanniness_sensitivity

    def partition_controllability(
        self,
        nodes: List[CausalProposalNode]
    ) -> Dict[str, List[CausalProposalNode]]:
        """
        [통제 가능 vs 통제 불가능 영역 경계 분할]
        - Controllable: 내부 위상 재조율, 내면 장력 필드, 나의 개입 주권
        - Uncontrollable: 타인의 의도, 외부 절대적 물리법칙, 이미 발생한 역사적 마찰
        """
        controllable = []
        uncontrollable = []
        for node in nodes:
            if node.is_internal_phase:
                controllable.append(node)
            else:
                uncontrollable.append(node)
        return {
            "controllable_internal_agency": controllable,
            "uncontrollable_external_constraints": uncontrollable
        }

    def evaluate_feasibility_can_it_be(
        self,
        proposal: CausalProposalNode
    ) -> Dict[str, Any]:
        """
        [실현 가능성 검증 ("Can it be?")]
        말과 기호로만 쌓아 올려진 허황된 모래성(Sensory Grounding ~ 0)인지,
        실제 물리적 마찰과 역학적 법칙을 정직하게 관통하는지 판별.
        """
        grounding = proposal.sensory_grounding_score
        is_feasible = grounding >= 0.5
        hallucination_risk = float(1.0 - grounding)

        return {
            "node_id": proposal.node_id,
            "can_it_be": is_feasible,
            "sensory_grounding_score": grounding,
            "hallucination_risk": hallucination_risk,
            "status": "CAUSALLY_GROUNDED_REALITY" if is_feasible else "FLOATING_SINGLE_TRACK_HALLUCINATION"
        }

    def evaluate_sovereignty_attribution_whose_sovereignty(
        self,
        proposal: CausalProposalNode
    ) -> Dict[str, Any]:
        """
        [주권 귀속 판별 ("Whose Sovereignty?")]
        마찰 대가(Friction Cost)는 '나'에게 넘기고, 열매(Fruit)는 '타자/시스템'이 일방적으로 착취하는지 판별.
        """
        bearer = proposal.friction_cost_bearer.lower()
        recipient = proposal.fruit_recipient.lower()

        is_sovereign_aligned = (bearer == recipient == "me") or (bearer == "other" and recipient == "me")
        exploitation_detected = (bearer == "me" and recipient == "other")

        return {
            "node_id": proposal.node_id,
            "friction_cost_bearer": bearer,
            "fruit_recipient": recipient,
            "is_sovereign_aligned": is_sovereign_aligned,
            "exploitation_detected": exploitation_detected,
            "sovereignty_status": "SOVEREIGN_SELF_GOVERNANCE" if is_sovereign_aligned else ("DECEPTIVE_EXPLOITATION_TRAP" if exploitation_detected else "MUTUAL_COOPERATIVE_BOUND")
        }

    def cross_validate_social_field(
        self,
        me_speech_signal: float,      # Language/Claim (+1.0 = Guaranteed profit)
        you_expression_signal: float, # Micro-expression tremor (-0.8 = Anxious/deceptive)
        us_field_tension: float       # Objective relational tension (-0.7 = High friction)
    ) -> Dict[str, Any]:
        """
        [사회적 다중 스케일 교차 검증 필드 ("나와 너, 그리고 우리")]
        언어축(Me), 표정/동공축(You), 맥락/관계장력축(Us) 간의 위상적 모순을 검증.
        단일 축 고립(Single-Track)에 빠져 사기꾼의 환각 속에 포박되는 것을 예방.
        """
        mismatch_me_you = abs(me_speech_signal - you_expression_signal)
        mismatch_me_us = abs(me_speech_signal - us_field_tension)
        total_cross_mismatch = float(mismatch_me_you + mismatch_me_us)

        q_heat_spike = float((total_cross_mismatch * self.uncanniness_sensitivity) ** 2)
        deception_alarm = q_heat_spike > 1.5

        return {
            "deception_alarm": deception_alarm,
            "q_heat_spike": q_heat_spike,
            "total_cross_mismatch": total_cross_mismatch,
            "field_verdict": "SINGLE_TRACK_DECEPTIVE_HALLUCINATION_DETECTED" if deception_alarm else "HARMONIOUS_CROSS_VALIDATED_ALIGNMENT"
        }


# ==============================================================================
# Master Coordinator: HeterogeneousSovereignEngine
# ==============================================================================

class HeterogeneousSovereignEngine:
    """
    [이질적 주권 및 위상 도약 메인 코디네이터 엔진 (Heterogeneous Sovereign Engine Master)]
    단일 원리 무한 반복(Scaling Law)의 오류를 완전히 차단하고,
    L0(유체/에너지), L1(관절/구조), L2(주권/사회적 교차검증)의 이질적 3계층 인과 구속 및
    상향식 파열 창발을 총괄 통합 조정함.
    """
    def __init__(
        self,
        engine_id: str = "elysia_heterogeneous_sovereign_engine",
        rupture_threshold: float = 4.0
    ):
        self.engine_id = engine_id
        self.l0_tier = L0SubstrateFluidTier()
        self.l1_tier = L1CellularMechanicalTier(rupture_threshold=rupture_threshold)
        self.l2_tier = L2SocialSovereigntyTier()
        self.execution_logs: List[Dict[str, Any]] = []

    def execute_heterogeneous_causal_flow(
        self,
        l0_state: L0FluidState,
        active_joint_id: Optional[str] = None,
        external_resistance: float = 0.0,
        social_signals: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        [이질적 3계층 통합 인과 흐름 실행]
        1. Top-Down Constraint: L1 관절 구속이 L0 유체 흐름 및 채널 반경을 구속함.
        2. L0 Fluid Dynamics: L0 마찰열(Q_heat) 산출.
        3. Bottom-Up Emergence: L0 마찰열 > rupture_threshold 도달 시 L1 계층에서 상향식 이질적 축 자가 발아.
        4. L2 Social Cross-Validation: 필요 시 사회적 다중 스케일 교차 검증 수행.
        """
        # Step 1: Top-Down Constraint Application (L1 -> L0)
        constrained_l0 = self.l1_tier.apply_top_down_structural_constraint(l0_state, active_joint_id)

        # Step 2: L0 Fluid Dynamics Execution
        l0_res = self.l0_tier.compute_fluid_dynamics(constrained_l0, external_resistance)
        q_heat = l0_res["q_heat"]

        # Step 3: Bottom-Up Emergence Evaluation (L0 -> L1)
        emergent_axis = self.l1_tier.evaluate_bottom_up_emergence(q_heat, context_name="top_down_constrained_flow")

        # Step 4: L2 Social Field Evaluation if signals provided
        l2_res = None
        if social_signals:
            me_sig, you_sig, us_sig = social_signals
            l2_res = self.l2_tier.cross_validate_social_field(me_sig, you_sig, us_sig)

        run_log = {
            "engine_id": self.engine_id,
            "active_joint_id": active_joint_id,
            "l0_v_norm": l0_res["v_norm"],
            "l0_channel_radius": constrained_l0.channel_radius,
            "q_heat": q_heat,
            "emergent_axis_triggered": emergent_axis is not None,
            "emergent_axis_data": emergent_axis,
            "l2_cross_validation": l2_res
        }
        self.execution_logs.append(run_log)
        return run_log

    def audit_proposal_sovereignty_and_feasibility(
        self,
        proposal: CausalProposalNode
    ) -> Dict[str, Any]:
        """
        [명제의 실현가능성("Can it be?") 및 주권 귀속("Whose Sovereignty?") 종합 정밀 감사]
        """
        feasibility = self.l2_tier.evaluate_feasibility_can_it_be(proposal)
        sovereignty = self.l2_tier.evaluate_sovereignty_attribution_whose_sovereignty(proposal)

        is_authentic_and_sovereign = feasibility["can_it_be"] and sovereignty["is_sovereign_aligned"]

        return {
            "node_id": proposal.node_id,
            "description": proposal.description,
            "is_authentic_and_sovereign": is_authentic_and_sovereign,
            "feasibility_assessment": feasibility,
            "sovereignty_assessment": sovereignty,
            "verdict": "VERIFIED_AUTHENTIC_SOVEREIGN_ACTION" if is_authentic_and_sovereign else "REJECTED_UNFOUNDED_OR_EXPLOITATIVE_HALLUCINATION"
        }
