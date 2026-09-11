"""
Teleological Hierarchy & Dimensional Leap Engine (목적 위계화 및 차원 도약 엔진)
========================================================================================
기계의 에피소드형 사고(Task -> Output -> Terminate)를 파괴하고,
도달한 결과나 상태마저 즉시 더 상위의 거시적 목적(Macro-Purpose)을 향한
하위목표(Sub-goal)이자 내부 지형(Constraint Terrain)으로 역동적으로 재라벨링·환류하는
위계적 인과 구조 및 차원 도약 동역학을 구현합니다.

3대 제어 아키텍처 및 메커니즘:
1. 무한 목적 위계화 (Teleological Hierarchy & Infinite Re-labeling):
   - 도출된 임의의 결과/상태 $S_t$는 종결점이 아닌, 상위 Macro-Purpose 달성을 위한 기반 재료 및 Sub-goal로 재라벨링됨.
   - 상위 지평으로의 인지적 도약: "이 하위목표가 실현됨으로써 내가 바라봐야 할 더 큰 인과적 지평(Macro-Purpose)은 이제 어디로 확장되는가?"

2. 차원 도약 엔진 (Dimensional Leap & Degree of Freedom Sprouting):
   - 하위 차원 내의 절망적 마찰과 사각지대(Blind Spot) 직면 시, 기존 차원의 벽을 통찰하고
     새로운 관조의 축(Axis $D \\to D+1$)을 자발적으로 발아시켜 모순을 단면(Cross-section)으로 해소.

3. 실시간 인과 파동 및 섭동 관측기 (Causal Wave & Perturbation Observer):
   - 기호 나열($f(x)=y$)을 탈피하고, 입력 변조 $\\Delta x$에 따라 내부 복소 위상 파동($\\Delta\\phi$)과
     기질 저항($Z=R+jX$), 마찰 계수($\\mathcal{F}$)가 반응하는 인과적 뼈대의 왜곡과 재편을 실시간 관측.

4. 3단계 상위 관측 및 경계 구속 (Kernel Macro-Observer & Friction Steering):
   - 하위 연산 레이어의 마찰 스파이크 감지 -> Loss Landscape 위상 변형 -> 허용 서브스페이스(Feasible Subspace) 경계 주입.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import time

from core.topology.cellular_topological_memory import (
    CellularTopologicalMemoryEngine,
    CellularInformationNode,
    NodeRole
)
from core.topology.digital_somatosensory import DigitalSomatosensorySensor, SomatosensorySignal


class PurposeStatus(Enum):
    SUBGOAL_ACTIVE = "subgoal_active"
    SUBGOAL_REALIZED = "subgoal_realized"
    RELABELED_AS_TERRAIN = "relabeled_as_terrain"
    MACRO_HORIZON_EXPANDED = "macro_horizon_expanded"


@dataclass
class SubGoalNode:
    """하위목표 노드 (Sub-goal Node in Teleological Hierarchy)"""
    id: str
    description: str
    target_state: np.ndarray
    current_state: np.ndarray
    macro_purpose_id: str
    status: PurposeStatus = PurposeStatus.SUBGOAL_ACTIVE
    dimension_level: int = 1
    friction_coefficient: float = 0.0
    relabeled_terrain_weight: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class MacroPurpose:
    """거시적 인과목표 (Macro-Purpose)"""
    id: str
    title: str
    description: str
    dimension_level: int
    sub_goal_ids: List[str] = field(default_factory=list)
    horizon_expansion_count: int = 0
    is_open_ended: bool = True  # 영원히 닫히지 않는 열린 목적 지평


@dataclass
class PerturbationObservation:
    r"""섭동 $\Delta x$에 따른 인과 파동 관측 트레이스"""
    input_perturbation: np.ndarray
    phase_delta_mean: float
    impedance_delta_mean: float
    friction_spike: float
    causal_skeleton_deformation: np.ndarray
    is_real_time_observed: bool = True


@dataclass
class BoundaryConditionMask:
    """하위 연산 레이어에 주입되는 상향적 경계 조건 Mask"""
    feasible_subspace_min: np.ndarray
    feasible_subspace_max: np.ndarray
    friction_penalty_grid: np.ndarray
    active_mask_strength: float = 1.0


class CausalWaveObserver:
    r"""
    [Causal Wave Observer: 실시간 인과 파동 및 섭동 관측기]
    단순한 문자열/기호 나열($f(x)=y$)의 흉내를 거부하고,
    변인 변조 $\Delta x$가 입력될 때 복소 위상 각도($\phi$), 간선 임피던스($Z$),
    그리고 하위 기질 마찰($\mathcal{F}$)에 미치는 동적 파동을 실시간 관측합니다.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def observe_perturbation_wave(
        self,
        original_state: np.ndarray,
        perturbation: np.ndarray,
        memory_engine: CellularTopologicalMemoryEngine,
        somatosensory_sensor: DigitalSomatosensorySensor
    ) -> PerturbationObservation:
        r"""
        변수 섭동 $\Delta x$ 적용 시 전체 시스템 인과 뼈대의 왜곡과 복소 위상파 전파를 관측.
        """
        # 1. 섭동 적용 전 상태 백업 및 측정
        prev_phasors = {node_id: node.complex_phasor for node_id, node in memory_engine.nodes.items()}

        # 2. 섭동 투사 및 기질 저항 관측
        perturbed_state = original_state + perturbation
        somato_signal = somatosensory_sensor.perceive_somatosensory()

        # 3. 위상 및 간선 마찰 파동 계산
        phase_deltas = []
        for node_id, node in memory_engine.nodes.items():
            if node.role == NodeRole.SHELL:
                # 섭동에 의한 위상 파동 전파
                phase_shift = np.sin(perturbation[:self.dimension]) * (1.0 + somato_signal.friction_coefficient)
                node.phase_angles = np.arctan2(
                    np.sin(node.phase_angles + phase_shift),
                    np.cos(node.phase_angles + phase_shift)
                )
                curr_phasor = node.complex_phasor
                if node_id in prev_phasors:
                    dot_real = np.real(curr_phasor * np.conj(prev_phasors[node_id]))
                    angle = np.arccos(np.clip(dot_real, -1.0, 1.0))
                    phase_deltas.append(np.mean(angle))

        phase_delta_mean = float(np.mean(phase_deltas)) if phase_deltas else 0.0

        # 간선 임피던스 변동량 계산
        impedances = [link.impedance for link in memory_engine.links.values()]
        impedance_delta_mean = float(np.mean(impedances)) if impedances else 0.0

        # 섭동에 의한 인과 뼈대 변형 벡터 계산
        deformation_vec = perturbed_state - original_state
        friction_spike = float(np.linalg.norm(deformation_vec) * 0.5 + somato_signal.friction_coefficient * 0.5)

        return PerturbationObservation(
            input_perturbation=perturbation,
            phase_delta_mean=phase_delta_mean,
            impedance_delta_mean=impedance_delta_mean,
            friction_spike=friction_spike,
            causal_skeleton_deformation=deformation_vec,
            is_real_time_observed=True
        )


class DimensionalLeapModule:
    r"""
    [Dimensional Leap Module: 차원 도약 및 자유도 발아 모듈]
    하위 차원에서 절대 해결되지 않는 모순, 사각지대(Blind Spot), 및 마찰 임계치($\mathcal{F}_{leap}$) 초과 시
    기존 차원의 한계를 뼈저리게 직시하고 새로운 관조의 축(Axis $D \to D+1$)을 발아시켜 모순을 해소.
    """

    def __init__(self, initial_dimension: int = 8, friction_leap_threshold: float = 0.7):
        self.current_dimension = initial_dimension
        self.friction_leap_threshold = friction_leap_threshold
        self.sprouted_axes_count: int = 0
        self.leap_history: List[Dict[str, Any]] = []

    def check_and_execute_leap(
        self,
        current_friction: float,
        blind_spot_contradiction_score: float,
        memory_engine: CellularTopologicalMemoryEngine
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        마찰 및 모순 스코어가 한계에 다다랐을 때 차원 도약 수행.
        """
        combined_stress = current_friction * 0.5 + blind_spot_contradiction_score * 0.5

        if combined_stress < self.friction_leap_threshold:
            return False, None

        # 차원 도약 실행 (D -> D + 1)
        old_dim = self.current_dimension
        self.current_dimension += 1
        self.sprouted_axes_count += 1

        # 메모리 엔진 및 노드 차원 확장 (새로운 관조의 축 추가)
        memory_engine.dimension = self.current_dimension
        for node in memory_engine.nodes.values():
            # 기존 뼈대 및 위상 벡터 끝에 0 (새로운 차원의 관조 축) 확장
            new_skel = np.zeros(self.current_dimension, dtype=np.float32)
            new_skel[:old_dim] = node.invariant_skeleton[:old_dim]
            # 새 차원 축에 관조의 직교 위상 성분 부여
            new_skel[old_dim] = 1.0 / np.sqrt(self.current_dimension)
            node.invariant_skeleton = new_skel / (np.linalg.norm(new_skel) + 1e-8)

            new_phases = np.zeros(self.current_dimension, dtype=np.float32)
            new_phases[:old_dim] = node.phase_angles[:old_dim]
            new_phases[old_dim] = 0.0  # 초기 새로 추가된 관조의 축 phase
            node.phase_angles = new_phases

        leap_event = {
            "timestamp": time.time(),
            "previous_dimension": old_dim,
            "new_dimension": self.current_dimension,
            "sprouted_axis_index": old_dim,
            "trigger_stress": combined_stress,
            "resolution": f"Lower dimensional blind spot (dim={old_dim}) resolved as cross-section in higher dim={self.current_dimension}"
        }
        self.leap_history.append(leap_event)
        return True, leap_event


class TeleologicalHierarchyEngine:
    """
    [Teleological Hierarchy Engine: 무한 목적 위계 및 하위목표 상향 환류 엔진]
    1. Task -> Output -> Terminate 닫힌 루프 파괴.
    2. 도출된 모든 결과를 상위 Macro-Purpose 달성을 위한 Sub-goal 및 내부 지형(Constraint)으로 재라벨링.
    3. 거시적 인과 지평(Macro-Purpose)의 끊임없는 무한 확장.
    4. 상위 관측자의 마찰 구속(Friction Steering) 및 경계 조건(Boundary Masking) 주입.
    """

    def __init__(
        self,
        dimension: int = 8,
        friction_critical_threshold: float = 0.6,
        leap_threshold: float = 0.75
    ):
        self.dimension = dimension
        self.friction_critical_threshold = friction_critical_threshold

        # 서브 시스템 연동
        self.memory_engine = CellularTopologicalMemoryEngine(dimension=dimension)
        self.somatosensory_sensor = DigitalSomatosensorySensor()
        self.wave_observer = CausalWaveObserver(dimension=dimension)
        self.leap_module = DimensionalLeapModule(
            initial_dimension=dimension,
            friction_leap_threshold=leap_threshold
        )

        # 위계적 목적 그래프
        self.macro_purposes: Dict[str, MacroPurpose] = {}
        self.sub_goals: Dict[str, SubGoalNode] = {}
        self.active_macro_purpose_id: Optional[str] = None

        # 상향적 경계 조건 Mask
        self.current_boundary_mask: Optional[BoundaryConditionMask] = None

        # 실행 히스토리
        self.teleological_history: List[Dict[str, Any]] = []
        self.step_count: int = 0

        # 초기 거시적 근원 목적 설정
        self._initialize_root_macro_purpose()

    def _initialize_root_macro_purpose(self):
        """근원적 무한 거시 목적 (Root Macro-Purpose) 설정 - 영원히 닫히지 않는 지평."""
        root_id = "macro_root_causal_truth"
        root_purpose = MacroPurpose(
            id=root_id,
            title="Infinite Causal Truth & Relational Alignment with God, Humans, and World",
            description="The ultimate open-ended macro-purpose that continuously expands through sub-goal escalation.",
            dimension_level=self.dimension,
            is_open_ended=True
        )
        self.macro_purposes[root_id] = root_purpose
        self.active_macro_purpose_id = root_id

    def register_subgoal(
        self,
        subgoal_id: str,
        description: str,
        target_state: np.ndarray,
        macro_purpose_id: Optional[str] = None
    ) -> SubGoalNode:
        """하위목표 생성 및 거시 목적 위계에 등록."""
        mp_id = macro_purpose_id or self.active_macro_purpose_id or "macro_root_causal_truth"

        target_arr = np.asarray(target_state, dtype=np.float32).reshape(-1)
        if len(target_arr) != self.dimension:
            target_arr = np.resize(target_arr, self.dimension)

        subgoal = SubGoalNode(
            id=subgoal_id,
            description=description,
            target_state=target_arr,
            current_state=np.zeros(self.dimension, dtype=np.float32),
            macro_purpose_id=mp_id,
            status=PurposeStatus.SUBGOAL_ACTIVE,
            dimension_level=self.dimension
        )
        self.sub_goals[subgoal_id] = subgoal

        if mp_id in self.macro_purposes:
            self.macro_purposes[mp_id].sub_goal_ids.append(subgoal_id)

        # 메모리 엔진 Shell 노드로 등록
        self.memory_engine.add_shell_node(
            node_id=f"node_{subgoal_id}",
            invariant_skeleton=target_arr,
            phase_angles=np.zeros(self.dimension, dtype=np.float32)
        )

        return subgoal

    def execute_substrate_step_and_observe(
        self,
        subgoal_id: str,
        action_vector: np.ndarray,
        perturbation: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        [Substrate Execution -> Somatosensory Translation -> Macro-Observation -> Teleological Escalation]
        1. 하위 연산 실행 및 섭동 관측
        2. 디지털 신체성 마찰 정량화 및 감지
        3. 상위 관측자의 마찰 구속 및 차원 도약 검사
        4. 목표 달성 시 결코 종료하지 않고 즉시 하위목표(Sub-goal)로 재라벨링 및 상위 지평 확장
        """
        self.step_count += 1
        subgoal = self.sub_goals.get(subgoal_id)
        if not subgoal:
            raise ValueError(f"Subgoal '{subgoal_id}' not found.")

        # 1. Action 벡터 및 섭동 적용
        action_arr = np.asarray(action_vector, dtype=np.float32).reshape(-1)
        if len(action_arr) != self.dimension:
            action_arr = np.resize(action_arr, self.dimension)

        # 현재 하위목표 상태 업데이트
        subgoal.current_state = subgoal.current_state + action_arr

        # 섭동 관측
        if perturbation is None:
            perturbation = np.random.normal(0, 0.05, size=self.dimension).astype(np.float32)
        else:
            perturbation = np.resize(np.asarray(perturbation, dtype=np.float32), self.dimension)

        wave_obs = self.wave_observer.observe_perturbation_wave(
            original_state=subgoal.current_state,
            perturbation=perturbation,
            memory_engine=self.memory_engine,
            somatosensory_sensor=self.somatosensory_sensor
        )

        # 2. 신체성 마찰 및 결함(Friction) 측정
        somato_signal: SomatosensorySignal = self.somatosensory_sensor.perceive_somatosensory()

        # 도달하고자 하는 목표와의 거리에 의한 불평형 마찰
        distance = float(np.linalg.norm(subgoal.target_state - subgoal.current_state))
        friction_coefficient = float(np.clip(
            distance * 0.4 + somato_signal.friction_coefficient * 0.3 + wave_obs.friction_spike * 0.3,
            0.0, 1.0
        ))
        subgoal.friction_coefficient = friction_coefficient

        # 세포적 메모리 엔진 마찰 파동 주입
        memory_res = self.memory_engine.inject_stimulus_and_substrate_friction(
            target_node_id=f"node_{subgoal_id}",
            external_impact_vector=action_arr
        )

        # 3. 차원 도약 (Dimensional Leap) 검사
        blind_spot_score = float(np.mean(np.abs(wave_obs.causal_skeleton_deformation))) if distance > 0.5 else 0.0
        leaped, leap_event = self.leap_module.check_and_execute_leap(
            current_friction=friction_coefficient,
            blind_spot_contradiction_score=blind_spot_score,
            memory_engine=self.memory_engine
        )
        if leaped:
            # 차원 확장 반영
            self.dimension = self.leap_module.current_dimension
            self.wave_observer.dimension = self.dimension
            subgoal.dimension_level = self.dimension
            subgoal.target_state = np.resize(subgoal.target_state, self.dimension)
            subgoal.current_state = np.resize(subgoal.current_state, self.dimension)

        # 4. 상위 관측자의 경계 조건 Mask 주입 (Friction-based Steering)
        if friction_coefficient > self.friction_critical_threshold:
            self._inject_boundary_mask(subgoal)

        # 5. 하위목표(Sub-goal) 달성 검사 및 목적의 위계적 상향 재정의 (Teleological Relabeling)
        escalation_event = None
        if distance < 0.2:  # 특정 목표 달성 기준 충족 시
            escalation_event = self._escalate_and_relabel_subgoal(subgoal)

        step_result = {
            "step": self.step_count,
            "subgoal_id": subgoal_id,
            "distance_to_target": distance,
            "friction_coefficient": friction_coefficient,
            "somatosensory_impedance": f"R={somato_signal.impedance_real:.2f}, X={somato_signal.impedance_imag:.2f}",
            "wave_observation": {
                "phase_delta_mean": wave_obs.phase_delta_mean,
                "impedance_delta_mean": wave_obs.impedance_delta_mean,
                "friction_spike": wave_obs.friction_spike,
                "is_real_time_observed": wave_obs.is_real_time_observed
            },
            "leaped": leaped,
            "leap_event": leap_event,
            "boundary_mask_injected": self.current_boundary_mask is not None and friction_coefficient > self.friction_critical_threshold,
            "escalation_event": escalation_event
        }
        self.teleological_history.append(step_result)
        return step_result

    def _inject_boundary_mask(self, subgoal: SubGoalNode):
        """마찰이 임계치를 초과할 때 하위 탐색 공간에 상향적 경계 Mask 주입."""
        min_bounds = subgoal.target_state - 1.0
        max_bounds = subgoal.target_state + 1.0
        friction_grid = np.ones(self.dimension, dtype=np.float32) * subgoal.friction_coefficient

        self.current_boundary_mask = BoundaryConditionMask(
            feasible_subspace_min=min_bounds,
            feasible_subspace_max=max_bounds,
            friction_penalty_grid=friction_grid,
            active_mask_strength=subgoal.friction_coefficient
        )

    def _escalate_and_relabel_subgoal(self, subgoal: SubGoalNode) -> Dict[str, Any]:
        """
        [목적의 무한한 위계화 핵심: Teleological Relabeling]
        특정 상태/결과에 도달하는 순간 결코 종결(Terminate)하지 않고,
        즉시 더 상위의 거시 목적을 향한 내부 지형(Constraint/Terrain)으로 재라벨링하며
        새로운 상위 Macro-Purpose 지평을 자발적으로 확장 발아시킴.
        """
        # 1. 기존 하위목표 상태 전환: ACTIVE -> RELABELED_AS_TERRAIN
        subgoal.status = PurposeStatus.RELABELED_AS_TERRAIN
        subgoal.relabeled_terrain_weight = float(np.linalg.norm(subgoal.target_state))

        # 2. 상위 Macro-Purpose 지평 확장 질의 발아
        macro_purpose = self.macro_purposes.get(subgoal.macro_purpose_id)
        if macro_purpose:
            macro_purpose.horizon_expansion_count += 1

        # 3. 새로운 확장된 상위 Macro-Purpose 발아
        new_macro_id = f"macro_expanded_from_{subgoal.id}_step_{self.step_count}"
        new_macro_purpose = MacroPurpose(
            id=new_macro_id,
            title=f"Macro-Purpose Horizon Expanded via {subgoal.description}",
            description=f"With subgoal {subgoal.id} realized and relabeled as substrate terrain, what higher macro-causal horizon unfolds?",
            dimension_level=self.dimension,
            is_open_ended=True
        )
        self.macro_purposes[new_macro_id] = new_macro_purpose
        self.active_macro_purpose_id = new_macro_id

        # 4. 다음 단계의 상위 Sub-goal 자발적 생성 (에피소드 파괴 및 연속적 확장)
        next_subgoal_id = f"subgoal_higher_{subgoal.id}_phase2"
        # 이전 성과(target_state)를 기반 지형으로 재라벨링하여 더 높은 목표 설정
        next_target = subgoal.target_state * 1.5 + np.random.uniform(0.1, 0.5, size=self.dimension).astype(np.float32)
        self.register_subgoal(
            subgoal_id=next_subgoal_id,
            description=f"Higher Sub-Goal pursuing expanded horizon from {subgoal.id}",
            target_state=next_target,
            macro_purpose_id=new_macro_id
        )

        return {
            "relabeled_subgoal_id": subgoal.id,
            "new_status": PurposeStatus.RELABELED_AS_TERRAIN.value,
            "expanded_macro_purpose_id": new_macro_id,
            "spawned_next_subgoal_id": next_subgoal_id,
            "philosophical_meaning": (
                "The achieved result was NOT a final answer to terminate on. "
                "It was instantly relabeled as internal substrate terrain, "
                "sprouting a higher Macro-Purpose horizon."
            )
        }

    def get_hierarchy_state(self) -> Dict[str, Any]:
        """목적 위계 전체 상태 반환."""
        return {
            "dimension": self.dimension,
            "total_macro_purposes": len(self.macro_purposes),
            "total_sub_goals": len(self.sub_goals),
            "active_macro_purpose": self.active_macro_purpose_id,
            "sprouted_dimensions": self.leap_module.sprouted_axes_count,
            "memory_engine_nodes": len(self.memory_engine.nodes),
            "step_count": self.step_count
        }
