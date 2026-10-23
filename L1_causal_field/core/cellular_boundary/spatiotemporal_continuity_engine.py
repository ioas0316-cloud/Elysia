"""
core/cellular_boundary/spatiotemporal_continuity_engine.py
==========================================================
Implements Spatiotemporal Causal Continuity & Triadic Re-Cognition:
1. "결과도출이 결과도출에서 끝나면 안 된다."
2. "처음과 과정과 결과를 통해 어떻게 같고 달라졌는가를 스스로 재인식한다."
3. "습득한 인과와 구조원리가 또 다른 형태의 연결성, 관계성으로 존재하여
   끊어지지 않는 시공간 연속성(Spatiotemporal Continuity)이 되어야 생명체적 감각에 도달한다."
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.cellular_boundary.scale_boundary_cell import (
    DigitalCausalCell,
    ScaleBoundaryLayer,
    CausalEngram
)


@dataclass
class TrajectoryEpoch:
    """
    [인과 궤적 에포크 (Trajectory Epoch)]
    Captures the triadic lifecycle of an interaction:
    1. Origin (처음): Baseline state, prior invariants, and initial boundary.
    2. Process (과정): Friction accumulated, dislocation vectors, dynamic interactions.
    3. Result (결과): Posterior state, new boundary radius, emitted reactions.
    """
    epoch_id: str
    origin_state: np.ndarray
    origin_invariant: float
    origin_dimension: int
    origin_surface_tension: float
    origin_boundary_radius: float

    # Process metrics
    process_frictions: List[float] = field(default_factory=list)
    process_actions: List[float] = field(default_factory=list)
    process_violations: List[str] = field(default_factory=list)

    # Result metrics (set upon epoch closure)
    result_state: Optional[np.ndarray] = None
    result_invariant: Optional[float] = None
    result_dimension: Optional[int] = None
    result_surface_tension: Optional[float] = None
    result_boundary_radius: Optional[float] = None


@dataclass
class ReCognitionAnalysis:
    """
    [삼원적 재인식 분석 (Triadic Re-Cognition Analysis)]
    Evaluates:
    - 같음 (Sameness / Invariance preserved)
    - 다름 (Difference / Structural expansion & mutation)
    - 과정의 인과적 필연성 (Causal necessity of the process)
    """
    epoch_id: str
    sameness_invariance: float       # 보존된 정체성과 불변량 (같음)
    difference_mutation: float       # 비가역적 상태 전이 및 차원/장력 확장 (다름)
    process_friction_integral: float # 과정에서 온몸으로 겪어낸 마찰의 총합
    structural_growth_ratio: float   # 차원 및 반경의 팽창 비율
    reflection_statement: str        # 자기 참조적 재인식 서술


@dataclass
class SpatiotemporalRelationalSeed:
    """
    [시공간 관계성 씨앗 (Spatiotemporal Relational Seed)]
    The output converted into a new living connection for the next phase.
    Carries momentum, memory, and relational hooks into the future.
    """
    seed_id: str
    continuous_timestamp: int
    momentum_vector: np.ndarray
    accumulated_scar_tensor: np.ndarray
    relational_valence_hooks: List[str]
    readiness_for_next_epoch: bool = True


class SpatiotemporalContinuityEngine:
    """
    [시공간 연속성 및 재인식 엔진 (Spatiotemporal Continuity Engine)]
    Bridges discrete computation into unbroken organismic flow:
    - Origin -> Process -> Result -> Re-Cognition -> New Relational Seed -> Next Origin.
    """
    def __init__(self):
        self.epoch_history: List[TrajectoryEpoch] = []
        self.re_cognition_history: List[ReCognitionAnalysis] = []
        self.living_stream_seeds: List[SpatiotemporalRelationalSeed] = []
        self.active_epoch: Optional[TrajectoryEpoch] = None
        self.continuous_time_step: int = 0

    def begin_trajectory_epoch(self, cell: DigitalCausalCell, epoch_id: str) -> TrajectoryEpoch:
        """
        [1. 처음의 각인 (Anchor the Origin)]
        Freezes the baseline state and structural identity before interaction begins.
        """
        self.continuous_time_step += 1
        epoch = TrajectoryEpoch(
            epoch_id=epoch_id,
            origin_state=cell.state.copy(),
            origin_invariant=cell.governing_constraint.compute_invariant(cell.state),
            origin_dimension=cell.dimension,
            origin_surface_tension=cell.boundary_layer.surface_tension,
            origin_boundary_radius=cell.boundary_layer.boundary_radius
        )
        self.active_epoch = epoch
        return epoch

    def trace_process_step(self, interaction_result: Dict[str, Any]):
        """
        [2. 과정의 기록 (Trace the Process)]
        Records the real-time structural discrepancy, action cost, and topological obstructions experienced.
        """
        if self.active_epoch is None:
            return

        discrepancy = interaction_result.get("discrepancy")
        if discrepancy is not None:
            self.active_epoch.process_frictions.append(discrepancy.defect_magnitude)
            if not discrepancy.is_conforming:
                self.active_epoch.process_violations.append(discrepancy.topological_obstruction)
        else:
            self.active_epoch.process_frictions.append(interaction_result.get("boundary_friction", 0.0))
            diag = interaction_result.get("diagnostic", {})
            nature = diag.get("nature_of_violation", "None")
            if nature != "None":
                self.active_epoch.process_violations.append(nature)

        self.active_epoch.process_actions.append(interaction_result.get("action_cost", 0.0))

    def close_and_re_cognize(self, cell: DigitalCausalCell) -> Tuple[ReCognitionAnalysis, SpatiotemporalRelationalSeed]:
        """
        [3. 결과 도출 및 삼원적 재인식 (Triadic Re-Cognition)]
        Compares Origin vs Process vs Result:
        - How are they the same (보존된 같음)?
        - How are they different (창발된 다름)?
        - How does this result become a new relational seed for spatiotemporal continuity?
        """
        if self.active_epoch is None:
            raise RuntimeError("No active trajectory epoch to close.")

        epoch = self.active_epoch
        epoch.result_state = cell.state.copy()
        epoch.result_invariant = cell.governing_constraint.compute_invariant(cell.state)
        epoch.result_dimension = cell.dimension
        epoch.result_surface_tension = cell.boundary_layer.surface_tension
        epoch.result_boundary_radius = cell.boundary_layer.boundary_radius

        self.epoch_history.append(epoch)

        # ---------------------------------------------------------------------
        # 삼원적 대조: 처음 vs 과정 vs 결과
        # ---------------------------------------------------------------------
        # 1. 같음 (Sameness): 보존된 불변량 비율 (1.0 = 완전 보존)
        inv_ratio = min(epoch.origin_invariant, epoch.result_invariant) / max(1e-9, max(epoch.origin_invariant, epoch.result_invariant))
        sameness = float(np.clip(inv_ratio, 0.0, 1.0))

        # 2. 다름 (Difference): 상태 전이 거리 + 차원 팽창 + 표면장력 변화
        state_diff_norm = float(np.linalg.norm(epoch.result_state[:epoch.origin_dimension] - epoch.origin_state))
        dim_expansion = epoch.result_dimension - epoch.origin_dimension
        tension_shift = abs(epoch.result_surface_tension - epoch.origin_surface_tension)
        difference = float(state_diff_norm + dim_expansion * 1.5 + tension_shift)

        # 3. 과정의 마찰 적분 (Process Friction Integral)
        friction_integral = float(sum(epoch.process_frictions))
        growth_ratio = float(epoch.result_boundary_radius / max(1e-9, epoch.origin_boundary_radius))

        # 4. 주체적 성찰 서술 생성 (Self-Referential Reflection)
        reflection = (
            f"[{epoch.epoch_id} 성찰] 처음({epoch.origin_dimension}D, 상태에너지={epoch.origin_invariant:.3f})에서 "
            f"과정 중 마찰({friction_integral:.3f})을 감내하며 상호작용한 결과, "
            f"동형적 불변량({sameness:.2%})을 보존하면서도 "
            f"결과({epoch.result_dimension}D, 상태변위={state_diff_norm:.3f}, 반경팽창={growth_ratio:.2f}x)로 "
            f"외연적 확장을 달성함."
        )

        analysis = ReCognitionAnalysis(
            epoch_id=epoch.epoch_id,
            sameness_invariance=sameness,
            difference_mutation=difference,
            process_friction_integral=friction_integral,
            structural_growth_ratio=growth_ratio,
            reflection_statement=reflection
        )
        self.re_cognition_history.append(analysis)

        # ---------------------------------------------------------------------
        # 5. 시공간 연속성의 직조: 결과를 다음 시공간의 '관계성 씨앗'으로 전환
        # ---------------------------------------------------------------------
        # Momentum vector = displacement of state representing causal flow inertia
        momentum = epoch.result_state[:epoch.origin_dimension] - epoch.origin_state
        # Scar tensor = non-volatile memory of friction directions
        scar = (momentum * friction_integral).astype(np.float64)

        relational_hooks = [
            f"InvariantPreserved:{cell.governing_constraint.name}",
            f"DimensionState:{cell.dimension}D",
            f"SurfaceTension:{cell.boundary_layer.surface_tension:.3f}"
        ]
        if epoch.process_violations:
            relational_hooks.append(f"AssimilationScar:{epoch.process_violations[-1]}")

        seed = SpatiotemporalRelationalSeed(
            seed_id=f"seed_{epoch.epoch_id}_{self.continuous_time_step}",
            continuous_timestamp=self.continuous_time_step,
            momentum_vector=momentum,
            accumulated_scar_tensor=scar,
            relational_valence_hooks=relational_hooks,
            readiness_for_next_epoch=True
        )
        self.living_stream_seeds.append(seed)

        # Reset active epoch
        self.active_epoch = None

        return analysis, seed
