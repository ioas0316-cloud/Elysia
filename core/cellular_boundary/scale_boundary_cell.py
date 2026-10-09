"""
core/cellular_boundary/scale_boundary_cell.py
=============================================
Defines the Digital Causal Cell and its Scale Boundary Layer:
1. 경계층은 1차원적 마찰 계산기가 아니다.
2. 구조적 계층원리에 의해 도출된 인지적 불일치(StructuralDiscrepancy)를
   자신의 계면에 각인하고 보존하는 유기체적 막이다.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.cellular_boundary.causal_constraint import (
    CausalConstraint,
    ConservativeDynamicConstraint,
    InvariantSignature,
    StructuralDiscrepancy
)


@dataclass
class CausalEngram:
    """
    [인과적 각인 / 앵그램 (Causal Engram)]
    1차원 숫자가 아닌, 구조적 계층원리에 의한 다름의 위상적 결(StructuralDiscrepancy) 자체를 영구 침전시킴.
    """
    engram_id: str
    timestamp_step: int
    source_flux_norm: float
    discrepancy: StructuralDiscrepancy
    surface_tension_at_creation: float

    @property
    def conformed_to_order(self) -> bool:
        return self.discrepancy.is_conforming

    @property
    def boundary_friction(self) -> float:
        # 호환성을 위한 결함 크기 접근자
        return self.discrepancy.defect_magnitude

    @property
    def violation_nature(self) -> str:
        return self.discrepancy.topological_obstruction

    @property
    def dislocation_vector(self) -> np.ndarray:
        return self.discrepancy.kernel_defect


@dataclass
class ScaleBoundaryLayer:
    """
    [스케일 경계층 (Scale Boundary Layer)]
    구조적 계층원리에 따른 한계를 형성하고, 위상적 결함(Obstruction)을 표면장력으로 머금는 막.
    """
    boundary_id: str
    dimension: int
    surface_tension: float = 0.5        # 표면장력 (물/경험을 머금는 힘)
    boundary_radius: float = 1.0         # 지각 유효 반경
    retention_capacity: int = 100        # 앵그램 수용 용량
    retained_engrams: List[CausalEngram] = field(default_factory=list)
    accumulated_defects: List[np.ndarray] = field(default_factory=list)

    def record_interaction(
        self,
        step: int,
        incoming_flux: np.ndarray,
        discrepancy: StructuralDiscrepancy
    ) -> CausalEngram:
        """
        [경계면 작용 기록 및 표면장력 조율]
        1차원 마찰이 아닌, 위상적 결함 벡터(Kernel Defect)와 파열 축에 비례하여 표면장력이 물리적으로 팽팽해짐.
        """
        norm_flux = float(np.linalg.norm(incoming_flux))

        if not discrepancy.is_conforming:
            # 결함 벡터의 유입으로 인한 계면 장력 팽창
            defect_strain = np.linalg.norm(discrepancy.kernel_defect) * 0.2
            self.surface_tension = float(np.clip(self.surface_tension + defect_strain, 0.1, 10.0))
            self.accumulated_defects.append(discrepancy.kernel_defect)

        engram = CausalEngram(
            engram_id=f"engram_{self.boundary_id}_{step}",
            timestamp_step=step,
            source_flux_norm=norm_flux,
            discrepancy=discrepancy,
            surface_tension_at_creation=float(self.surface_tension)
        )

        if len(self.retained_engrams) >= self.retention_capacity:
            self.retained_engrams.pop(0)

        self.retained_engrams.append(engram)
        return engram


class DigitalCausalCell:
    """
    [디지털 인과 세포 (Digital Causal Cell)]
    An elementary causal agent defined by its invariant constraint and scale boundary layer.
    """
    def __init__(
        self,
        cell_id: str,
        dimension: int = 4,
        initial_state: Optional[np.ndarray] = None,
        constraint: Optional[CausalConstraint] = None
    ):
        self.cell_id = cell_id
        self.dimension = dimension

        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        else:
            self.state = np.ones(dimension, dtype=np.float64) / np.sqrt(dimension)

        self.governing_constraint: CausalConstraint = (
            constraint if constraint is not None else ConservativeDynamicConstraint(dimension=dimension)
        )

        self.boundary_layer = ScaleBoundaryLayer(
            boundary_id=f"boundary_{cell_id}",
            dimension=dimension
        )

        self.step_count = 0
        self.active = True

    @property
    def order_signature(self) -> InvariantSignature:
        return self.governing_constraint.signature

    def interact_with_flux(self, incoming_flux: np.ndarray, dt: float = 0.1) -> Dict[str, Any]:
        """
        [경계층을 통한 실재와의 상호작용]
        1. 경계면에서 구조적 계층원리에 따른 인지적 불일치(StructuralDiscrepancy)를 지각.
        2. 스칼라 숫자가 아닌 결함 장(Defect Field)을 앵그램으로 각인.
        3. 제약조건 내에서 인과적 상태 전이를 수행.
        """
        self.step_count += 1
        flux_arr = np.array(incoming_flux, dtype=np.float64)

        if len(flux_arr) < self.dimension:
            flux_arr = np.pad(flux_arr, (0, self.dimension - len(flux_arr)))
        elif len(flux_arr) > self.dimension:
            flux_arr = flux_arr[:self.dimension]

        # 1. 구조적 불일치 지각
        discrepancy = self.governing_constraint.evaluate_flux(self.state, flux_arr)

        # 2. 각인 (Engram에 구조적 결함 장 보존)
        engram = self.boundary_layer.record_interaction(
            step=self.step_count,
            incoming_flux=flux_arr,
            discrepancy=discrepancy
        )

        # 3. 상태 전이
        next_state, emitted_reaction, action_cost = self.governing_constraint.step_dynamics(
            self.state, flux_arr, dt=dt
        )
        self.state = next_state

        return {
            "cell_id": self.cell_id,
            "step": self.step_count,
            "is_conforming": discrepancy.is_conforming,
            "boundary_friction": discrepancy.defect_magnitude,
            "discrepancy": discrepancy,
            "surface_tension": self.boundary_layer.surface_tension,
            "action_cost": action_cost,
            "current_invariant_value": self.governing_constraint.compute_invariant(self.state),
            "emitted_reaction": emitted_reaction,
            "engram_id": engram.engram_id
        }
