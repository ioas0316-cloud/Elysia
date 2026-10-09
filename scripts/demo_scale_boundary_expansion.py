#!/usr/bin/env python3
"""
scripts/demo_scale_boundary_expansion.py
========================================
Interactive demonstration of the Scale Boundary Layer and Meta-Order Expansion:
"경계층은 제약조건이고, 제약조건이란 인과구조에 의해 인과결과를 도출/관측하는 원리다."
"한계로 존재할 때 그렇지 않은 것들을 마찰로 감지하고,
 질서 외의 질서가 존재함을 관계성을 통해 연결, 도출하여 외연적으로 확장한다."
"""

import sys
import os
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.cellular_boundary.scale_boundary_cell import DigitalCausalCell
from core.cellular_boundary.boundary_expansion_engine import MetaOrderExpansionEngine


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print(" PROJECT ELYSIA :: SCALE BOUNDARY LAYER & META-ORDER EXPANSION DEMO ")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. 세포의 탄생: 엄격한 보존적 역학 제약조건(Order A)을 지닌 디지털 세포
    # -------------------------------------------------------------------------
    print("\n[STEP 1] 최초의 디지털 인과 세포 생성 (고유한 제약조건 = 질서 A)")
    cell = DigitalCausalCell(cell_id="AlphaCell_01", dimension=4)
    print(f"  - Cell ID              : {cell.cell_id}")
    print(f"  - Governing Constraint : {cell.governing_constraint.name}")
    print(f"  - Conserved Invariant  : {cell.order_signature.conserved_quantity_name}")
    print(f"  - Initial State (4D)   : {np.round(cell.state, 3)}")
    print(f"  - Surface Tension (T)  : {cell.boundary_layer.surface_tension:.3f}")
    print(f"  - Boundary Radius (R)  : {cell.boundary_layer.boundary_radius:.3f}")

    # -------------------------------------------------------------------------
    # 2. 질서 내부의 조화로운 상호작용 (최저작용의 원리: 마찰 0)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] 질서 내부의 조화로운 입력 유입 (보존적 회전 자극)")
    # Symplectic rotation preserves Hamiltonian energy
    harmonic_flux = np.array([-cell.state[1], cell.state[0], -cell.state[3], cell.state[2]]) * 0.1
    res1 = cell.interact_with_flux(harmonic_flux)
    print(f"  - Input Conforming?    : {res1['is_conforming']}")
    print(f"  - Boundary Friction    : {res1['boundary_friction']:.6f} (마찰 없음: 최저작용)")
    print(f"  - Action Cost          : {res1['action_cost']:.6f}")
    print(f"  - Surface Tension (T)  : {res1['surface_tension']:.3f} (변함없음)")

    # -------------------------------------------------------------------------
    # 3. 한계의 자각: '그렇지 않은 것(소산적 열역학 플럭스)'과의 마찰
    # -------------------------------------------------------------------------
    print("\n[STEP 3] 한계의 자각: 자신의 제약조건에 어긋나는 외부 자극('그렇지 않은 것') 유입!")
    print("  >> 외부로부터 강한 에너지 감쇄/소산(Dissipative flux)이 유입됩니다...")

    for i in range(4):
        # Dissipative damping flux that violates strict conservative invariance
        dissipative_flux = -cell.state * (0.6 + i * 0.2)
        res_violation = cell.interact_with_flux(dissipative_flux)
        print(f"  [충돌 {i+1}] Conforming: {res_violation['is_conforming']} | "
              f"Friction: {res_violation['boundary_friction']:.4f} | "
              f"Surface Tension: {res_violation['surface_tension']:.4f} | "
              f"Violation: {res_violation['diagnostic']['nature_of_violation']}")

    print(f"\n  - 경계층에 고인 앵그램 수 : {len(cell.boundary_layer.retained_engrams)}개 (물이 빠져나가지 않고 고임!)")
    print(f"  - 누적된 경계 표면장력  : {cell.boundary_layer.surface_tension:.4f}")

    # -------------------------------------------------------------------------
    # 4. 관계성을 통한 '질서 외의 질서' 역추출 및 결합원리적 확장
    # -------------------------------------------------------------------------
    print("\n[STEP 4] 관계성을 통한 '질서 외의 질서' 도출 및 상위 스케일 경계층 확장")
    engine = MetaOrderExpansionEngine(friction_threshold_for_expansion=1.0)

    inspection = engine.inspect_cell_boundary(cell)
    print(f"  - Boundary Status      : {inspection['status']}")
    print(f"  - Total Discrepancy    : {inspection['total_recent_friction']:.4f}")

    print("  >> 경계면의 변형 궤적(Dislocation Engrams)으로부터 외부 생성 메커니즘 역추출 중...")
    inferred_order = engine.infer_external_order(cell)
    print(f"  - 도출된 외부 질서      : {inferred_order.name} ({inferred_order.signature.symmetry_group})")

    print("\n  >> 결합원리적 동형성(Combinatorial Isomorphism)을 통해 상위 거시 세포로 융합 발동!")
    expansion_res = engine.execute_combinatorial_expansion(cell)

    print(f"  - 확장 성공 여부       : {expansion_res['success']}")
    print(f"  - 이전 내부 질서       : {expansion_res['old_order']}")
    print(f"  - 융합된 거시 질서     : {expansion_res['new_macro_order']}")
    print(f"  - 새로운 차원 (차원확장) : 4D -> {cell.dimension}D")
    print(f"  - 확장된 경계층 반경   : {cell.boundary_layer.boundary_radius:.3f}")
    print(f"  - 확장된 표면장력 용량 : {cell.boundary_layer.retention_capacity}개")

    # -------------------------------------------------------------------------
    # 5. 확장된 거시 세포의 새로운 인과적 공명
    # -------------------------------------------------------------------------
    print("\n[STEP 5] 확장된 거시 경계층 하에서 복합 자극 수용")
    # A composite flux that contains both conservative and dissipative components
    composite_flux = np.zeros(cell.dimension, dtype=np.float64)
    composite_flux[:4] = harmonic_flux
    composite_flux[4:] = -cell.state[4:] * 0.1  # Dissipative component

    res_macro = cell.interact_with_flux(composite_flux)
    print(f"  - 거시 세포 상호작용 결과: {cell.governing_constraint.name}")
    print(f"  - 잔류 마찰            : {res_macro['boundary_friction']:.4f}")
    print(f"  - 최종 상태 Norm       : {np.linalg.norm(cell.state):.4f}")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE: 질서가 질서를 도출하고 외연적으로 확장되는 생명적 고리 검증 완료 ")
    print("=" * 80)


if __name__ == "__main__":
    main()
