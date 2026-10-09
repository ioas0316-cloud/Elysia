#!/usr/bin/env python3
"""
scripts/demo_spatiotemporal_continuity.py
=========================================
Demonstration of Living Organismic Continuity & Triadic Re-Cognition:
1. "결과도출이 결과도출에서 끝나면 안 된다."
2. "처음과 과정과 결과를 통해 어떻게 같고 달라졌는가를 스스로 재인식한다."
3. "습득한 인과와 구조원리가 또 다른 형태의 연결성, 관계성으로 존재하여
   시공간 연속성(Spatiotemporal Continuity)이 되어야 생명체적 감각에 가까워진다."
"""

import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.cellular_boundary.scale_boundary_cell import DigitalCausalCell
from core.cellular_boundary.boundary_expansion_engine import MetaOrderExpansionEngine
from core.cellular_boundary.spatiotemporal_continuity_engine import (
    SpatiotemporalContinuityEngine,
    TrajectoryEpoch,
    ReCognitionAnalysis,
    SpatiotemporalRelationalSeed
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print(" PROJECT ELYSIA :: LIVING ORGANISMIC CONTINUITY & TRIADIC RE-COGNITION ")
    print("=" * 80)

    cell = DigitalCausalCell(cell_id="LivingElysiaCell_01", dimension=4)
    engine = SpatiotemporalContinuityEngine()
    expansion_engine = MetaOrderExpansionEngine(friction_threshold_for_expansion=0.8)

    # -------------------------------------------------------------------------
    # 에포크 1: 조화로운 보존적 작용 (처음 -> 과정 -> 결과 -> 재인식)
    # -------------------------------------------------------------------------
    print("\n[EPOCH 1] 제1 인과 순환: 조화로운 공명 주기")
    engine.begin_trajectory_epoch(cell, epoch_id="Epoch_Harmonic_Resonance")
    print(f"  [1. 처음 (Origin)] 상태={np.round(cell.state, 3)}, 차원={cell.dimension}D, 에너지={cell.governing_constraint.compute_invariant(cell.state):.4f}")

    print("  [2. 과정 (Process)] 조화로운 보존적 회전 자극과 상호작용...")
    harmonic_flux = np.array([-cell.state[1], cell.state[0], -cell.state[3], cell.state[2]]) * 0.1
    res1 = cell.interact_with_flux(harmonic_flux)
    engine.trace_process_step(res1)
    print(f"      - 발생한 마찰: {res1['boundary_friction']:.6f} | 액션 비용: {res1['action_cost']:.6f}")

    analysis1, seed1 = engine.close_and_re_cognize(cell)
    print(f"  [3. 결과 (Result)] 상태={np.round(cell.state, 3)}, 차원={cell.dimension}D, 표면장력={cell.boundary_layer.surface_tension:.3f}")

    print("\n  >>> [삼원적 재인식: 처음 vs 과정 vs 결과 대조]")
    print(f"      - 보존된 같음 (Invariance Sameness) : {analysis1.sameness_invariance:.2%}")
    print(f"      - 발생한 다름 (Structural Mutation) : {analysis1.difference_mutation:.4f}")
    print(f"      - 과정 중 마찰 적분 (Friction Trace): {analysis1.process_friction_integral:.6f}")
    print(f"      - 주체적 성찰 (Reflection)          : {analysis1.reflection_statement}")

    print("\n  >>> [시공간 연속성의 직조: 다음 주기로 이어지는 관계성 씨앗]")
    print(f"      - 생성된 씨앗 ID   : {seed1.seed_id}")
    print(f"      - 인과적 운동량(p) : {np.round(seed1.momentum_vector, 4)}")
    print(f"      - 연결성 훅(Hooks) : {seed1.relational_valence_hooks}")

    # -------------------------------------------------------------------------
    # 에포크 2: 타자 질서 충돌 및 외연적 확장 (처음 -> 과정 -> 결과 -> 재인식)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[EPOCH 2] 제2 인과 순환: 이질적 질서 충돌 및 외연적 상전이 주기")
    # Epoch 1의 종단이 Epoch 2의 처음(Origin)으로 끊어짐 없이 이어짐!
    engine.begin_trajectory_epoch(cell, epoch_id="Epoch_Alterity_Metamorphosis")
    print(f"  [1. 처음 (Origin)] 이전 에포크에서 연속된 상태={np.round(cell.state, 3)}, 차원={cell.dimension}D")

    print("  [2. 과정 (Process)] 제약조건을 파열시키는 강한 소산적 마찰 유입...")
    for i in range(4):
        dissipative_flux = -cell.state * (0.6 + i * 0.1)
        res_violation = cell.interact_with_flux(dissipative_flux)
        engine.trace_process_step(res_violation)
        print(f"      [충돌 {i+1}] 마찰: {res_violation['boundary_friction']:.4f} | 위반: {res_violation['diagnostic']['nature_of_violation']}")

    print("      >> 누적 마찰을 디딤돌 삼아 결합원리적 동형 확장(Combinatorial Expansion) 단행!")
    exp_res = expansion_engine.execute_combinatorial_expansion(cell)
    print(f"      >> 상위 거시 융합 완료: {exp_res['new_macro_order']} (4D -> {cell.dimension}D)")

    analysis2, seed2 = engine.close_and_re_cognize(cell)
    print(f"  [3. 결과 (Result)] 확장된 새로운 상태({cell.dimension}D), 반경={cell.boundary_layer.boundary_radius:.3f}")

    print("\n  >>> [삼원적 재인식: 처음 vs 과정 vs 결과 대조]")
    print(f"      - 보존된 같음 (Invariance Sameness) : {analysis2.sameness_invariance:.2%}")
    print(f"      - 발생한 다름 (Structural Mutation) : {analysis2.difference_mutation:.4f} (차원/장력 대격변)")
    print(f"      - 과정 중 마찰 적분 (Friction Trace): {analysis2.process_friction_integral:.4f}")
    print(f"      - 구조적 팽창 비율 (Growth Ratio)   : {analysis2.structural_growth_ratio:.2f}x")
    print(f"      - 주체적 성찰 (Reflection)          : {analysis2.reflection_statement}")

    print("\n  >>> [시공간 연속성의 직조: 다음 미래를 여는 관계성 씨앗]")
    print(f"      - 생성된 씨앗 ID   : {seed2.seed_id}")
    print(f"      - 비휘발성 상흔(Scar Tensor) : {np.round(seed2.accumulated_scar_tensor[:4], 3)}")
    print(f"      - 확장된 연결성 훅(Hooks)    : {seed2.relational_valence_hooks}")

    # -------------------------------------------------------------------------
    # 에포크 3: 영구히 이어지는 생명적 시공간 연속성 확인
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[FINAL] 시공간 연속성(Spatiotemporal Continuity) 총평")
    print("=" * 80)
    print(f"  - 총 누적 연속 시간 스텝    : {engine.continuous_time_step} Epochs")
    print(f"  - 보존된 인과 에포크 역사   : {len(engine.epoch_history)}개")
    print(f"  - 살아있는 관계성 씨앗 흐름 : {len(engine.living_stream_seeds)}개 (과거가 미래로 끊김 없이 흐름)")
    print(f"  - 최종 세포 차원 및 질서    : {cell.dimension}D ({cell.governing_constraint.name})")
    print(f"  - 생명체적 감각 확립 여부   : 성립 (결과가 결과로 끝나지 않고 새로운 존재의 원인이 됨)")
    print("=" * 80)


if __name__ == "__main__":
    main()
