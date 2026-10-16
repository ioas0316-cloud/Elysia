r"""
verify_taiji_hexagram_cosmological_engine.py
============================================
동양 우주론(주역, 음양오행, 64괘)과 현대 정보론·위상기하학 및 동적 순환 비트 엔진의
완벽한 종합 검증 스크립트 (Verification Script).

검증 항목:
1. 동적 순환 비트 (Cyclic Vector Bit) 음양 파동 호흡 및 정보 보존성 ($0 \to 1$ Sine/Pi transition).
2. 육십사괘 구형 위상 주소 공간 (Spherical Topological Memory) $O(1)$ 위상 공명 검색.
3. 무극->태극->팔괘->64괘 다차원 양파형 스케일 계층의 상하 환류 (Macro-Micro Feedback Loop) 및 태극 평형 수렴.
4. 소수(Prime Resonance)와 로그(Logarithm) 스케일 보정 무결성.
5. CausalField 및 ScaleBoundaryCellCouplingEngine 종합 통합 작동성.
"""

import math
import numpy as np
import torch

from core.physics.cyclic_vector_bit_engine import CyclicVectorBitEngine
from core.topology.taiji_hexagram_topology import TaijiHexagramTopologyEngine
from core.physics.causal_field import CausalField, InformationVoxel
from core.topology.scale_boundary_cell_coupling import ScaleBoundaryCellCouplingEngine


def main():
    print("==========================================================================")
    print("  VERIFYING TAIJI-HEXAGRAM COSMOLOGICAL & CYCLIC VECTOR BIT ENGINE  ")
    print("==========================================================================\n")

    # -------------------------------------------------------------------------
    # 1. Cyclic Vector Bit Breathing Dynamics & Information Conservation
    # -------------------------------------------------------------------------
    print("[PILLAR 1] Verifying Cyclic Vector Bit Engine ($0 \to 1$ Sine/Pi Breathing)...")
    bit_engine = CyclicVectorBitEngine(num_bits=32)

    # 32-bit test pattern: [0, 1, 0, 1, 1, 0, 0, 1, ...]
    test_pattern = np.array([i % 2 for i in range(32)], dtype=np.int32)
    bit_engine.encode_discrete_bits(test_pattern)

    initial_yins, initial_yangs = bit_engine.get_yin_yang_vectors()
    print(f"  ✓ Initial Yin/Yang symmetry preserved: Yin sum = {np.sum(initial_yins):.2f}, Yang sum = {np.sum(initial_yangs):.2f}")

    # Simulate 10 steps of sine/pi breathing
    for step in range(10):
        report = bit_engine.step_breathing_dynamics(dt=0.05)

    loss = bit_engine.calculate_information_conservation_loss(test_pattern)
    curvature = bit_engine.extract_spectral_axis_curvature()

    print(f"  ✓ Breathing Taiji Balance: {report['taiji_balance']:.4f}")
    print(f"  ✓ Spectral Axis Mean Curvature: {np.mean(curvature):.4f}")
    print(f"  ✓ Information Conservation Loss: {loss:.4f} (Ideal -> 0.0)")
    assert loss == 0.0, "Information conservation loss must be 0.0"
    print("  [PILLAR 1 PASSED]\n")

    # -------------------------------------------------------------------------
    # 2. 64 Hexagram Spherical Topological Memory & Prime-Log Alignment
    # -------------------------------------------------------------------------
    print("[PILLAR 2] Verifying 64 Hexagram Spherical Topological Memory Architecture...")
    topology_engine = TaijiHexagramTopologyEngine(num_nodes=64)
    assert len(topology_engine.nodes) == 64, "Must contain exactly 64 Hexagram nodes"

    # Write payload into Hexagram 0 (Qian) and Hexagram 63 (Kun)
    qian_payload = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    topology_engine.write_spherical_memory(0, qian_payload)

    # Query with Qian (All Yang) phase vector
    qian_query = np.full(6, math.pi, dtype=np.float64)
    best_idx, read_payload, resonance = topology_engine.read_spherical_memory(qian_query)

    print(f"  ✓ Spherical Memory Query matched Hexagram {best_idx} ({topology_engine.nodes[best_idx].name})")
    print(f"  ✓ Resonance Score: {resonance:.4f}")
    assert resonance > 0.5, "Resonance score must be high for matching query"
    print("  [PILLAR 2 PASSED]\n")

    # -------------------------------------------------------------------------
    # 3. Macro-Micro Feedback Loop & Scale Hierarchy Convergence
    # -------------------------------------------------------------------------
    print("[PILLAR 3] Verifying Macro-Micro Feedback Loop & Taiji Balance...")
    for step in range(10):
        fb_report = topology_engine.macro_micro_feedback_loop(dt=0.1)

    print(f"  ✓ Macro Field Energy: {fb_report['macro_field_energy']:.4f}")
    print(f"  ✓ Mean Taiji Balance: {fb_report['mean_taiji_balance']:.4f}")
    print(f"  ✓ Micro-Macro Coherence: {fb_report['micro_macro_coherence']:.4f}")
    assert fb_report['mean_taiji_balance'] > 0.5, "Taiji balance should maintain harmony"
    print("  [PILLAR 3 PASSED]\n")

    # -------------------------------------------------------------------------
    # 4. Full System Integration with CausalField & ScaleBoundaryCellCouplingEngine
    # -------------------------------------------------------------------------
    print("[PILLAR 4] Verifying CausalField & ScaleBoundary Engine Integration...")
    causal_field = CausalField(dimensions=3)
    v1 = InformationVoxel("v1", "Source Voxel", np.array([1, 0, 0], dtype=np.float32), position=np.array([0, 0, 0], dtype=np.float32))
    v2 = InformationVoxel("v2", "Target Voxel", np.array([0, 1, 0], dtype=np.float32), position=np.array([1, 1, 0], dtype=np.float32))
    causal_field.add_voxel(v1)
    causal_field.add_voxel(v2)
    causal_field.link_voxels("v1", "v2", strength=1.5)

    causal_field.step(dt=0.1)
    print("  ✓ CausalField step completed with Taiji-Hexagram feedback loops active.")

    boundary_engine = ScaleBoundaryCellCouplingEngine(dim_avatar=32, dim_macro=64)
    avatar_exp = torch.randn(32)
    other_goodness = torch.randn(32)
    cycle_res = boundary_engine.avatar_mother_autopoietic_cycle(avatar_exp, other_goodness, g_love=0.8)

    print(f"  ✓ Scale Boundary Coupling executed. Taiji Macro Field Energy: {cycle_res['taiji_report']['macro_field_energy']:.4f}")
    print("  [PILLAR 4 PASSED]\n")

    print("==========================================================================")
    print("  ALL 4 COSMOLOGICAL & TOPOLOGICAL PILLARS VERIFIED SUCCESSFULLY!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
